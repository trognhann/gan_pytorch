import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
import numpy as np
import cv2

from net.generator import Generator
from net.discriminator import Discriminator
from tools.dataset import AnimeDataset
from tools.utils import save_checkpoint, load_checkpoint, get_logger, save_images, denormalize
from losses import AnimeGANLoss, GuidedFilter


def main():
    parser = argparse.ArgumentParser(description='AnimeGANv3 Training')
    # ... (args unchanged)
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_epochs', type=int, default=5,
                        help='Epochs for generator initialization')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logger = get_logger(args.log_dir)

    dataset = AnimeDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    netG = Generator().to(device)
    netDm = Discriminator(input_nc=3).to(device)
    netDs = Discriminator(input_nc=1).to(device)

    loss_fn = AnimeGANLoss(device)
    # ColorLoss is now integrated into AnimeGANLoss
    l1_loss = torch.nn.L1Loss()
    # Increased strength for Teacher-Student target generation (slower but cleaner)
    guided_filter = GuidedFilter(r=4, eps=1e-2).to(device)

    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optDm = optim.Adam(netDm.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optDs = optim.Adam(netDs.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        e1, s1 = load_checkpoint(netG, optG, os.path.join(
            args.checkpoint_dir, 'latest_netG.pth'))
        e2, s2 = load_checkpoint(netDm, optDm, os.path.join(
            args.checkpoint_dir, 'latest_netDm.pth'))
        e3, s3 = load_checkpoint(netDs, optDs, os.path.join(
            args.checkpoint_dir, 'latest_netDs.pth'))
        start_epoch = max(e1, e2, e3)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Initialization
    if start_epoch < args.init_epochs:
        logger.info("Starting initialization phase...")
        for epoch in range(start_epoch, args.init_epochs):
            pbar = tqdm(dataloader)
            for i, data in enumerate(pbar):
                photo = data[0].to(device)
                optG.zero_grad()
                generated = netG(photo, training=False)
                # Init with Content Loss
                loss_init = 1.5 * loss_fn.content_loss(generated, photo)
                loss_init.backward()
                optG.step()
                pbar.set_description(
                    f"[Init] Epoch {epoch}/{args.init_epochs} Loss: {loss_init.item():.4f}")
            save_checkpoint(netG, optG, epoch, 0,
                            args.checkpoint_dir, 'latest_netG')

    # GAN Training
    logger.info("Starting GAN training phase...")
    gan_start_epoch = max(start_epoch, args.init_epochs)

    for epoch in range(gan_start_epoch, args.epochs):
        pbar = tqdm(dataloader)
        for i, (photo, anime, smooth, photo_smooth) in enumerate(pbar):
            photo = photo.to(device)
            anime = anime.to(device)
            smooth = smooth.to(device)
            photo_smooth = photo_smooth.to(device)

            # Use loss_fn.rgb_to_gray
            anime_gray = loss_fn.rgb_to_gray(anime)

            fake_main, fake_support = netG(photo, training=True)
            fake_support_gray = loss_fn.rgb_to_gray(fake_support)

            # ---------------------
            #  Train Discriminators
            # ---------------------
            optDm.zero_grad()
            optDs.zero_grad()

            # Main D
            pred_real_m = netDm(anime)
            pred_fake_m = netDm(fake_main.detach())
            loss_dm = loss_fn.mse_loss(pred_real_m, torch.ones_like(pred_real_m)) + \
                loss_fn.mse_loss(pred_fake_m, torch.zeros_like(pred_fake_m))
            loss_dm.backward()
            optDm.step()

            # Support D
            pred_real_s = netDs(anime_gray)
            pred_fake_s = netDs(fake_support_gray.detach())
            loss_ds = loss_fn.mse_loss(pred_real_s, torch.ones_like(pred_real_s)) + \
                loss_fn.mse_loss(pred_fake_s, torch.zeros_like(pred_fake_s))
            loss_ds.backward()
            optDs.step()

            # -----------------
            #  Train Generator
            # -----------------
            optG.zero_grad()

            # --- SUPPORT TAIL LOSS ---
            # Lambda_con=0.5, Lambda_col=20, Lambda_tv=0.001
            # Assuming Adv weight is 0.1 or 1.0?
            # We keep it 0.1 to avoid overpowering.

            l_con_s = loss_fn.content_loss(fake_support, photo)
            l_col_s = loss_fn.color_loss(
                fake_support, photo)  # Use loss_fn.color_loss
            l_tv_s = loss_fn.variation_loss(fake_support)

            pred_fake_s = netDs(fake_support_gray)
            l_adv_s = loss_fn.mse_loss(
                pred_fake_s, torch.ones_like(pred_fake_s))

            loss_support = (0.5 * l_con_s) + \
                           (20.0 * l_col_s) + \
                           (0.001 * l_tv_s) + \
                           (0.1 * l_adv_s)

            # --- MAIN TAIL LOSS ---
            # Eta_pp=50 (Per-Pixel/Color), Eta_per=0.5 (Perceptual/Content), Eta_adv=0.02, Eta_tv=0.001

            l_adv_m = loss_fn.mse_loss(
                netDm(fake_main), torch.ones_like(pred_fake_m))
            l_con_m = loss_fn.content_loss(fake_main, photo)  # Perception
            l_col_m = loss_fn.color_loss(fake_main, photo)  # Per-pixel?
            l_sty_m = loss_fn.style_loss(fake_main, anime)
            l_tv_m = loss_fn.variation_loss(fake_main)

            # Teacher Student
            with torch.no_grad():
                teacher_feature = guided_filter(photo, fake_support.detach())
            l_teacher = l1_loss(fake_main, teacher_feature)

            # Region Smoothing
            l_rs = loss_fn.content_loss(fake_main, photo_smooth)

            # Weights mapped
            loss_main = (0.02 * l_adv_m) + \
                        (0.5 * l_con_m) + \
                        (50.0 * l_col_m) + \
                        (0.001 * l_tv_m) + \
                        (3.0 * l_sty_m) + \
                        (1.0 * l_teacher) + \
                        (1.0 * l_rs)

            loss_g = loss_support + loss_main

            loss_g.backward()
            optG.step()

            if i % 50 == 0:
                logger.info(f"Ep [{epoch}] S [{i}] "
                            f"Dm: {loss_dm.item():.3f} Ds: {loss_ds.item():.3f} G: {loss_g.item():.3f} "
                            f"Sup(Con:{l_con_s:.2f} Col:{l_col_s:.2f}) "
                            f"Main(Adv:{l_adv_m:.2f} Con:{l_con_m:.2f} Col:{l_col_m:.2f} Sty:{l_sty_m:.2f})")

        save_checkpoint(netG, optG, epoch, 0,
                        args.checkpoint_dir, 'latest_netG')
        save_checkpoint(netDm, optDm, epoch, 0,
                        args.checkpoint_dir, 'latest_netDm')
        save_checkpoint(netDs, optDs, epoch, 0,
                        args.checkpoint_dir, 'latest_netDs')

        if epoch % 1 == 0:
            with torch.no_grad():
                sample = netG(photo, training=False)
                save_images(sample, os.path.join(
                    args.checkpoint_dir, f'epoch_{epoch}.jpg'))

    logger.info("Training Finished.")


if __name__ == '__main__':
    main()
