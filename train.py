import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

from net.generator import Generator
from net.discriminator import Discriminator
from tools.dataset import AnimeDataset
from tools.utils import save_checkpoint, load_checkpoint, get_logger, save_images, denormalize
from losses import AnimeGANLoss, ColorLoss


def main():
    parser = argparse.ArgumentParser(description='AnimeGANv3 Training')
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
    parser.add_argument('--wadv', type=float, default=10.0,
                        help='Adversarial loss weight')
    parser.add_argument('--wcon', type=float, default=1.5,
                        help='Content loss weight')
    parser.add_argument('--wsty', type=float, default=3.0,
                        help='Style loss weight')
    parser.add_argument('--wcol', type=float, default=10.0,
                        help='Color loss weight')
    parser.add_argument('--wtv', type=float, default=1.0,
                        help='Total variation loss weight')
    parser.add_argument('--vgg_path', type=str,
                        default='vgg19_weight/vgg19_no_fc.npy', help='Path to vgg19_no_fc.npy')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup directories
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logger = get_logger(args.log_dir)

    # Data
    dataset = AnimeDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    # Models
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Losses
    loss_fn = AnimeGANLoss(device, args.vgg_path)
    color_loss_fn = ColorLoss()
    l1_loss = torch.nn.L1Loss()

    # Optimizers
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    start_step = 0

    # Resume
    if args.resume:
        e1, s1 = load_checkpoint(netG, optG, os.path.join(
            args.checkpoint_dir, 'latest_netG.pth'))
        e2, s2 = load_checkpoint(netD, optD, os.path.join(
            args.checkpoint_dir, 'latest_netD.pth'))
        start_epoch = max(e1, e2)
        start_step = max(s1, s2)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Initialization Phase (Content Loss only)
    if start_epoch < args.init_epochs:
        logger.info("Starting initialization phase...")
        for epoch in range(start_epoch, args.init_epochs):
            pbar = tqdm(dataloader)
            for i, (photo, _, _) in enumerate(pbar):
                photo = photo.to(device)

                optG.zero_grad()
                # Use Main Tail for init? or both? Usually just Main.
                # Inference mode returns main tail only
                generated = netG(photo, training=False)

                loss_init = args.wcon * loss_fn.content_loss(generated, photo)
                loss_init.backward()
                optG.step()

                pbar.set_description(
                    f"[Init] Epoch {epoch}/{args.init_epochs} Loss: {loss_init.item():.4f}")

            save_checkpoint(netG, optG, epoch, 0,
                            args.checkpoint_dir, 'latest_netG')

    # Main Training Phase
    logger.info("Starting GAN training phase...")
    total_steps = len(dataloader)

    # Adjust start epoch if we did init
    gan_start_epoch = max(start_epoch, args.init_epochs)

    for epoch in range(gan_start_epoch, args.epochs):
        pbar = tqdm(dataloader)
        for i, (photo, anime, smooth) in enumerate(pbar):
            current_step = i

            photo = photo.to(device)
            anime = anime.to(device)
            smooth = smooth.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optD.zero_grad()

            # Fake
            fake_main, fake_support = netG(photo, training=True)

            # D Loss
            # Real Anime
            pred_real = netD(anime)
            loss_d_real = loss_fn.mse_loss(
                pred_real, torch.ones_like(pred_real))

            # Fake Anime (Main Detached)
            pred_fake = netD(fake_main.detach())
            loss_d_fake = loss_fn.mse_loss(
                pred_fake, torch.zeros_like(pred_fake))

            # Gray/Smooth Anime (Semi-Supervised/Texture Loss) - Optional but good for avoiding artifacts
            # Often treated as 'fake' (label 0) to force G to produce clean edges?
            # Or treated as 'real' but simpler?
            # In AnimeGANv2, 'gray' is used as 'fake' for texture?
            # Let's stick to standard GAN loss for simplicity: Real=1, Fake=0.
            # We can use smooth as a "clean" reference if needed.
            # For simplicity:
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optG.zero_grad()

            # Re-generate (to get gradients) or just use existing graph if retain_graph=True
            # But we detached D inputs, so we need to forward G again or keep graph?
            # Simpler to re-forward or use scalar tensors.
            # We want gradients for G.
            # fake_main, fake_support = netG(photo) -- computed above, graph for G is still alive?
            # Yes, we only detached for D.

            # Adversarial Loss (Main)
            pred_fake_g = netD(fake_main)
            loss_g_adv_main = loss_fn.mse_loss(
                pred_fake_g, torch.ones_like(pred_fake_g))

            # Content Loss
            loss_g_con_main = loss_fn.content_loss(fake_main, photo)
            loss_g_con_support = loss_fn.content_loss(fake_support, photo)

            # Style Loss (Main) - using Gram Matrix of Anime
            loss_g_sty_main = loss_fn.style_loss(fake_main, anime)

            # Color Loss
            loss_g_col_main = color_loss_fn(fake_main, photo)

            # TV Loss
            loss_g_tv_main = loss_fn.variation_loss(fake_main)

            # Total G Loss
            loss_g = (args.wadv * loss_g_adv_main) + \
                     (args.wcon * loss_g_con_main) + \
                     (args.wcon * loss_g_con_support) + \
                     (args.wsty * loss_g_sty_main) + \
                     (args.wcol * loss_g_col_main) + \
                     (args.wtv * loss_g_tv_main)

            loss_g.backward()
            optG.step()

            # Logging
            if i % 100 == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{total_steps}] "
                            f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")
                logger.info(
                    f"Loss_G Adv: {loss_g_adv_main.item():.4f} Loss_G Con: {loss_g_con_main.item():.4f} Loss_G Sty: {loss_g_sty_main.item():.4f} Loss_G Col: {loss_g_col_main.item():.4f} Loss_G TV: {loss_g_tv_main.item():.4f}")

        # Save Model
        save_checkpoint(netG, optG, epoch, 0,
                        args.checkpoint_dir, 'latest_netG')
        save_checkpoint(netD, optD, epoch, 0,
                        args.checkpoint_dir, 'latest_netD')

        # Save Sample
        if epoch % 1 == 0:
            with torch.no_grad():
                sample = netG(photo, training=False)
                save_images(sample, os.path.join(
                    args.checkpoint_dir, f'epoch_{epoch}.jpg'))

    logger.info("Training Finished.")


if __name__ == '__main__':
    main()
