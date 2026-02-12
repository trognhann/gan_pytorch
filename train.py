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
from losses import AnimeGANLoss, ColorLoss, GuidedFilter


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
    parser.add_argument('--wadv', type=float, default=10.0)
    parser.add_argument('--wcon', type=float, default=1.5)
    parser.add_argument('--wsty', type=float, default=3.0)
    parser.add_argument('--wcol', type=float, default=10.0)
    parser.add_argument('--wtv', type=float, default=1.0)
    parser.add_argument('--vgg_path', type=str,
                        default='vgg19_weight/vgg19_no_fc.npy', help='Path to vgg19_no_fc.npy')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logger = get_logger(args.log_dir)

    dataset = AnimeDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    loss_fn = AnimeGANLoss(device, args.vgg_path)
    color_loss_fn = ColorLoss()
    l1_loss = torch.nn.L1Loss()
    guided_filter = GuidedFilter(r=1, eps=5e-3).to(device)  # For refinement

    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        e1, s1 = load_checkpoint(netG, optG, os.path.join(
            args.checkpoint_dir, 'latest_netG.pth'))
        e2, s2 = load_checkpoint(netD, optD, os.path.join(
            args.checkpoint_dir, 'latest_netD.pth'))
        start_epoch = max(e1, e2)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Initialization
    if start_epoch < args.init_epochs:
        logger.info("Starting initialization phase...")
        for epoch in range(start_epoch, args.init_epochs):
            pbar = tqdm(dataloader)
            for i, (photo, _, _) in enumerate(pbar):
                photo = photo.to(device)
                optG.zero_grad()
                generated = netG(photo, training=False)
                loss_init = args.wcon * loss_fn.content_loss(generated, photo)
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
        for i, (photo, anime, smooth) in enumerate(pbar):
            photo = photo.to(device)
            anime = anime.to(device)
            smooth = smooth.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optD.zero_grad()

            # Generate
            fake_main, fake_support = netG(photo, training=True)

            # Predict
            pred_real = netD(anime)
            # Main tail only for G-D interaction?
            pred_fake = netD(fake_main.detach())

            # Real/Fake Loss
            loss_d_real = loss_fn.mse_loss(
                pred_real, torch.ones_like(pred_real))
            loss_d_fake = loss_fn.mse_loss(
                pred_fake, torch.zeros_like(pred_fake))

            # Region Smoothing / Gray Loss (Optional: Treat gray/smooth as fake or real with different label?)
            # AnimeGANv3 often relies on simple LSGAN: Real=1 (Anime), Fake=0 (Generated)
            # Some versions treat 'Smooth' as 'Fake' to enforce edge preservation?
            # Let's keep it simple: Real Anime vs Fake Main.

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optG.zero_grad()

            # 1. Main Tail Adversarial (Fool Discriminator)
            pred_fake_g = netD(fake_main)
            loss_g_adv = loss_fn.mse_loss(
                pred_fake_g, torch.ones_like(pred_fake_g))

            # 2. Content Loss (VGG)
            loss_g_con = loss_fn.content_loss(fake_main, photo) + \
                loss_fn.content_loss(fake_support, photo)

            # 3. Style Loss (Grayscale Gram Matrix)
            loss_g_sty = loss_fn.style_loss(fake_main, anime)

            # 4. Color Loss (Lab Space)
            loss_g_col = color_loss_fn(fake_main, photo)

            # 5. Teacher-Student Loss (Fine-grained Revision)
            # Teacher: Support Tail output processed by Guided Filter (or just Support Output?)
            # Prompt: "Teacher-Student relationship... Support Tail -> Guided Filter -> Target for Main Tail"
            # Logic: Main Tail should look like Support Tail but refined.
            # Filter output of support tail to remove artifacts? GuidedFilter(support, support)?
            # Or GuidedFilter(support, photo) -> using photo as guide?
            # Typically: Refined = GuidedFilter(Support, Photo)
            # Loss = L1(Main, Refined)

            # Let's use Photo as guidance structure for Support Image content
            # x: guidance (photo), y: input (support)
            with torch.no_grad():
                # Normalize photo to [0,1] for guidance if filter expects it?
                # Our filter is generic.
                refined_target = guided_filter(photo, fake_support.detach())

            loss_g_teacher = l1_loss(
                fake_main, refined_target) * 0.5  # Weight?
            # Or is this "Fine Grained Revision Loss"?

            # 6. TV Loss
            loss_g_tv = loss_fn.variation_loss(fake_main)

            # Total Loss
            loss_g = (args.wadv * loss_g_adv) + \
                     (args.wcon * loss_g_con) + \
                     (args.wsty * loss_g_sty) + \
                     (args.wcol * loss_g_col) + \
                     (args.wtv * loss_g_tv) + \
                     (1.0 * loss_g_teacher)  # Add revision loss

            loss_g.backward()
            optG.step()

            if i % 100 == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(dataloader)}] "
                            f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} "
                            f"Adv: {loss_g_adv.item():.4f} Teach: {loss_g_teacher.item():.4f}")

        save_checkpoint(netG, optG, epoch, 0,
                        args.checkpoint_dir, 'latest_netG')
        save_checkpoint(netD, optD, epoch, 0,
                        args.checkpoint_dir, 'latest_netD')

        if epoch % 1 == 0:
            with torch.no_grad():
                sample = netG(photo, training=False)
                save_images(sample, os.path.join(
                    args.checkpoint_dir, f'epoch_{epoch}.jpg'))

    logger.info("Training Finished.")


if __name__ == '__main__':
    main()
