"""
AnimeGANv3 PyTorch — Training Script
=====================================
Two-phase training:
  Phase 1 (init): Generator pre-train with content loss only
  Phase 2 (GAN):  Full adversarial training with dual discriminators

Usage:
  python train.py                           # uses config.yaml defaults
  python train.py --dataset Shinkai --epochs 20
  python train.py --resume checkpoint/latest_G.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import Discriminator
from models.guided_filter import guided_filter
from losses.vgg import VGG19
from losses.content import ContentLoss
from losses.style import StyleLoss
from losses.color_lab import ColorLoss
from losses.gan_loss import GANLoss
from losses.region_smoothing import RegionSmoothingLoss
from tools.dataset import AnimeDataset
from tools.utils import save_checkpoint, load_checkpoint, get_logger, save_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rgb_to_grayscale(x):
    """Convert [-1,1] RGB tensor to 3-channel grayscale (for D_s input)."""
    gray = 0.2125 * x[:, 0:1] + 0.7154 * x[:, 1:2] + 0.0721 * x[:, 2:3]
    return gray.repeat(1, 3, 1, 1)


def r1_penalty(real_pred, real_img):
    """R1 gradient penalty for discriminator regularization."""
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_cfg = cfg['training']
    w_cfg = cfg['loss_weights']
    m_cfg = cfg['model']

    # ── Logger ──────────────────────────────────────────────────────────
    logger = get_logger(t_cfg['checkpoint_dir'], name='train')
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # ── Dataset & DataLoader ────────────────────────────────────────────
    dataset_root = t_cfg['dataset_dir']
    dataset = AnimeDataset(dataset_root, img_size=t_cfg['img_size'][0])
    dataloader = DataLoader(
        dataset,
        batch_size=t_cfg['batch_size'],
        shuffle=True,
        num_workers=t_cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {dataset_root} | photos={dataset.len_photo}, "
                f"animes={dataset.len_anime} | batches/epoch={len(dataloader)}")

    # ── Models ──────────────────────────────────────────────────────────
    G = Generator(in_channels=3).to(device)
    D_s = Discriminator(in_channels=3, ch=m_cfg['ch'], sn=m_cfg['sn']).to(device)
    D_m = Discriminator(in_channels=3, ch=m_cfg['ch'], sn=m_cfg['sn']).to(device)

    # ── Loss modules (share single VGG) ─────────────────────────────────
    vgg = VGG19().to(device).eval()
    content_loss_fn = ContentLoss(weight=1.0, vgg=vgg).to(device)
    style_loss_fn = StyleLoss(weights=w_cfg['sty_weight'], vgg=vgg).to(device)
    color_loss_fn = ColorLoss().to(device)
    gan_loss_fn = GANLoss().to(device)
    region_loss_fn = RegionSmoothingLoss(weight=1.0, vgg=vgg).to(device)
    l1_loss_fn = nn.L1Loss()

    # ── Optimizers ──────────────────────────────────────────────────────
    betas = tuple(t_cfg['adam_betas'])
    opt_G = torch.optim.Adam(G.parameters(), lr=t_cfg['g_lr'], betas=betas)
    opt_D_s = torch.optim.Adam(D_s.parameters(), lr=t_cfg['d_lr'], betas=betas)
    opt_D_m = torch.optim.Adam(D_m.parameters(), lr=t_cfg['d_lr'], betas=betas)
    opt_G_init = torch.optim.Adam(G.parameters(), lr=t_cfg['init_lr'], betas=betas)

    # ── GradScaler (optional AMP) ───────────────────────────────────────
    use_amp = t_cfg.get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ──────────────────────────────────────────────────────────
    ckpt_dir = t_cfg['checkpoint_dir']
    start_epoch, global_step = 0, 0

    g_ckpt = os.path.join(ckpt_dir, 'latest_G.pth')
    ds_ckpt = os.path.join(ckpt_dir, 'latest_D_s.pth')
    dm_ckpt = os.path.join(ckpt_dir, 'latest_D_m.pth')

    if os.path.exists(g_ckpt):
        start_epoch, global_step = load_checkpoint(G, opt_G, g_ckpt)
        logger.info(f"Resumed G from epoch {start_epoch}, step {global_step}")
    if os.path.exists(ds_ckpt):
        load_checkpoint(D_s, opt_D_s, ds_ckpt)
        logger.info("Resumed D_s")
    if os.path.exists(dm_ckpt):
        load_checkpoint(D_m, opt_D_m, dm_ckpt)
        logger.info("Resumed D_m")

    # =====================================================================
    #  PHASE 1 — Init Pre-train (Generator content loss only)
    # =====================================================================
    if start_epoch < t_cfg['init_epochs']:
        logger.info("=" * 60)
        logger.info("PHASE 1: Generator pre-training (content loss)")
        logger.info("=" * 60)

        for epoch in range(start_epoch, t_cfg['init_epochs']):
            G.train()
            pbar = tqdm(dataloader, desc=f"[Init {epoch+1}/{t_cfg['init_epochs']}]")

            for i, (photo, anime, smooth, photo_sp) in enumerate(pbar):
                photo = photo.to(device, non_blocking=True)

                opt_G_init.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, fake_m = G(photo)
                    loss_con = content_loss_fn(photo, fake_m)

                scaler.scale(loss_con).backward()
                scaler.unscale_(opt_G_init)
                nn.utils.clip_grad_norm_(G.parameters(), 5.0)
                scaler.step(opt_G_init)
                scaler.update()

                global_step += 1
                if i % 50 == 0:
                    pbar.set_postfix(loss_con=f"{loss_con.item():.4f}")

            # Save at end of each init epoch
            logger.info(f"[Init Epoch {epoch+1}] loss_con={loss_con.item():.4f}")
            save_checkpoint(G, opt_G_init, epoch + 1, global_step, ckpt_dir, 'latest_G')

            # Save sample
            with torch.no_grad():
                sample_m = G(photo[:4], inference=True)
            save_images(sample_m, os.path.join(ckpt_dir, f'init_epoch{epoch+1}.png'), nrow=4)

        start_epoch = t_cfg['init_epochs']
        # Switch to GAN lr
        for pg in opt_G.param_groups:
            pg['lr'] = t_cfg['g_lr']

    # =====================================================================
    #  PHASE 2 — Full GAN Training
    # =====================================================================
    total_epochs = t_cfg['init_epochs'] + t_cfg['epochs']
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Full GAN training (epochs {start_epoch+1}→{total_epochs})")
    logger.info("=" * 60)

    d_update_ratio = w_cfg.get('d_update_ratio', 1)

    for epoch in range(start_epoch, total_epochs):
        G.train()
        D_s.train()
        D_m.train()

        pbar = tqdm(dataloader,
                    desc=f"[GAN {epoch+1}/{total_epochs}]")

        for i, (photo, anime, smooth, photo_sp) in enumerate(pbar):
            photo = photo.to(device, non_blocking=True)
            anime = anime.to(device, non_blocking=True)
            smooth = smooth.to(device, non_blocking=True)
            photo_sp = photo_sp.to(device, non_blocking=True)

            # ==============================================================
            #  D step  (repeat d_update_ratio times per G step)
            # ==============================================================
            for _ in range(d_update_ratio):
                # ── Generate fakes (no grad for G) ───────────────────────
                with torch.no_grad():
                    fake_s_raw, fake_m = G(photo)
                    fake_si = guided_filter(photo, fake_s_raw, r=1)

                # ── D_support (grayscale domain) ─────────────────────────
                anime_gray = rgb_to_grayscale(anime)
                fake_gray = rgb_to_grayscale(fake_si)
                smooth_gray = rgb_to_grayscale(smooth)

                opt_D_s.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    d_s_anime = D_s(anime_gray)
                    d_s_fake = D_s(fake_gray)
                    d_s_smooth = D_s(smooth_gray)
                    loss_D_s = gan_loss_fn.d_support_loss(d_s_anime, d_s_fake, d_s_smooth)
                    loss_D_s = loss_D_s * w_cfg.get('d_smooth_weight', 1.0)

                scaler.scale(loss_D_s).backward()
                scaler.step(opt_D_s)
                scaler.update()

                # ── D_main (RGB domain) ──────────────────────────────────
                anime_for_dm = anime.requires_grad_(True)
                opt_D_m.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    d_m_real = D_m(anime_for_dm)
                    d_m_fake = D_m(fake_m)
                    loss_D_m = gan_loss_fn.d_main_loss(d_m_real, d_m_fake)
                    loss_D_m = loss_D_m * w_cfg.get('d_main_weight', 1.0)

                    # R1 penalty
                    r1 = r1_penalty(d_m_real, anime_for_dm)
                    loss_D_m_total = loss_D_m + w_cfg.get('r1_weight', 1.0) * r1

                scaler.scale(loss_D_m_total).backward()
                scaler.step(opt_D_m)
                scaler.update()

            # ==============================================================
            #  G step
            # ==============================================================
            opt_G.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                fake_s_raw, fake_m = G(photo)
                fake_si = guided_filter(photo, fake_s_raw, r=1)

                # ── Support tail losses ──────────────────────────────────
                fake_si_gray = rgb_to_grayscale(fake_si)
                g_adv_s = w_cfg['adv_weight'] * gan_loss_fn.g_support_adv_loss(D_s(fake_si_gray))
                g_con = w_cfg['con_weight'] * content_loss_fn(photo, fake_si)
                g_sty = style_loss_fn(anime, fake_si)
                g_color = w_cfg['color_weight'] * color_loss_fn(fake_si, photo)
                g_tv_s = w_cfg['tv_weight'] * gan_loss_fn.tv_loss(fake_si)
                g_region = w_cfg['region_smooth_weight'] * region_loss_fn(photo_sp, fake_si)

                loss_G_support = g_adv_s + g_con + g_sty + g_color + g_tv_s + g_region

                # ── Main tail losses ─────────────────────────────────────
                g_adv_m = w_cfg['adv_weight_m'] * gan_loss_fn.g_main_adv_loss(D_m(fake_m))
                g_pixel = w_cfg['p0_weight'] * l1_loss_fn(fake_m, photo)
                g_percep = w_cfg['p4_weight'] * content_loss_fn(photo, fake_m)
                g_tv_m = w_cfg['tv_weight_m'] * gan_loss_fn.tv_loss(fake_m)

                loss_G_main = g_adv_m + g_pixel + g_percep + g_tv_m

                # ── Total G loss ─────────────────────────────────────────
                loss_G = loss_G_support + loss_G_main

            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt_G)
            scaler.update()

            global_step += 1

            # ── Logging ──────────────────────────────────────────────────
            if i % 50 == 0:
                pbar.set_postfix(
                    G=f"{loss_G.item():.3f}",
                    Ds=f"{loss_D_s.item():.3f}",
                    Dm=f"{loss_D_m.item():.3f}",
                )
                logger.info(
                    f"E{epoch+1} S{global_step} | "
                    f"G={loss_G.item():.4f} "
                    f"[adv_s={g_adv_s.item():.3f} con={g_con.item():.3f} "
                    f"sty={g_sty.item():.3f} color={g_color.item():.3f} "
                    f"tv_s={g_tv_s.item():.3f} region={g_region.item():.3f} | "
                    f"adv_m={g_adv_m.item():.3f} pix={g_pixel.item():.3f} "
                    f"percep={g_percep.item():.3f} tv_m={g_tv_m.item():.3f}] | "
                    f"D_s={loss_D_s.item():.4f} D_m={loss_D_m.item():.4f} "
                    f"R1={r1.item():.4f}"
                )

        # ── End of epoch ─────────────────────────────────────────────────
        logger.info(f"[Epoch {epoch+1}/{total_epochs}] completed. step={global_step}")

        # Save checkpoints
        if (epoch + 1) % t_cfg['save_freq'] == 0 or (epoch + 1) == total_epochs:
            save_checkpoint(G, opt_G, epoch + 1, global_step, ckpt_dir, 'latest_G')
            save_checkpoint(D_s, opt_D_s, epoch + 1, global_step, ckpt_dir, 'latest_D_s')
            save_checkpoint(D_m, opt_D_m, epoch + 1, global_step, ckpt_dir, 'latest_D_m')

            # Also save epoch-specific for G
            save_checkpoint(G, opt_G, epoch + 1, global_step, ckpt_dir, f'G_epoch{epoch+1}')
            logger.info(f"  → Checkpoints saved.")

        # Save sample images
        with torch.no_grad():
            G.eval()
            sample_photos = photo[:4]
            sample_m = G(sample_photos, inference=True)
            grid = torch.cat([sample_photos, sample_m], dim=0)
            save_images(grid, os.path.join(ckpt_dir, f'sample_epoch{epoch+1}.png'), nrow=4)
            G.train()

    logger.info("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='AnimeGANv3 PyTorch Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Override dataset root directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of GAN training epochs')
    parser.add_argument('--init_epochs', type=int, default=None,
                        help='Override number of init pre-train epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Override checkpoint directory')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Override training image size')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override from CLI args
    if args.dataset_dir:
        cfg['training']['dataset_dir'] = args.dataset_dir
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.init_epochs is not None:
        cfg['training']['init_epochs'] = args.init_epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.checkpoint_dir:
        cfg['training']['checkpoint_dir'] = args.checkpoint_dir
    if args.img_size is not None:
        cfg['training']['img_size'] = [args.img_size, args.img_size]

    train(cfg)


if __name__ == '__main__':
    main()
