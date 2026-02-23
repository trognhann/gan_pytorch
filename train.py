import os
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import set_seed, setup_ddp, cleanup_ddp, load_config, denormalize
from datasets.anime_dataset import AnimeDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.guided_filter import guided_filter
from losses.content import ContentLoss
from losses.style import StyleLoss
from losses.color_lab import ColorLoss
from losses.region_smoothing import RegionSmoothingLoss
from losses.gan_loss import GANLoss


def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANv3 PyTorch Training")
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument(
        '--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(42)
    local_rank, rank, world_size = setup_ddp()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        print(f"[{rank}] Using NVIDIA GPU: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[{rank}] Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print(f"[{rank}] No GPU found. Using CPU")

    is_main_process = (rank == 0)

    use_amp = config['training'].get(
        'use_amp', True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = AnimeDataset(config['training']['dataset_dir'], config['training']
                           ['dataset'], img_size=config['training']['img_size'])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                            sampler=sampler, num_workers=config['training']['num_workers'], drop_last=True)
    if len(dataloader) == 0:
        print("Dataloader yields 0 batches. Ensure your dataset paths are valid.")
        return

    G = Generator(in_channels=3).to(device)
    D = Discriminator(
        in_channels=3, ch=config['model']['ch'], sn=config['model']['sn']).to(device)

    if world_size > 1:
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
        D = DDP(D, device_ids=[local_rank], output_device=local_rank)

    G_net = G.module if isinstance(G, DDP) else G

    optim_G_init = optim.Adam(G.parameters(
    ), lr=config['training']['init_lr'], betas=tuple(config['training']['adam_betas']))
    optim_G = optim.Adam(G.parameters(), lr=config['training']['g_lr'], betas=tuple(
        config['training']['adam_betas']))
    optim_D = optim.Adam(D.parameters(), lr=config['training']['d_lr'], betas=tuple(
        config['training']['adam_betas']))

    content_loss_fn = ContentLoss().to(device)
    style_loss_fn = StyleLoss(
        weights=config['loss_weights']['sty_weight']).to(device)
    color_loss_fn = ColorLoss(
        weight=config['loss_weights']['color_weight']).to(device)
    region_smooth_loss_fn = RegionSmoothingLoss().to(device)
    gan_loss_fn = GANLoss().to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    epochs = config['training']['epochs']
    init_epochs = config['training']['init_epochs']

    log_dir = f"logs/AnimeGANv3_{config['training']['dataset']}"
    if is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
        ckpt_dir = config['training'].get('checkpoint_dir', 'checkpoint')
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    if is_main_process:
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        if is_main_process:
            logger.info(f"Loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=device)
        G_net.load_state_dict(ckpt['G'])
        (D.module if isinstance(D, DDP) else D).load_state_dict(ckpt['D'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        if 'optim_G_init' in ckpt:
            optim_G_init.load_state_dict(ckpt['optim_G_init'])
        if 'scaler' in ckpt and ckpt['scaler'] is not None and use_amp:
            scaler.load_state_dict(ckpt['scaler'])

        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', start_epoch * len(dataloader))
        if is_main_process:
            logger.info(
                f"Loaded checkpoint '{args.resume}' (Resuming from epoch {start_epoch})")

    logger.info(f"[{rank}] Starting training...")

    for epoch in range(start_epoch, epochs):
        if sampler:
            sampler.set_epoch(epoch)

        for idx, batch in enumerate(dataloader):
            real_photo = batch['photo'].to(device)
            photo_seg = batch['photo_seg'].to(device)
            anime_style = batch['style'].to(device)
            anime_smooth = batch['smooth'].to(device)

            # --- Pre-training G ---
            if epoch < init_epochs:
                optim_G_init.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    generated_s, generated_m = G(real_photo, inference=False)
                    # Use guided filter from 0~1 domain scaled back to -1~1
                    gf_input = (generated_s + 1.0) / 2.0
                    generated = (guided_filter(
                        gf_input, gf_input, r=2, eps=0.01) * 2.0) - 1.0

                    init_loss = content_loss_fn(
                        real_photo, generated) + content_loss_fn(real_photo, generated_m)

                scaler.scale(init_loss).backward()
                scaler.step(optim_G_init)
                scaler.update()

                if is_main_process and idx % 10 == 0:
                    logger.info(
                        f"Epoch: {epoch}/{init_epochs} Step: {idx}/{len(dataloader)} Pre_train_G_loss: {init_loss.item():.6f}")
                    writer.add_scalar("Loss/Pre_train_G_loss",
                                      init_loss.item(), global_step)
            else:
                # --- Adversarial Training ---
                optim_G.zero_grad()
                optim_D.zero_grad()

                # --- Update G ---
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    generated_s, generated_m = G(real_photo, inference=False)

                    # 1. GPU Surrogate Teacher via Guided Filter (Fast & Clean)
                    gf_input = (generated_s.detach() + 1.0) / 2.0
                    teacher_l0_approx = (guided_filter(
                        gf_input, gf_input, r=2, eps=0.01) * 2.0) - 1.0

                    # 2. Support Map for Loss (With Gradients)
                    gf_input_grad = (generated_s + 1.0) / 2.0
                    generated = (guided_filter(
                        gf_input_grad, gf_input_grad, r=2, eps=0.01) * 2.0) - 1.0

                    def to_gray_3_ch(x):
                        gray = 0.2125 * x[:, 0:1] + 0.7154 * \
                            x[:, 1:2] + 0.0721 * x[:, 2:3]
                        return gray.repeat(1, 3, 1, 1)

                    anime_sty_gray = to_gray_3_ch(anime_style)
                    fake_sty_gray = to_gray_3_ch(generated)
                    gray_anime_smooth = to_gray_3_ch(anime_smooth)

                    fake_gray_logit = D(fake_sty_gray)
                    generated_m_logit = D(generated_m)

                    # --- Support G Loss ---
                    con_loss = config['loss_weights']['con_weight'] * \
                        content_loss_fn(real_photo, generated)
                    sty_loss = style_loss_fn(anime_sty_gray, fake_sty_gray)

                    # Skip get_seg() online bottleneck. Use VGG Region Loss directly
                    rs_loss = config['loss_weights']['vgg_region_weight'] * \
                        content_loss_fn(photo_seg, generated)

                    color_loss = color_loss_fn(real_photo, generated)
                    tv_loss = config['loss_weights']['tv_weight'] * \
                        gan_loss_fn.tv_loss(generated)
                    g_adv_loss = gan_loss_fn.g_support_adv_loss(
                        fake_gray_logit)

                    G_support_loss = g_adv_loss + con_loss + \
                        sty_loss + rs_loss + color_loss + tv_loss

                    # --- Main G Loss ---
                    tv_loss_m = config['loss_weights']['tv_weight_m'] * \
                        gan_loss_fn.tv_loss(generated_m)

                    # Replacing fake_NLMean_l0 with GPU teacher_l0_approx
                    p4_loss = config['loss_weights']['p4_weight'] * \
                        content_loss_fn(teacher_l0_approx, generated_m)
                    p0_loss = config['loss_weights']['p0_weight'] * \
                        l1_loss_fn(teacher_l0_approx, generated_m)

                    g_m_loss = config['loss_weights']['adv_weight_m'] * \
                        gan_loss_fn.g_main_adv_loss(generated_m_logit)

                    G_main_loss = g_m_loss + p0_loss + p4_loss + tv_loss_m
                    Generator_loss = G_support_loss + G_main_loss

                scaler.scale(Generator_loss).backward()
                scaler.step(optim_G)
                scaler.update()

                # --- Update D ---
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    fake_sty_gray_d = to_gray_3_ch(generated.detach())
                    generated_m_d = generated_m.detach()

                    anime_gray_logit = D(anime_sty_gray)
                    fake_gray_logit_d = D(fake_sty_gray_d)
                    gray_anime_smooth_logit = D(gray_anime_smooth)

                    # Thay fake_NLMean_l0 bằng teacher_l0_approx
                    teacher_l0_logit = D(teacher_l0_approx.detach())
                    generated_m_logit_d = D(generated_m_d)

                    D_support_loss = gan_loss_fn.d_support_loss(
                        anime_gray_logit, fake_gray_logit_d, gray_anime_smooth_logit)
                    D_main_loss = config['loss_weights']['d_main_weight'] * \
                        gan_loss_fn.d_main_loss(
                            teacher_l0_logit, generated_m_logit_d)

                    Discriminator_loss = D_support_loss + D_main_loss

                scaler.scale(Discriminator_loss).backward()
                scaler.step(optim_D)
                scaler.update()

                if is_main_process and idx % 10 == 0:
                    logger.info(
                        f"Epoch: {epoch}/{epochs} Step: {idx}/{len(dataloader)}")
                    logger.info(f"G_loss: {Generator_loss.item():.4f} "
                                f"[Sup: Adv={g_adv_loss.item():.4f}, Con={con_loss.item():.4f}, Sty={sty_loss.item():.4f}, Col={color_loss.item():.4f}, RS={rs_loss.item():.4f}, TV={tv_loss.item():.4f}] "
                                f"[Main: Adv={g_m_loss.item():.4f}, P0={p0_loss.item():.4f}, P4={p4_loss.item():.4f}, TV={tv_loss_m.item():.4f}]")
                    logger.info(f"D_loss: {Discriminator_loss.item():.4f} "
                                f"[Sup={D_support_loss.item():.4f}, Main={D_main_loss.item():.4f}]")
                    writer.add_scalar(
                        "Loss/G", Generator_loss.item(), global_step)
                    writer.add_scalar(
                        "Loss/D", Discriminator_loss.item(), global_step)

                if is_main_process and idx % 100 == 0:
                    import torchvision
                    with torch.no_grad():
                        # Photo | Support | Teacher | Main
                        img_grid = torch.cat([
                            real_photo[:4],
                            generated[:4],
                            teacher_l0_approx[:4],
                            generated_m[:4]
                        ], dim=0)
                        img_grid = (img_grid + 1.0) / 2.0

                        grid = torchvision.utils.make_grid(img_grid, nrow=4)
                        writer.add_image(
                            'Visuals/Photo_Support_Teacher_Main', grid, global_step)

            global_step += 1

        if is_main_process and (epoch + 1) % config['training']['save_freq'] == 0:
            ckpt_dir = config['training'].get('checkpoint_dir', 'checkpoint')
            torch.save({
                'G': G_net.state_dict(),
                'D': D.module.state_dict() if isinstance(D, DDP) else D.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'optim_G_init': optim_G_init.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'epoch': epoch,
                'global_step': global_step
            }, os.path.join(ckpt_dir, f"AnimeGANv3_ep{epoch}.pt"))
            logger.info(
                f"Saved checkpoint to {os.path.join(ckpt_dir, f'AnimeGANv3_ep{epoch}.pt')}")

    if is_main_process:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    main()
