import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import set_seed, setup_ddp, cleanup_ddp, load_config, get_seg, get_NLMean_l0, denormalize
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

    if is_main_process:
        writer = SummaryWriter(
            log_dir=f"logs/AnimeGANv3_{config['training']['dataset']}")
        ckpt_dir = config['training'].get('checkpoint_dir', 'checkpoint')
        os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    print(f"[{rank}] Starting training...")

    for epoch in range(epochs):
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
                with torch.cuda.amp.autocast(enabled=use_amp):
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
                    print(
                        f"Epoch: {epoch}/{init_epochs} Step: {idx}/{len(dataloader)} Pre_train_G_loss: {init_loss.item():.6f}")
                    writer.add_scalar("Loss/Pre_train_G_loss",
                                      init_loss.item(), global_step)
            else:
                # --- Adversarial Training ---
                optim_G.zero_grad()
                optim_D.zero_grad()

                # --- Update G ---
                with torch.cuda.amp.autocast(enabled=use_amp):
                    generated_s, generated_m = G(real_photo, inference=False)
                    gf_input = (generated_s + 1.0) / 2.0
                    generated = (guided_filter(
                        gf_input, gf_input, r=2, eps=0.01) * 2.0) - 1.0

                # Numpy revisions
                gen_np = generated.detach().cpu().numpy().transpose(0, 2, 3, 1)
                gen_s_np = generated_s.detach().cpu().numpy().transpose(0, 2, 3, 1)

                fake_superpixel_np = get_seg(gen_np)
                fake_NLMean_l0_np = get_NLMean_l0(gen_s_np)

                fake_superpixel = torch.from_numpy(
                    fake_superpixel_np).permute(0, 3, 1, 2).to(device)
                fake_NLMean_l0 = torch.from_numpy(
                    fake_NLMean_l0_np).permute(0, 3, 1, 2).to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    def to_gray_3_ch(x):
                        gray = 0.2125 * x[:, 0:1] + 0.7154 * \
                            x[:, 1:2] + 0.0721 * x[:, 2:3]
                        return gray.repeat(1, 3, 1, 1)

                    anime_sty_gray = to_gray_3_ch(anime_style)
                    fake_sty_gray = to_gray_3_ch(generated)
                    gray_anime_smooth = to_gray_3_ch(anime_smooth)

                    fake_gray_logit = D(fake_sty_gray)
                    generated_m_logit = D(generated_m)

                    # Support G Loss
                    con_loss = config['loss_weights']['con_weight'] * \
                        content_loss_fn(real_photo, generated)
                    sty_loss = sum(style_loss_fn(
                        anime_sty_gray, fake_sty_gray))

                    rs_loss = config['loss_weights']['region_smooth_weight'] * region_smooth_loss_fn(fake_superpixel, generated, 1.0) \
                        + config['loss_weights']['vgg_region_weight'] * \
                        content_loss_fn(photo_seg, generated)

                    color_loss = color_loss_fn(real_photo, generated)
                    tv_loss = config['loss_weights']['tv_weight'] * \
                        gan_loss_fn.tv_loss(generated)
                    g_adv_loss = gan_loss_fn.g_support_adv_loss(
                        fake_gray_logit)

                    G_support_loss = g_adv_loss + con_loss + \
                        sty_loss + rs_loss + color_loss + tv_loss

                    # Main G Loss
                    tv_loss_m = config['loss_weights']['tv_weight_m'] * \
                        gan_loss_fn.tv_loss(generated_m)
                    p4_loss = config['loss_weights']['p4_weight'] * \
                        content_loss_fn(fake_NLMean_l0, generated_m)
                    p0_loss = config['loss_weights']['p0_weight'] * \
                        l1_loss_fn(fake_NLMean_l0, generated_m)
                    g_m_loss = config['loss_weights']['adv_weight_m'] * \
                        gan_loss_fn.g_main_adv_loss(generated_m_logit)

                    G_main_loss = g_m_loss + p0_loss + p4_loss + tv_loss_m
                    Generator_loss = G_support_loss + G_main_loss

                scaler.scale(Generator_loss).backward()
                scaler.step(optim_G)
                scaler.update()

                # --- Update D ---
                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_sty_gray_d = to_gray_3_ch(generated.detach())
                    generated_m_d = generated_m.detach()

                    anime_gray_logit = D(anime_sty_gray)
                    fake_gray_logit_d = D(fake_sty_gray_d)
                    gray_anime_smooth_logit = D(gray_anime_smooth)

                    fake_NLMean_logit = D(fake_NLMean_l0)
                    generated_m_logit_d = D(generated_m_d)

                    D_support_loss = gan_loss_fn.d_support_loss(
                        anime_gray_logit, fake_gray_logit_d, gray_anime_smooth_logit)
                    D_main_loss = config['loss_weights']['d_main_weight'] * \
                        gan_loss_fn.d_main_loss(
                            fake_NLMean_logit, generated_m_logit_d)

                    Discriminator_loss = D_support_loss + D_main_loss

                scaler.scale(Discriminator_loss).backward()
                scaler.step(optim_D)
                scaler.update()

                if is_main_process and idx % 10 == 0:
                    print(
                        f"Epoch: {epoch}/{epochs} Step: {idx}/{len(dataloader)}")
                    print(
                        f"G_loss: {Generator_loss.item():.4f} D_loss: {Discriminator_loss.item():.4f}")
                    writer.add_scalar(
                        "Loss/G", Generator_loss.item(), global_step)
                    writer.add_scalar(
                        "Loss/D", Discriminator_loss.item(), global_step)

            global_step += 1

        if is_main_process and (epoch + 1) % config['training']['save_freq'] == 0:
            ckpt_dir = config['training'].get('checkpoint_dir', 'checkpoint')
            torch.save({
                'G': G_net.state_dict(),
                'D': D.module.state_dict() if isinstance(D, DDP) else D.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'epoch': epoch
            }, os.path.join(ckpt_dir, f"AnimeGANv3_ep{epoch}.pt"))

    if is_main_process:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    main()
