import torch
from models.generator import Generator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cpu')
    print("Testing Generator Architecture...")

    # 1. Parameter count check
    G = Generator(in_channels=3).to(device)
    params = count_parameters(G)
    print(f"Total Parameters in Generator: {params} (~{params/1e6:.2f}M)")

    # 2. Shape consistency check
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # Training pass
    fake_s, fake_m = G(dummy_input, inference=False)
    print(f"Training Forward - Support Tail Shape: {fake_s.shape}")
    print(f"Training Forward - Main Tail Shape: {fake_m.shape}")

    # Inference pass
    fake_inf = G(dummy_input, inference=True)
    print(f"Inference Forward - Output Shape: {fake_inf.shape}")

    print(
        "\nVerification successful if parameters are ~1.02M and shapes match input [B, 3, 256, 256].")


if __name__ == "__main__":
    main()
