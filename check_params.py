from net.generator import Generator
from net.discriminator import Discriminator
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


g = Generator()
d = Discriminator()

print(f"Generator Parameters: {count_parameters(g)/1e6:.2f}M")
print(f"Discriminator Parameters: {count_parameters(d)/1e6:.2f}M")
