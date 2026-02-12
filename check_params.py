from net.generator import Generator
from net.discriminator import Discriminator
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


g = Generator()
dm = Discriminator(input_nc=3)
ds = Discriminator(input_nc=1)

print(f"Generator Parameters: {count_parameters(g)/1e6:.2f}M")
print(f"Main Discriminator (Dm) Parameters: {count_parameters(dm)/1e6:.2f}M")
print(
    f"Support Discriminator (Ds) Parameters: {count_parameters(ds)/1e6:.2f}M")
