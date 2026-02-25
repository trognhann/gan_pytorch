"""
Verify ContentLoss and StyleLoss correctness.
Run: python verify_losses.py

Checks:
1. Output shape and dtype
2. Loss = 0 for identical inputs
3. Loss > 0 for different inputs
4. No NaN/Inf under AMP (fp16)
5. Gradient flow correctness
6. Scale reasonableness with default weights
"""

import torch
import torch.nn as nn
from losses.vgg import VGG19
from losses.content import ContentLoss
from losses.style import StyleLoss
from losses.color_lab import ColorLoss

device = torch.device('cpu')
print("=" * 60)
print("AnimeGANv3 Loss Verification")
print("=" * 60)

# Shared VGG
vgg = VGG19().to(device)

content_loss_fn = ContentLoss(vgg=vgg).to(device)
style_loss_fn = StyleLoss(weights=[0.1, 5.0, 25.0], vgg=vgg).to(device)
color_loss_fn = ColorLoss(weight=10.0).to(device)

# Test data: random [-1, 1] images
torch.manual_seed(42)
img_a = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(device)
img_b = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(device)
img_a.requires_grad_(True)

passed = 0
total = 0


def check(name, condition, detail=""):
    global passed, total
    total += 1
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}: {name}" + (f" ({detail})" if detail else ""))
    if condition:
        passed += 1
    return condition


# ============================================================
# 1. ContentLoss
# ============================================================
print("\n--- ContentLoss ---")

# Identical inputs -> loss = 0
con_same = content_loss_fn(img_a.detach(), img_a.detach())
check("Identical inputs -> loss ~= 0", con_same.item()
      < 1e-6, f"got {con_same.item():.2e}")

# Different inputs -> loss > 0
con_diff = content_loss_fn(img_a, img_b.detach())
check("Different inputs -> loss > 0", con_diff.item()
      > 0, f"got {con_diff.item():.4f}")

# No NaN
check("No NaN/Inf", not (torch.isnan(con_diff) or torch.isinf(con_diff)))

# Gradient flows
con_diff.backward()
check("Gradient flows to input",
      img_a.grad is not None and img_a.grad.abs().sum() > 0)
img_a.grad = None

# VGG layer correctness: slices 1->relu2_2(128ch), 2->relu3_3(256ch), 3->relu4_4(512ch)
with torch.no_grad():
    r2, r3, r4 = vgg(img_a)
    check("VGG relu2_2 channels = 128",
          r2.shape[1] == 128, f"got {r2.shape[1]}")
    check("VGG relu3_3 channels = 256",
          r3.shape[1] == 256, f"got {r3.shape[1]}")
    check("VGG relu4_4 channels = 512",
          r4.shape[1] == 512, f"got {r4.shape[1]}")

# ContentLoss uses relu4_4 only
# Verify: L1(real_feat, fake_feat) / C where C=512
with torch.no_grad():
    _, _, feat_a = vgg(img_a)
    _, _, feat_b = vgg(img_b)
    expected = nn.L1Loss()(feat_a, feat_b) / 512.0
    actual = content_loss_fn(img_a, img_b)
    check("ContentLoss = L1(relu4_4) / 512",
          abs(expected.item() - actual.item()) < 1e-5,
          f"expected={expected.item():.6f}, actual={actual.item():.6f}")

# Scale check
print(f"\n  ContentLoss scale (random inputs): {con_diff.item():.6f}")
print(f"     After con_weight=0.5: {0.5 * con_diff.item():.6f}")

# ============================================================
# 2. StyleLoss
# ============================================================
print("\n--- StyleLoss ---")


# Need grayscale 3ch inputs (matches training usage)
def to_gray_3ch(x):
    gray = 0.2125 * x[:, 0:1] + 0.7154 * x[:, 1:2] + 0.0721 * x[:, 2:3]
    return gray.repeat(1, 3, 1, 1)


gray_a = to_gray_3ch(img_a.detach())
gray_b = to_gray_3ch(img_b.detach())
gray_a.requires_grad_(True)

# Identical inputs -> loss ~= 0
sty_same = style_loss_fn(gray_a.detach(), gray_a.detach())
check("Identical inputs -> loss ~= 0", sty_same.item()
      < 1e-5, f"got {sty_same.item():.2e}")

# Different inputs -> loss > 0
sty_diff = style_loss_fn(gray_a, gray_b.detach())
check("Different inputs -> loss > 0", sty_diff.item()
      > 0, f"got {sty_diff.item():.4f}")

# No NaN
check("No NaN/Inf", not (torch.isnan(sty_diff) or torch.isinf(sty_diff)))

# Gradient flows
sty_diff.backward()
check("Gradient flows to input",
      gray_a.grad is not None and gray_a.grad.abs().sum() > 0)

# Gram matrix verification
print(f"\n  StyleLoss scale (random grayscale): {sty_diff.item():.6f}")
print(f"     Weights: relu2_2={0.1}, relu3_3={5.0}, relu4_4={25.0}")

# Verify Gram matrix shape
with torch.no_grad():
    feat = torch.randn(2, 128, 64, 64)
    gram = style_loss_fn.gram_matrix(feat)
    check("Gram matrix shape [B, C, C]", gram.shape ==
          (2, 128, 128), f"got {gram.shape}")
    check("Gram matrix dtype = float32", gram.dtype == torch.float32)

# ============================================================
# 3. ColorLoss (AMP safety)
# ============================================================
print("\n--- ColorLoss (AMP safety) ---")

color_a = img_a.detach().requires_grad_(True)

col_diff = color_loss_fn(color_a, img_b.detach())
check("No NaN/Inf (fp32)", not (torch.isnan(col_diff) or torch.isinf(col_diff)),
      f"got {col_diff.item():.4f}")

col_diff.backward()
check("Gradient flows", color_a.grad is not None and color_a.grad.abs().sum() > 0)

# Test with fp16 input (simulates AMP autocast)
if torch.cuda.is_available():
    device_cuda = torch.device('cuda')
    color_loss_cuda = ColorLoss(weight=10.0).to(device_cuda)
    img_fp16 = img_a.detach().to(device_cuda).half()
    img_fp16_b = img_b.detach().to(device_cuda).half()

    col_amp = color_loss_cuda(img_fp16, img_fp16_b)
    check("No NaN/Inf with fp16 input (AMP)",
          not (torch.isnan(col_amp) or torch.isinf(col_amp)),
          f"got {col_amp.item():.4f}")
    check("Output dtype = float32 (upcasted)", col_amp.dtype == torch.float32)
else:
    print("  (Skipping fp16/AMP test -- no CUDA)")

print(f"\n  ColorLoss scale (random inputs, weight=10): {col_diff.item():.4f}")

# ============================================================
# 4. Scale Summary
# ============================================================
print("\n" + "=" * 60)
print("Scale Summary (random inputs, before loss weights)")
print("=" * 60)

with torch.no_grad():
    raw_con = content_loss_fn(img_a, img_b).item()
    raw_sty = style_loss_fn(to_gray_3ch(img_a), to_gray_3ch(img_b)).item()
    raw_col = color_loss_fn(img_a, img_b).item()

print(f"  ContentLoss (raw):         {raw_con:.6f}")
print(f"  ContentLoss x con(0.5):    {0.5 * raw_con:.6f}")
print(f"  StyleLoss (with weights):  {raw_sty:.6f}")
print(f"  ColorLoss (weight=10):     {raw_col:.4f}")
print()
print(f"  Con / Sty ratio:           {raw_con / (raw_sty + 1e-8):.2f}")
print(f"  Con x 0.5 / Sty ratio:     {0.5 * raw_con / (raw_sty + 1e-8):.2f}")

if raw_sty < 1e-8:
    print("  WARNING: StyleLoss near 0 -- likely dominated by other losses")
elif 0.5 * raw_con / raw_sty > 100:
    print("  WARNING: ContentLoss dominates StyleLoss by 100x+ -- style may not learn")
elif 0.5 * raw_con / raw_sty < 0.01:
    print("  WARNING: StyleLoss dominates ContentLoss by 100x+ -- may lose content")
else:
    print("  OK: Con/Sty balance looks reasonable")

# ============================================================
# Final
# ============================================================
print(f"\n{'=' * 60}")
print(f"Results: {passed}/{total} checks passed")
print(f"{'=' * 60}")
