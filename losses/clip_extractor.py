"""
CLIP ViT-B/16 Feature Extractor for Loss Computation
=====================================================
Drop-in replacement for VGG19: hooks into intermediate ViT layers
to produce 3 spatial feature maps with the same API signature.

Output tuple: (f_low, f_mid, f_high)
  - f_low:  (B, 128, H/16, W/16)  ← from ViT layer 4
  - f_mid:  (B, 256, H/16, W/16)  ← from ViT layer 8
  - f_high: (B, 512, H/16, W/16)  ← from ViT layer 12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor


class CLIPFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        layers: list = None,
        out_channels: list = None,
    ):
        """
        Args:
            model_name:    HuggingFace model id for CLIP vision encoder.
            layers:        Which hidden layers to tap (1-indexed).
                           Default: [4, 8, 12].
            out_channels:  Target channel dims after 1×1 projection.
                           Default: [128, 256, 512] to match VGG19 dims.
        """
        super().__init__()

        if layers is None:
            layers = [4, 8, 12]
        if out_channels is None:
            out_channels = [128, 256, 512]

        assert len(layers) == 3, "Exactly 3 layers required for API compat"
        assert len(out_channels) == 3, "Exactly 3 out_channels required"

        self.layers = layers

        # ── Load CLIP vision encoder ────────────────────────────────────
        clip_model = CLIPVisionModel.from_pretrained(model_name)
        self.vision_model = clip_model.vision_model
        self.hidden_size = self.vision_model.config.hidden_size  # 768
        self.patch_size = self.vision_model.config.patch_size      # 16
        self.image_size = self.vision_model.config.image_size      # 224

        # Freeze all CLIP weights
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # ── 1×1 Conv projections (trainable) ────────────────────────────
        # These project 768-dim ViT features down to VGG-compatible dims
        self.proj_low = nn.Conv2d(self.hidden_size, out_channels[0], 1, bias=False)
        self.proj_mid = nn.Conv2d(self.hidden_size, out_channels[1], 1, bias=False)
        self.proj_high = nn.Conv2d(self.hidden_size, out_channels[2], 1, bias=False)

        # Init projections with kaiming
        nn.init.kaiming_normal_(self.proj_low.weight)
        nn.init.kaiming_normal_(self.proj_mid.weight)
        nn.init.kaiming_normal_(self.proj_high.weight)

        # ── CLIP preprocessing constants ────────────────────────────────
        # CLIP expects images normalized with these stats (after [0,1] range)
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] training tensor → CLIP-ready input."""
        # [-1, 1] → [0, 1]
        x = (x + 1.0) / 2.0
        # Resize to CLIP expected size if needed
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        # Normalize with CLIP stats
        x = (x - self.mean) / self.std
        return x

    def _tokens_to_spatial(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Convert ViT hidden state [B, seq_len, D] → spatial [B, D, h, w].
        Removes the CLS token (index 0), reshapes remaining patch tokens.
        """
        # Remove CLS token
        patch_tokens = hidden_state[:, 1:, :]  # [B, num_patches, D]
        B, N, D = patch_tokens.shape
        h = w = int(N ** 0.5)
        return patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input images in [-1, 1] range, shape (B, 3, H, W).

        Returns:
            Tuple of 3 feature maps: (f_low, f_mid, f_high)
            matching VGG19 API signature.
        """
        x = self._preprocess(x)

        # Forward through CLIP vision encoder with hidden states output
        outputs = self.vision_model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (num_layers + 1) tensors, each [B, seq_len, D]
        # Index 0 = embedding output, index i = output of layer i
        hidden_states = outputs.hidden_states

        # Extract features from specified layers
        f_low_raw = self._tokens_to_spatial(hidden_states[self.layers[0]])
        f_mid_raw = self._tokens_to_spatial(hidden_states[self.layers[1]])
        f_high_raw = self._tokens_to_spatial(hidden_states[self.layers[2]])

        # Project to target channel dimensions
        f_low = self.proj_low(f_low_raw)
        f_mid = self.proj_mid(f_mid_raw)
        f_high = self.proj_high(f_high_raw)

        return f_low, f_mid, f_high
