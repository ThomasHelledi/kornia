"""Neural Style Transfer — VGG19 perceptual loss + gram matrix optimization.

Transfers artistic style from a reference image onto the content frame.
Uses Gatys et al. approach: minimize content loss + style loss simultaneously.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# VGG19 layers for style/content extraction
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def gram_matrix(x):
    """Compute Gram matrix for style representation."""
    b, c, h, w = x.shape
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)


class VGGFeatureExtractor(nn.Module):
    """Extract features from specific VGG19 layers."""

    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        # Map layer names to indices
        self.layer_map = {}
        layer_names = {
            0: 'conv1_1', 2: 'conv1_2',
            5: 'conv2_1', 7: 'conv2_2',
            10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3', 16: 'conv3_4',
            19: 'conv4_1', 21: 'conv4_2', 23: 'conv4_3', 25: 'conv4_4',
            28: 'conv5_1', 30: 'conv5_2', 32: 'conv5_3', 34: 'conv5_4',
        }

        # Build sequential slices for each needed layer
        self.slices = nn.ModuleList()
        needed = set(CONTENT_LAYERS + STYLE_LAYERS)
        prev = 0
        for idx in sorted(layer_names.keys()):
            name = layer_names[idx]
            if name in needed:
                self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
                self.layer_map[name] = len(self.slices) - 1
                prev = idx + 1

        self._mean = MEAN.to(device)
        self._std = STD.to(device)

    def forward(self, x):
        """Extract features from all registered layers."""
        x = (x - self._mean) / self._std
        features = {}
        for i, sl in enumerate(self.slices):
            x = sl(x)
            # Find which layer name maps to this index
            for name, idx in self.layer_map.items():
                if idx == i:
                    features[name] = x
        return features


class StyleTransferEffect:
    """Neural style transfer using VGG19 perceptual loss.

    Params:
        style_image: str — path to style reference image
        content_weight: float — content loss weight (default: 1.0)
        style_weight: float — style loss weight (default: 1e6)
        iterations: int — optimization steps (default: 50)
        lr: float — learning rate (default: 0.05)
        process_resolution: int — max dimension for optimization (default: 0 = full res)
            Set to 256-384 for ~4x speedup. Result is upscaled + blended with original.
        transition_frames: int — smooth crossfade frames on style change (default: 5)
    """

    def __init__(self, device):
        self.device = device
        self.extractor = None
        self._style_cache = {}  # path → gram matrices
        self._prev_output = None  # Previous frame output for temporal coherence
        self._prev_style = None   # Previous style path
        self._transition_remaining = 0  # Frames left in crossfade transition
        self._process_res = 0     # Cached process resolution for prev_output matching

    def _ensure_extractor(self):
        if self.extractor is not None:
            return
        sys.stderr.write("kornia-server: loading VGG19 for style transfer...\n")
        sys.stderr.flush()
        self.extractor = VGGFeatureExtractor(self.device)
        sys.stderr.write("kornia-server: VGG19 ready\n")
        sys.stderr.flush()

    def _load_style_grams(self, style_path):
        """Load style image and compute gram matrices (cached)."""
        if style_path in self._style_cache:
            return self._style_cache[style_path]

        from PIL import Image
        import torchvision.transforms as T

        if not os.path.exists(style_path):
            sys.stderr.write(f"kornia-server: style image not found: {style_path}\n")
            sys.stderr.flush()
            return None

        img = Image.open(style_path).convert('RGB')
        transform = T.Compose([
            T.Resize(512),
            T.ToTensor(),
        ])
        style_tensor = transform(img).unsqueeze(0).to(self.device)

        # Extract style features + gram matrices
        with torch.no_grad():
            style_features = self.extractor(style_tensor)
        grams = {name: gram_matrix(feat) for name, feat in style_features.items()
                 if name in STYLE_LAYERS}

        self._style_cache[style_path] = grams
        sys.stderr.write(f"kornia-server: style image loaded ({style_path})\n")
        sys.stderr.flush()
        return grams

    def _optimize(self, tensor, style_grams, content_weight, style_weight,
                  iterations, lr, temporal_weight, prev_output):
        """Run iterative Gatys optimization at the given tensor resolution.

        Args:
            tensor: (1, 3, H, W) input at optimization resolution
            style_grams: precomputed gram matrices from style image
            prev_output: previous frame at same resolution (or None)

        Returns:
            (1, 3, H, W) optimized result
        """
        # Extract content features
        with torch.no_grad():
            content_features = self.extractor(tensor)
        content_targets = {name: feat.detach() for name, feat in content_features.items()
                          if name in CONTENT_LAYERS}

        # Seed from previous frame for temporal coherence
        if prev_output is not None and prev_output.shape == tensor.shape:
            init = 0.7 * prev_output + 0.3 * tensor
        else:
            init = tensor.detach().clone()
        output = init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([output], lr=lr)

        for i in range(iterations):
            optimizer.zero_grad()
            features = self.extractor(output.clamp(0, 1))

            # Content loss
            c_loss = 0
            for name in CONTENT_LAYERS:
                if name in features and name in content_targets:
                    c_loss += F.mse_loss(features[name], content_targets[name])

            # Style loss
            s_loss = 0
            for name in STYLE_LAYERS:
                if name in features and name in style_grams:
                    G = gram_matrix(features[name])
                    s_loss += F.mse_loss(G, style_grams[name])

            # Temporal coherence loss
            t_loss = 0
            if prev_output is not None and prev_output.shape == output.shape:
                t_loss = F.mse_loss(output, prev_output.detach())

            loss = content_weight * c_loss + style_weight * s_loss + temporal_weight * t_loss
            loss.backward()
            optimizer.step()

        return output.detach().clamp(0, 1)

    def process(self, tensor, params):
        """Apply neural style transfer with temporal coherence + resolution scaling.

        Args:
            tensor: (1, 3, H, W) float tensor in [0, 1]
            params: dict with style_image, content_weight, style_weight, iterations, lr,
                    temporal_weight (default 1e4), process_resolution (default 0),
                    transition_frames (default 5)
        Returns:
            (1, 3, H, W) float tensor in [0, 1]
        """
        self._ensure_extractor()

        style_path = params.get('style_image', '')
        content_weight = float(params.get('content_weight', 1.0))
        style_weight = float(params.get('style_weight', 1e6))
        iterations = int(params.get('iterations', 50))
        lr = float(params.get('lr', 0.05))
        temporal_weight = float(params.get('temporal_weight', 1e4))
        process_res = int(params.get('process_resolution', 0))
        transition_frames = int(params.get('transition_frames', 5))

        # If no style image, apply a default artistic tint
        if not style_path:
            return (tensor * 1.05 + 0.02).clamp(0, 1)

        # Load style gram matrices
        style_grams = self._load_style_grams(style_path)
        if style_grams is None:
            return tensor

        # Handle style transitions: don't hard-cut, crossfade via temporal weight
        style_changed = (style_path != self._prev_style)
        if style_changed:
            # Keep prev_output for smooth crossfade (don't reset to None)
            self._prev_style = style_path
            self._transition_remaining = transition_frames

        # During transition: boost temporal weight for smoother crossfade
        effective_temporal = temporal_weight
        if self._transition_remaining > 0:
            # Fade: 3x temporal at start → 1x at end of transition
            t_factor = self._transition_remaining / max(transition_frames, 1)
            effective_temporal = temporal_weight * (1.0 + t_factor * 2.0)
            self._transition_remaining -= 1

        orig_h, orig_w = tensor.shape[2], tensor.shape[3]

        # Resolution scaling: optimize at lower res for speed
        if process_res > 0 and max(orig_h, orig_w) > process_res:
            scale = process_res / max(orig_h, orig_w)
            small_h = max(64, int(orig_h * scale) - int(orig_h * scale) % 2)
            small_w = max(64, int(orig_w * scale) - int(orig_w * scale) % 2)

            # Downscale input
            tensor_small = F.interpolate(tensor, size=(small_h, small_w),
                                         mode='bilinear', align_corners=False)

            # Match prev_output resolution
            prev_small = None
            if self._prev_output is not None:
                if self._prev_output.shape[2:] == (small_h, small_w):
                    prev_small = self._prev_output
                else:
                    # Rescale cached output to match new working resolution
                    prev_small = F.interpolate(self._prev_output, size=(small_h, small_w),
                                               mode='bilinear', align_corners=False)

            # Optimize at small res
            result_small = self._optimize(tensor_small, style_grams, content_weight,
                                          style_weight, iterations, lr,
                                          effective_temporal, prev_small)

            # Store at small res for next frame
            self._prev_output = result_small.clone()
            self._process_res = process_res

            # Upscale + blend with original for detail preservation
            result = F.interpolate(result_small, size=(orig_h, orig_w),
                                   mode='bilinear', align_corners=False)
            # 80% stylized + 20% original detail
            result = result * 0.8 + tensor * 0.2

        else:
            # Full resolution optimization
            prev = self._prev_output if (self._prev_output is not None and
                                          self._prev_output.shape == tensor.shape) else None
            result = self._optimize(tensor, style_grams, content_weight, style_weight,
                                    iterations, lr, effective_temporal, prev)
            self._prev_output = result.clone()
            self._process_res = 0

        return result.clamp(0, 1)
