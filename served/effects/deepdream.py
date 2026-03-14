"""DeepDream — InceptionV3 gradient ascent feature visualization.

Multi-octave processing amplifies features at different scales,
producing the characteristic hallucinated eyes, faces, and fractal patterns.

V2: Multiple loss functions, layer presets, audio-driven layer switching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .audio_reactive import AudioReactiveEngine, get_spectral


# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Layer presets: named configs mapping to InceptionV3 layers
LAYER_PRESETS = {
    'abstract': 'mixed3a',   # Early features: edges, textures
    'faces':    'mixed4c',   # Mid features: face-like patterns
    'fractal':  'mixed5a',   # Deep features: complex fractal forms
    'eyes':     'mixed5b',   # Very deep: eye-like hallucinations
    'soft':     'mixed3b',   # Early-mid: soft flowing patterns
}


class FeatureExtractor(nn.Module):
    """Extract features from a specific InceptionV3 layer."""

    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self._features = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        layer = self._resolve_layer(self.target_layer)
        self._hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._features = output

    def _resolve_layer(self, name):
        """Map friendly names to InceptionV3 submodules."""
        layer_map = {
            'mixed3a': self.model.Mixed_5b,
            'mixed3b': self.model.Mixed_5c,
            'mixed4a': self.model.Mixed_5d,
            'mixed4b': self.model.Mixed_6a,
            'mixed4c': self.model.Mixed_6b,
            'mixed4d': self.model.Mixed_6c,
            'mixed4e': self.model.Mixed_6d,
            'mixed5a': self.model.Mixed_6e,
            'mixed5b': self.model.Mixed_7a,
            'mixed5c': self.model.Mixed_7b,
        }
        if name in layer_map:
            return layer_map[name]
        # Try direct attribute access
        parts = name.split('.')
        mod = self.model
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def forward(self, x):
        self._features = None
        try:
            self.model(x)
        except Exception:
            pass  # We only need the hook output
        return self._features

    def cleanup(self):
        if self._hook:
            self._hook.remove()


class DeepDreamEffect:
    """Multi-octave DeepDream via InceptionV3 gradient ascent.

    Params:
        layer: str — target layer name or preset (default: 'mixed5a')
        iterations: int — gradient steps per octave (default: 20)
        lr: float — learning rate (default: 0.01)
        octaves: int — number of resolution octaves (default: 4)
        octave_scale: float — scale factor between octaves (default: 1.4)
        jitter: int — random shift for tiling artifacts reduction (default: 32)
        loss_fn: str — 'norm' | 'mse' | 'cosine' (default: 'norm')
        preset: str — layer preset name (overrides layer param)

    Audio-reactive:
        bass_energy, beat_pulse, audio_intensity → modulate iterations + lr
        Downbeats → cycle through layer presets
    """

    def __init__(self, device):
        self.device = device
        self.model = None
        self._extractors = {}
        self._mean = MEAN.to(device)
        self._std = STD.to(device)
        self._audio = AudioReactiveEngine()
        self._preset_idx = 0
        self._preset_names = list(LAYER_PRESETS.keys())

    def _ensure_model(self):
        if self.model is not None:
            return
        import sys
        sys.stderr.write("kornia-server: loading InceptionV3...\n")
        sys.stderr.flush()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        sys.stderr.write("kornia-server: InceptionV3 ready\n")
        sys.stderr.flush()

    def _get_extractor(self, layer):
        if layer not in self._extractors:
            self._ensure_model()
            self._extractors[layer] = FeatureExtractor(self.model, layer)
        return self._extractors[layer]

    def _normalize(self, img):
        return (img - self._mean) / self._std

    def _compute_loss(self, features, loss_fn):
        """Compute loss based on selected function."""
        if loss_fn == 'mse':
            # MSE from zero — maximize activation magnitude
            return (features ** 2).mean()
        elif loss_fn == 'cosine':
            # Cosine similarity with mean activation direction
            flat = features.flatten(1)
            mean_dir = flat.mean(dim=1, keepdim=True)
            return F.cosine_similarity(flat, mean_dir, dim=1).mean()
        else:
            # Default: L2 norm (original DeepDream)
            return features.norm()

    def _dream_octave(self, img, extractor, iterations, lr, jitter, loss_fn):
        """Single octave: gradient ascent to maximize feature activation."""
        img = img.detach().clone().requires_grad_(True)

        for i in range(iterations):
            # Random jitter to reduce tiling artifacts
            ox, oy = 0, 0
            if jitter > 0:
                ox = torch.randint(-jitter, jitter + 1, (1,)).item()
                oy = torch.randint(-jitter, jitter + 1, (1,)).item()
                img_shifted = torch.roll(img, shifts=(ox, oy), dims=(2, 3))
            else:
                img_shifted = img

            # Forward through model to target layer
            normed = self._normalize(img_shifted)
            features = extractor(normed)

            if features is None:
                break

            # Compute loss based on selected function
            loss = self._compute_loss(features, loss_fn)
            loss.backward()

            # Gradient ascent
            with torch.no_grad():
                grad = img.grad
                # Normalize gradient
                grad = grad / (grad.std() + 1e-8)
                img += lr * grad
                img.clamp_(0, 1)
                img.grad = None
                img.requires_grad_(True)

        return img.detach()

    def process(self, tensor, params):
        """Apply multi-octave DeepDream.

        Args:
            tensor: (1, 3, H, W) float tensor in [0, 1]
            params: dict with optional keys
        Returns:
            (1, 3, H, W) float tensor in [0, 1]
        """
        # Resolve layer: preset overrides direct layer param
        preset = params.get('preset', '')
        if preset and preset in LAYER_PRESETS:
            layer = LAYER_PRESETS[preset]
        else:
            layer = params.get('layer', 'mixed5a')
            # Resolve if layer name matches a preset
            if layer in LAYER_PRESETS:
                layer = LAYER_PRESETS[layer]

        base_iterations = int(params.get('iterations', 20))
        base_lr = float(params.get('lr', 0.01))
        octaves = int(params.get('octaves', 4))
        octave_scale = float(params.get('octave_scale', 1.4))
        jitter = int(params.get('jitter', 32))
        loss_fn = params.get('loss_fn', 'norm')

        # Audio-reactive modulation
        audio_intensity = float(params.get('audio_intensity', 0))
        beat_pulse = float(params.get('beat_pulse', 0))
        spectral = get_spectral(params)
        flux = spectral['flux']
        bpm = float(params.get('bpm', 0))
        time = float(params.get('time', 0))

        if audio_intensity > 0 or beat_pulse > 0:
            # Iterations: 15-30 based on audio energy
            iterations = max(10, int(base_iterations * (0.75 + audio_intensity * 0.5)))
            # Learning rate: boost on beats for pulsing hallucinations
            lr = base_lr * (1.0 + beat_pulse * 0.5 + audio_intensity * 0.3)
        else:
            iterations = base_iterations
            lr = base_lr

        # Audio layer switching: cycle presets on downbeats
        is_downbeat, _ = self._audio.detect_downbeat(flux, bpm, time, threshold=0.2)
        if is_downbeat:
            self._preset_idx = (self._preset_idx + 1) % len(self._preset_names)
            new_preset = self._preset_names[self._preset_idx]
            layer = LAYER_PRESETS[new_preset]

        extractor = self._get_extractor(layer)

        _, _, orig_h, orig_w = tensor.shape

        # Generate octave sizes (smallest first)
        octave_sizes = []
        h, w = orig_h, orig_w
        for _ in range(octaves - 1):
            h = int(h / octave_scale)
            w = int(w / octave_scale)
            octave_sizes.append((h, w))
        octave_sizes.reverse()
        octave_sizes.append((orig_h, orig_w))

        # Start from smallest octave
        detail = torch.zeros_like(tensor)

        for i, (oh, ow) in enumerate(octave_sizes):
            # Resize base image
            if oh != orig_h or ow != orig_w:
                base = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=False)
                detail_resized = F.interpolate(detail, size=(oh, ow), mode='bilinear', align_corners=False)
            else:
                base = tensor
                detail_resized = detail

            # Add accumulated detail
            img = (base + detail_resized).clamp(0, 1)

            # Dream this octave
            dreamed = self._dream_octave(img, extractor, iterations, lr, jitter, loss_fn)

            # Extract detail (what the dream added)
            detail = dreamed - base
            if oh != orig_h or ow != orig_w:
                detail = F.interpolate(detail, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        # Final result
        result = (tensor + detail).clamp(0, 1)
        return result
