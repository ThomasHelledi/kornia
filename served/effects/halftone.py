"""Halftone — comic-book / posterize effects.

Modes: posterize (color quantization), halftone (dot grid pattern),
comic (posterize + black edge outlines). Beats reduce color levels
for more abstract impact.
"""

import torch
import kornia.filters as KF
import kornia.color as KC

from .audio_reactive import AudioReactiveEngine


class HalftoneEffect:
    """Comic-book style posterization and halftone dot patterns.

    Params:
        mode: str — 'posterize' | 'halftone' | 'comic' (default: 'comic')
        levels: int — color quantization levels (default: 6)
        dot_size: int — halftone dot grid spacing in pixels (default: 4)
        outline_strength: float — black outline strength for comic mode (default: 0.8)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()

    def _posterize(self, tensor, levels):
        """Quantize colors to N levels."""
        return torch.round(tensor * (levels - 1)) / (levels - 1)

    def _halftone_dots(self, tensor, dot_size):
        """Convert to halftone dot pattern."""
        _, _, h, w = tensor.shape
        gray = KC.rgb_to_grayscale(tensor)

        # Create dot grid
        y_coords = torch.arange(h, device=self.device).float()
        x_coords = torch.arange(w, device=self.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Distance from nearest grid point
        grid_y = (yy % dot_size) - dot_size / 2.0
        grid_x = (xx % dot_size) - dot_size / 2.0
        dist = torch.sqrt(grid_y ** 2 + grid_x ** 2)

        # Dot radius proportional to brightness (brighter = larger dot)
        max_radius = dot_size * 0.6
        radius = gray.squeeze(0).squeeze(0) * max_radius

        # White dots on black where luminance is high
        dots = (dist.unsqueeze(0).unsqueeze(0) < radius.unsqueeze(0).unsqueeze(0)).float()

        # Colorize dots with original color
        return tensor * dots

    def _detect_edges(self, tensor, strength):
        """Detect edges for comic-book outlines."""
        gray = KC.rgb_to_grayscale(tensor)
        # Sobel edge detection
        edges_x = KF.sobel(gray)
        edge_mag = edges_x.abs()
        # Normalize and threshold
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        outline = (edge_mag > 0.15).float() * strength
        return outline

    def process(self, tensor, params):
        mode = params.get('mode', 'comic')
        base_levels = int(params.get('levels', 6))
        dot_size = int(params.get('dot_size', 4))
        outline_strength = float(params.get('outline_strength', 0.8))

        # Audio-reactive: beats reduce levels (more abstract on impact)
        beat_pulse = float(params.get('beat_pulse', 0))
        beat = self._audio.smooth('beat', beat_pulse, attack=0.5, release=0.08)

        # Fewer levels = more abstract = stronger effect on beats
        levels = max(2, int(base_levels - beat * 3))

        if mode == 'posterize':
            result = self._posterize(tensor, levels)

        elif mode == 'halftone':
            posterized = self._posterize(tensor, levels)
            result = self._halftone_dots(posterized, dot_size)

        elif mode == 'comic':
            # Posterize + black outlines
            posterized = self._posterize(tensor, levels)
            outlines = self._detect_edges(tensor, outline_strength)
            # Subtract outlines (black lines)
            result = posterized * (1.0 - outlines)

        else:
            result = tensor

        return result.clamp(0, 1)
