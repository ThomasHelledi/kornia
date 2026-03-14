"""Warp — audio-driven geometry distortion via grid_sample.

Modes: wave, radial, vortex, fisheye. Bass drives warp strength,
time drives phase for continuous animation.
"""

import math
import torch
import torch.nn.functional as F

from .audio_reactive import AudioReactiveEngine, get_6bands


class WarpEffect:
    """Audio-reactive geometry distortion.

    Params:
        mode: str — 'wave' | 'radial' | 'vortex' | 'fisheye' (default: 'wave')
        strength: float — base warp amplitude (default: 0.03)
        frequency: float — wave frequency for wave mode (default: 4.0)
        speed: float — animation speed multiplier (default: 1.0)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()

    def _make_base_grid(self, h, w):
        """Create normalized coordinate grid [-1, 1]."""
        y = torch.linspace(-1, 1, h, device=self.device)
        x = torch.linspace(-1, 1, w, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    def process(self, tensor, params):
        mode = params.get('mode', 'wave')
        base_strength = float(params.get('strength', 0.03))
        freq = float(params.get('frequency', 4.0))
        speed = float(params.get('speed', 1.0))

        # Audio-reactive: bass drives warp strength
        bands = get_6bands(params)
        beat_pulse = float(params.get('beat_pulse', 0))
        time = float(params.get('time', 0))
        bass = self._audio.smooth('bass', bands['bass'] + bands['sub_bass'], attack=0.3, release=0.06)

        strength = base_strength * (1.0 + bass * 3.0 + beat_pulse * 1.5)
        phase = time * speed

        _, _, h, w = tensor.shape
        grid = self._make_base_grid(h, w)
        gx = grid[..., 0]  # (1, H, W)
        gy = grid[..., 1]

        if mode == 'wave':
            # Horizontal sine wave displacement
            offset_x = strength * torch.sin(gy * freq * math.pi + phase * 2.0)
            offset_y = strength * 0.5 * torch.cos(gx * freq * math.pi + phase * 1.5)
            gx = gx + offset_x
            gy = gy + offset_y

        elif mode == 'radial':
            # Radial pulsation from center
            dist = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-6)
            radial_offset = strength * torch.sin(dist * freq * math.pi - phase * 2.0)
            gx = gx + gx / dist * radial_offset
            gy = gy + gy / dist * radial_offset

        elif mode == 'vortex':
            # Rotational distortion, stronger at center
            dist = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-6)
            angle = strength * 3.0 * torch.exp(-dist * 2.0) * math.sin(phase)
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            new_gx = gx * cos_a - gy * sin_a
            new_gy = gx * sin_a + gy * cos_a
            gx = new_gx
            gy = new_gy

        elif mode == 'fisheye':
            # Barrel distortion (fisheye lens)
            dist = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-6)
            # Audio modulates fisheye strength
            k = strength * 5.0
            r = dist
            distorted_r = r * (1.0 + k * r ** 2)
            scale = distorted_r / dist
            gx = gx * scale
            gy = gy * scale

        warp_grid = torch.stack([gx, gy], dim=-1)  # (1, H, W, 2)
        result = F.grid_sample(tensor, warp_grid, mode='bilinear', padding_mode='reflection', align_corners=True)

        return result.clamp(0, 1)
