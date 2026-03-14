"""Glitch — beat-synced digital corruption effects.

RGB channel shift, scanline displacement, block corruption. Driven by
spectral flux and beat detection for music-synced visual corruption.
"""

import torch
import torch.nn.functional as F

from .audio_reactive import AudioReactiveEngine, get_spectral


class GlitchEffect:
    """Beat-synced digital glitch corruption.

    Params:
        beat_sync: bool — sync glitch intensity to beats (default: true)
        channel_shift: float — max RGB channel offset in pixels (default: 8.0)
        scanline_strength: float — scanline displacement intensity (default: 0.5)
        block_size: int — corruption block size in pixels (default: 16)
        block_chance: float — probability of block corruption per frame (default: 0.3)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()
        self._frame_idx = 0

    def process(self, tensor, params):
        beat_sync = params.get('beat_sync', True)
        max_shift = float(params.get('channel_shift', 8.0))
        scanline_str = float(params.get('scanline_strength', 0.5))
        block_size = int(params.get('block_size', 16))
        block_chance = float(params.get('block_chance', 0.3))

        # Audio reactivity
        spectral = get_spectral(params)
        beat_pulse = float(params.get('beat_pulse', 0))
        flux = spectral['flux']

        if beat_sync:
            is_beat, beat_strength = self._audio.detect_beat(flux, threshold=0.12)
            intensity = self._audio.smooth('intensity', beat_strength, attack=0.6, release=0.04)
        else:
            intensity = 0.5

        self._frame_idx += 1
        _, _, h, w = tensor.shape
        result = tensor.clone()

        # 1. RGB channel shift on beats
        shift_px = int(max_shift * intensity)
        if shift_px > 0:
            # Shift red channel right, blue channel left
            r, g, b = result[:, 0:1], result[:, 1:2], result[:, 2:3]
            r = torch.roll(r, shifts=shift_px, dims=3)
            b = torch.roll(b, shifts=-shift_px, dims=3)
            result = torch.cat([r, g, b], dim=1)

        # 2. Scanline displacement driven by spectral flux
        if scanline_str > 0 and flux > 0.05:
            scan_intensity = scanline_str * self._audio.smooth('flux', flux, attack=0.4, release=0.06)
            # Displace random scanlines horizontally
            num_lines = max(1, int(h * scan_intensity * 0.1))
            for _ in range(num_lines):
                y = torch.randint(0, h, (1,)).item()
                shift = torch.randint(-int(w * scan_intensity * 0.05) - 1,
                                      int(w * scan_intensity * 0.05) + 2, (1,)).item()
                if shift != 0 and 0 <= y < h:
                    result[:, :, y, :] = torch.roll(result[:, :, y, :], shifts=shift, dims=2)

        # 3. Block corruption on onset
        onset = self._audio.onset_strength(flux)
        if onset > 0.1 or (not beat_sync and self._frame_idx % 5 == 0):
            corrupt_prob = block_chance * max(onset, 0.2 if not beat_sync else 0.0)
            n_blocks_h = max(1, h // block_size)
            n_blocks_w = max(1, w // block_size)
            for by in range(n_blocks_h):
                for bx in range(n_blocks_w):
                    if torch.rand(1).item() < corrupt_prob:
                        y0 = by * block_size
                        x0 = bx * block_size
                        y1 = min(y0 + block_size, h)
                        x1 = min(x0 + block_size, w)
                        # Random corruption: invert, shift, or smear
                        corruption_type = torch.randint(0, 3, (1,)).item()
                        if corruption_type == 0:
                            # Color invert
                            result[:, :, y0:y1, x0:x1] = 1.0 - result[:, :, y0:y1, x0:x1]
                        elif corruption_type == 1:
                            # Horizontal smear (repeat first column)
                            if x0 < w:
                                result[:, :, y0:y1, x0:x1] = result[:, :, y0:y1, x0:x0+1].expand_as(
                                    result[:, :, y0:y1, x0:x1])
                        else:
                            # Channel swap
                            block = result[:, :, y0:y1, x0:x1].clone()
                            result[:, 0:1, y0:y1, x0:x1] = block[:, 2:3]
                            result[:, 2:3, y0:y1, x0:x1] = block[:, 0:1]

        return result.clamp(0, 1)
