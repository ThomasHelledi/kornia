"""Edge Glow — Kornia Canny edge detection + HSV glow + morphological cleanup.

V2: HSV-space hue shift by spectral centroid, morphological edge cleanup,
bilateral bloom for edge-preserving glow.
"""

import torch
import kornia.filters as KF
import kornia.color as KC
import kornia.morphology as KM

from .audio_reactive import AudioReactiveEngine, get_spectral


class EdgeGlowEffect:
    """Detect edges with Canny, HSV-colorize, bloom with bilateral blur.

    Params:
        threshold_low: float — Canny low threshold (default: 0.1)
        threshold_high: float — Canny high threshold (default: 0.3)
        glow_radius: int — blur radius (default: 5)
        glow_color: [R, G, B] — glow tint 0-255 (default: [0, 200, 255])
        blend: float — blend strength (default: 0.7)
        morph_cleanup: bool — apply morphological cleanup to edges (default: true)
        bilateral_bloom: bool — use bilateral blur for edge-preserving glow (default: false)
        hsv_glow: bool — shift hue by spectral centroid (default: true)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()

    def process(self, tensor, params):
        low = float(params.get('threshold_low', 0.1))
        high = float(params.get('threshold_high', 0.3))
        radius = int(params.get('glow_radius', 5))
        color = params.get('glow_color', [0, 200, 255])
        blend = float(params.get('blend', 0.7))
        morph_cleanup = params.get('morph_cleanup', True)
        use_bilateral = params.get('bilateral_bloom', False)
        hsv_glow = params.get('hsv_glow', True)

        # Audio-reactive
        beat_pulse = float(params.get('beat_pulse', 0))
        bass_energy = float(params.get('bass_energy', 0))
        high_energy = float(params.get('high_energy', 0))
        spectral = get_spectral(params)
        centroid = spectral['centroid']

        beat = self._audio.smooth('beat', beat_pulse, attack=0.5, release=0.06)
        bass = self._audio.smooth('bass', bass_energy, attack=0.3, release=0.05)

        if beat > 0 or bass > 0:
            blend = blend * (1.0 + beat * 0.6)
            radius = max(3, int(radius + bass * 8))
            cr, cg, cb = color
            cb = min(255, int(cb + high_energy * 55))
            cg = min(255, int(cg + high_energy * 30))
            color = [cr, cg, cb]

        # Edge detection
        gray = KC.rgb_to_grayscale(tensor)
        _, edges = KF.canny(gray, low_threshold=low, high_threshold=high)

        # Morphological cleanup: close small gaps, remove noise
        if morph_cleanup:
            kernel = torch.ones(3, 3, device=self.device)
            edges = KM.closing(edges, kernel)   # Close gaps
            edges = KM.opening(edges, kernel)   # Remove small noise

        # Bloom: gaussian or bilateral blur
        k = radius * 2 + 1
        sigma = float(radius)
        if use_bilateral:
            # Bilateral blur preserves edge structure in the glow
            glow = KF.bilateral_blur(edges.expand_as(tensor), (k, k), sigma, (sigma, sigma))
            glow = KC.rgb_to_grayscale(glow)  # Back to single channel
        else:
            glow = KF.gaussian_blur2d(edges, (k, k), (sigma, sigma))

        # HSV glow: shift hue based on spectral centroid
        if hsv_glow and centroid > 0:
            # Centroid 0-1 maps to hue shift: low centroid = warm (red/orange), high = cool (cyan/blue)
            r, g, b = [c / 255.0 for c in color]
            hue_shift = centroid * 0.5  # 0-0.5 hue rotation
            # Create base colored glow
            colored = torch.cat([glow * r, glow * g, glow * b], dim=1)
            # Convert to HSV, shift hue, convert back
            hsv = KC.rgb_to_hsv(colored.clamp(1e-6, 1))
            hsv[:, 0:1] = (hsv[:, 0:1] + hue_shift * 2 * 3.14159) % (2 * 3.14159)
            hsv[:, 1:2] = (hsv[:, 1:2] * (1.0 + beat * 0.3)).clamp(0, 1)  # Saturate on beats
            colored = KC.hsv_to_rgb(hsv)
        else:
            r, g, b = [c / 255.0 for c in color]
            colored = torch.cat([glow * r, glow * g, glow * b], dim=1)

        # Additive blend
        return (tensor + colored * blend).clamp(0, 1)
