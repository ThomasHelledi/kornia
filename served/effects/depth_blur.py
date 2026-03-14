"""Depth Blur — luminance-based synthetic depth + kornia bilateral blur.

V2: Sigmoid falloff for smooth depth→blur curve, chromatic aberration
per-channel shift based on depth, audio-reactive focus plane position.
"""

import torch
import kornia.filters as KF
import kornia.color as KC

from .audio_reactive import AudioReactiveEngine, get_6bands


class DepthBlurEffect:
    """Depth-aware blur with chromatic aberration and audio-reactive focus.

    Params:
        focus_depth: float — focus point depth 0-1 (default: 0.5)
        blur_strength: float — max blur sigma (default: 5.0)
        depth_mode: str — 'luminance' | 'radial' | 'vertical' (default: 'luminance')
        bilateral: bool — use bilateral blur to preserve edges (default: true)
        kernel_size: int — blur kernel size (default: 9)
        sigmoid_falloff: float — sigmoid steepness for depth→blur curve (default: 8.0)
        chromatic: float — chromatic aberration strength 0-1 (default: 0.0)
        audio_focus: bool — bass drives focus plane position (default: false)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()

    def _depth_from_luminance(self, tensor):
        """Estimate depth from luminance — brighter pixels are 'closer'."""
        gray = KC.rgb_to_grayscale(tensor)
        depth = gray / (gray.max() + 1e-6)
        return depth

    def _depth_radial(self, h, w):
        """Radial depth — center is in focus, edges are blurred."""
        cy, cx = h / 2.0, w / 2.0
        y = torch.arange(h, device=self.device).float()
        x = torch.arange(w, device=self.device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = torch.sqrt(torch.tensor(cx ** 2 + cy ** 2, device=self.device))
        depth = (dist / max_dist).unsqueeze(0).unsqueeze(0)
        return depth.clamp(0, 1)

    def _depth_vertical(self, h, w):
        """Vertical depth — top is far, bottom is close (tilt-shift)."""
        y = torch.linspace(1, 0, h, device=self.device)
        depth = y.view(1, 1, h, 1).expand(1, 1, h, w)
        return depth

    def _sigmoid_blur_map(self, depth, focus, steepness):
        """Smooth sigmoid falloff from focus plane to blur.

        Instead of linear abs(depth - focus), uses sigmoid for smooth transition.
        """
        # Signed distance from focus
        dist = depth - focus
        # Sigmoid: 0 at focus, 1 far from focus
        blur_map = torch.sigmoid(torch.abs(dist) * steepness - steepness * 0.3)
        return blur_map.clamp(0, 1)

    def _chromatic_aberration(self, tensor, depth, strength):
        """Apply per-channel shift based on depth — simulates lens dispersion."""
        if strength <= 0:
            return tensor

        _, _, h, w = tensor.shape
        # Shift amount based on depth (deeper = more shift)
        max_shift = int(strength * 5) + 1

        r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]

        # Red shifts outward (positive), blue shifts inward (negative)
        # Modulated by depth: deeper areas get more aberration
        shift_map = (depth * max_shift).int()

        # Simple approach: roll channels by different amounts based on average depth
        avg_depth = depth.mean().item()
        r_shift = max(1, int(avg_depth * max_shift))
        b_shift = -r_shift

        r = torch.roll(r, shifts=r_shift, dims=3)
        b = torch.roll(b, shifts=b_shift, dims=3)

        return torch.cat([r, g, b], dim=1)

    def process(self, tensor, params):
        focus = float(params.get('focus_depth', 0.5))
        strength = float(params.get('blur_strength', 5.0))
        mode = params.get('depth_mode', 'luminance')
        use_bilateral = params.get('bilateral', True)
        ks = int(params.get('kernel_size', 9))
        steepness = float(params.get('sigmoid_falloff', 8.0))
        chromatic = float(params.get('chromatic', 0.0))
        audio_focus = params.get('audio_focus', False)

        # Audio-reactive focus plane
        if audio_focus:
            bands = get_6bands(params)
            bass = self._audio.smooth('bass', bands['bass'] + bands['sub_bass'],
                                       attack=0.2, release=0.04)
            # Bass drives focus plane deeper (0.5 → 0.2 on heavy bass)
            focus = focus - bass * 0.3
            focus = max(0.0, min(1.0, focus))
            # Also modulate blur strength with audio
            beat = float(params.get('beat_pulse', 0))
            beat_s = self._audio.smooth('beat', beat, attack=0.4, release=0.06)
            strength *= (1.0 + beat_s * 0.5)

        _, _, h, w = tensor.shape

        # Generate depth map
        if mode == 'radial':
            depth = self._depth_radial(h, w)
        elif mode == 'vertical':
            depth = self._depth_vertical(h, w)
        else:
            depth = self._depth_from_luminance(tensor)

        # Compute blur amount: sigmoid falloff for smooth transition
        if steepness > 0:
            blur_map = self._sigmoid_blur_map(depth, focus, steepness)
        else:
            # Linear falloff (legacy behavior)
            blur_map = (torch.abs(depth - focus) * 2.0).clamp(0, 1)

        # Apply blur at max strength
        if ks % 2 == 0:
            ks += 1
        sigma = strength

        if use_bilateral:
            blurred = KF.bilateral_blur(tensor, (ks, ks), sigma, (sigma, sigma))
        else:
            blurred = KF.gaussian_blur2d(tensor, (ks, ks), (sigma, sigma))

        # Blend based on blur_map
        result = tensor * (1 - blur_map) + blurred * blur_map

        # Chromatic aberration
        if chromatic > 0:
            result = self._chromatic_aberration(result, depth, chromatic)

        return result.clamp(0, 1)
