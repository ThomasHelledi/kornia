"""MiDaS Depth — Real monocular depth estimation for cinema-grade effects.

Uses MiDaS DPT-Hybrid via torch.hub for robust depth maps, then applies:
  - Cinematic bokeh (multi-scale depth-of-field)
  - Atmospheric fog (depth-based haze)
  - Depth color grading (warm near, cool far)

Usage via DreamWave post_params:
    depth_real: true               # Enable MiDaS (replaces luminance depth)
    depth_bokeh: 3.0               # Bokeh blur strength (0=off, 5=heavy)
    depth_fog: 0.2                 # Fog density (0=off, 0.5=heavy)
    depth_fog_color: [0.75, 0.78, 0.82]  # Fog RGB (default: cool gray)
    depth_focus: 0.35              # Focus plane (0=near, 1=far, 0.35=person)
    depth_grade: true              # Warm near / cool far color grading
"""

import sys
import torch
import torch.nn.functional as F
import kornia.filters as KF


class DepthMidasEffect:
    """Real monocular depth via MiDaS DPT-Hybrid."""

    def __init__(self, device):
        self.device = device
        self._model = None
        self._frame_count = 0
        self._prev_depth = None

    def _ensure_model(self):
        if self._model is not None:
            return
        sys.stderr.write("kornia-server: loading MiDaS DPT-Hybrid for depth estimation...\n")
        sys.stderr.flush()

        try:
            self._model = torch.hub.load('isl-org/MiDaS', 'DPT_Hybrid',
                                         trust_repo=True, verbose=False)
        except Exception:
            # Fallback: try older repo name
            self._model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid',
                                         trust_repo=True, verbose=False)

        self._model = self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        sys.stderr.write("kornia-server: MiDaS DPT-Hybrid ready\n")
        sys.stderr.flush()

    # ─── Depth Estimation ───────────────────────────────────────────────

    def estimate(self, tensor, params=None):
        """RGB tensor [1,3,H,W] → depth map [1,1,H,W] normalized 0-1.

        Higher values = farther from camera.
        Audio-reactive temporal smoothing: stable during quiet, responsive on beats.
        """
        self._ensure_model()
        _, _, h, w = tensor.shape

        # MiDaS DPT-Hybrid expects 384x384 input, ImageNet normalization
        resized = F.interpolate(tensor, size=(384, 384),
                                mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        normed = (resized - mean) / std

        with torch.no_grad():
            depth = self._model(normed)

        # [1, 384, 384] → [1, 1, 384, 384]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # MiDaS outputs inverse depth (closer = higher values). Invert for our convention.
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = 1.0 - (depth - d_min) / (d_max - d_min)

        # Upscale to original resolution
        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)

        # Audio-reactive temporal smoothing:
        # Quiet → heavy smoothing (0.4) for stable depth. Beats → light smoothing (0.1) for responsive.
        self._frame_count += 1
        beat_pulse = float(params.get('beat_pulse', 0)) if params else 0
        temporal_w = 0.4 - beat_pulse * 0.3  # 0.4 quiet → 0.1 on beats
        temporal_w = max(0.05, temporal_w)    # Never fully raw
        if self._prev_depth is not None and self._prev_depth.shape == depth.shape:
            depth = depth * (1.0 - temporal_w) + self._prev_depth * temporal_w
        self._prev_depth = depth.detach().clone()

        return depth

    # ─── Depth Effects ──────────────────────────────────────────────────

    def _apply_bokeh(self, tensor, depth, strength, focus):
        """Multi-scale cinematic bokeh based on depth distance from focus plane."""
        # Distance from focus → blur intensity
        dist = (depth - focus).abs()
        blur_map = (dist * strength * 2.0).clamp(0, 1)

        # Three blur levels for smooth progressive bokeh
        b1 = KF.gaussian_blur2d(tensor, (9, 9), (2.0, 2.0))
        b2 = KF.gaussian_blur2d(tensor, (21, 21), (5.0, 5.0))
        b3 = KF.gaussian_blur2d(tensor, (41, 41), (10.0, 10.0))

        # Progressive: sharp → slight blur → medium → heavy
        m1 = blur_map.clamp(0, 0.33) * 3.0         # 0-1 for first level
        m2 = (blur_map - 0.33).clamp(0, 0.33) * 3.0  # 0-1 for second
        m3 = (blur_map - 0.66).clamp(0, 0.34) * 3.0   # 0-1 for third

        result = tensor * (1.0 - m1)
        result = result + b1 * (m1 - m2).clamp(0, 1)
        result = result + b2 * (m2 - m3).clamp(0, 1)
        result = result + b3 * m3

        return result

    def _apply_fog(self, tensor, depth, density, color):
        """Atmospheric fog that increases with distance."""
        fog_amount = (depth * density).clamp(0, 0.85)
        return tensor * (1.0 - fog_amount) + color * fog_amount

    def _apply_grade(self, tensor, depth, strength=0.06):
        """Depth-based color grading: warm near, cool far."""
        near = (1.0 - depth).clamp(0, 1) * strength
        far = depth.clamp(0, 1) * strength

        result = tensor.clone()
        result[:, 0:1] += near * 0.8   # Warm reds near camera
        result[:, 1:2] += near * 0.2   # Subtle green warmth
        result[:, 2:3] += far * 0.7    # Cool blues in distance

        return result

    # ─── Main Process ───────────────────────────────────────────────────

    def process(self, tensor, params):
        """Apply MiDaS depth-based effects.

        Estimates real monocular depth, then applies bokeh, fog, and grading.
        """
        depth = self.estimate(tensor, params)

        focus = float(params.get('depth_focus', 0.35))
        result = tensor

        # Cinematic bokeh
        bokeh = float(params.get('depth_bokeh', 0.0))
        if bokeh > 0:
            result = self._apply_bokeh(result, depth, bokeh, focus)

        # Atmospheric fog
        fog = float(params.get('depth_fog', 0.0))
        if fog > 0:
            fog_r = float(params.get('depth_fog_r', 0.75))
            fog_g = float(params.get('depth_fog_g', 0.78))
            fog_b = float(params.get('depth_fog_b', 0.82))
            fog_color = torch.tensor([fog_r, fog_g, fog_b],
                                     device=self.device).view(1, 3, 1, 1)
            result = self._apply_fog(result, depth, fog, fog_color)

        # Depth color grading
        if params.get('depth_grade', False):
            result = self._apply_grade(result, depth)

        # Cache depth map for next frame's segmentation (depth-guided refinement)
        try:
            from .segmentation import set_cached_depth
            set_cached_depth(depth)
        except Exception:
            pass  # Segmentation module might not be loaded yet

        if self._frame_count <= 2:
            sys.stderr.write(f"kornia-server: depth_midas frame {self._frame_count}, "
                             f"depth range: [{depth.min():.3f}, {depth.max():.3f}], "
                             f"bokeh={bokeh}, fog={fog}\n")
            sys.stderr.flush()

        return result.clamp(0, 1)
