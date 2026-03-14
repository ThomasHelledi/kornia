"""Watermark Removal — detect and remove static semi-transparent text overlays.

Two-phase approach:
  Phase 1 (warmup): Collect N frames, compute temporal median to find static pixels.
                     Static + bright pixels = watermark. Build binary mask.
  Phase 2 (active): For each frame, iteratively inpaint the masked region using
                     Gaussian blur diffusion. Fast because mask is small (~5% of frame).

Supports explicit ROI to limit detection to a specific region (e.g., top-right corner).
"""

import sys
import torch
import kornia.filters as KF
import kornia.morphology as KM


class WatermarkRemovalEffect:
    """Remove static semi-transparent watermarks from video frames.

    Params:
        warmup_frames: int — frames to collect before building mask (default: 30)
        var_threshold: float — max temporal variance for watermark pixels (default: 0.008)
        bright_threshold: float — min brightness for watermark pixels (default: 0.5)
        inpaint_iterations: int — Gaussian inpainting iterations (default: 8)
        roi: list[float] — [x0_frac, y0_frac, x1_frac, y1_frac] region of interest
                            (default: full frame). Example: [0.5, 0.0, 1.0, 0.33] = top-right
        mask_dilate: int — dilation iterations to expand mask over anti-aliased edges (default: 2)
    """

    def __init__(self, device):
        self.device = device
        self._warmup_frames = 30
        self._frame_buffer = []
        self._mask = None  # (1, 1, H, W) float mask, 0=clean 1=watermark
        self._frame_count = 0

    def _build_mask(self, params):
        """Build watermark mask from temporal statistics of collected frames."""
        var_thresh = float(params.get('var_threshold', 0.008))
        bright_thresh = float(params.get('bright_threshold', 0.5))
        dilate_iters = int(params.get('mask_dilate', 2))
        roi = params.get('roi', None)  # [x0, y0, x1, y1] as fractions

        # Stack frames: (N, 3, H, W)
        stack = torch.stack(self._frame_buffer, dim=0)
        N, C, H, W = stack.shape

        # Temporal variance per pixel (low = static = watermark candidate)
        temporal_var = stack.var(dim=0).mean(dim=0, keepdim=True)  # (1, H, W)

        # Mean brightness across time
        brightness = stack.mean(dim=0).mean(dim=0, keepdim=True)  # (1, H, W)

        # Watermark: low temporal variance AND high brightness
        mask = ((temporal_var < var_thresh) & (brightness > bright_thresh)).float()  # (1, H, W)

        # Apply ROI constraint (zero out everything outside)
        if roi is not None and len(roi) == 4:
            x0 = int(roi[0] * W)
            y0 = int(roi[1] * H)
            x1 = int(roi[2] * W)
            y1 = int(roi[3] * H)
            roi_mask = torch.zeros_like(mask)
            roi_mask[:, y0:y1, x0:x1] = 1.0
            mask = mask * roi_mask

        mask = mask.unsqueeze(0)  # (1, 1, H, W)

        # Morphological cleanup: dilate to cover anti-aliased edges
        kernel = torch.ones(5, 5, device=self.device)
        for _ in range(dilate_iters):
            mask = KM.dilation(mask, kernel)

        # Smooth mask edges for natural blending
        mask = KF.gaussian_blur2d(mask, (11, 11), (3.0, 3.0))
        mask = mask.clamp(0, 1)

        if mask.max() < 0.05:
            sys.stderr.write("kornia-server: watermark removal — no watermark detected\n")
            sys.stderr.flush()
            self._mask = None
        else:
            pixel_count = (mask > 0.5).sum().item()
            pct = pixel_count / (H * W) * 100
            sys.stderr.write(f"kornia-server: watermark mask built — {pixel_count} pixels ({pct:.1f}% of frame)\n")
            sys.stderr.flush()
            self._mask = mask

        self._frame_buffer = []  # Free memory

    def _inpaint(self, frame, mask, iterations=8):
        """Iterative Gaussian inpainting: fill masked region from surroundings.

        Each iteration blurs the frame and replaces masked pixels with blurred values.
        Progressively propagates surrounding content inward.
        """
        result = frame.clone()
        inv_mask = 1.0 - mask

        for i in range(iterations):
            k = 7 + i * 4  # Increasing kernel: 7, 11, 15, 19, 23, 27, 31, 35
            if k % 2 == 0:
                k += 1
            sigma = k / 3.0
            blurred = KF.gaussian_blur2d(result, (k, k), (sigma, sigma))
            result = result * inv_mask + blurred * mask

        return result

    def process(self, tensor, params):
        """Process a frame: collect during warmup, then remove watermark."""
        self._frame_count += 1
        self._warmup_frames = int(params.get('warmup_frames', 30))
        iterations = int(params.get('inpaint_iterations', 8))

        # Warmup phase: collect frames
        if self._mask is None and len(self._frame_buffer) < self._warmup_frames:
            self._frame_buffer.append(tensor.squeeze(0).clone())

            if len(self._frame_buffer) >= self._warmup_frames:
                self._build_mask(params)

            return tensor  # Pass through during warmup

        # No watermark detected
        if self._mask is None:
            return tensor

        # Active phase: inpaint watermark
        return self._inpaint(tensor, self._mask, iterations)
