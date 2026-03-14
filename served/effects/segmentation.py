"""Segmentation — MobileNetV3 DeepLabV3 semantic segmentation with audio-reactive effects.

21 COCO class labels. Separates foreground from background with feathered
mask edges. Modes: blur_bg, highlight_fg, stylize_regions, isolate.

Also provides shared_seg_model() — singleton DeepLabV3 for recolor + bg_replace.
"""

import sys
import torch
import torch.nn.functional as F
import kornia.filters as KF

from .audio_reactive import AudioReactiveEngine, get_6bands

# ─── Shared Model Singleton ──────────────────────────────────────────
# DeepLabV3-MobileNetV3-Large loaded once, shared across recolor + bg_replace
_shared_seg = None
_shared_seg_device = None
_shared_seg_mean = None
_shared_seg_std = None

# Cross-frame depth cache: populated by depth_midas for N-1, used by segmentation for N
_cached_depth = None  # [1, 1, H, W] float 0-1 (0=near, 1=far)

# Person mask cache: reuse previous mask when frame is similar (no scene change)
_cached_person_mask = None
_cached_person_frame_hash = None


def shared_seg_model(device):
    """Get the shared DeepLabV3-MobileNetV3-Large model (loaded once).

    Returns: (model, mean_tensor, std_tensor)
    """
    global _shared_seg, _shared_seg_device, _shared_seg_mean, _shared_seg_std

    if _shared_seg is not None and _shared_seg_device == device:
        return _shared_seg, _shared_seg_mean, _shared_seg_std

    sys.stderr.write("kornia-server: loading DeepLabV3-MobileNetV3 (shared)...\n")
    sys.stderr.flush()
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    _shared_seg = deeplabv3_mobilenet_v3_large(weights=weights).to(device).eval()
    for p in _shared_seg.parameters():
        p.requires_grad_(False)
    _shared_seg_device = device
    _shared_seg_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _shared_seg_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    sys.stderr.write("kornia-server: DeepLabV3-MobileNetV3 ready (shared)\n")
    sys.stderr.flush()

    return _shared_seg, _shared_seg_mean, _shared_seg_std


def set_cached_depth(depth_map):
    """Store depth map from previous frame for depth-guided segmentation."""
    global _cached_depth
    _cached_depth = depth_map.detach().clone() if depth_map is not None else None


def get_cached_depth():
    """Get cached depth map from previous frame (or None)."""
    return _cached_depth


def person_mask_refined(tensor, device, threshold=0.5, force_recompute=False):
    """High-quality person mask with edge refinement.

    Multi-scale segmentation + morphological cleanup + alpha matte feathering
    + image-guided edge sharpening (Sobel-based guided refinement).
    Includes frame-similarity caching: reuses mask when frame is similar to previous.
    Returns: mask [1,1,H,W] float32 with clean edges.
    """
    global _cached_person_mask, _cached_person_frame_hash

    model, mean, std = shared_seg_model(device)
    _, _, h, w = tensor.shape

    # Frame similarity check: skip DeepLabV3 if frame is very similar to cached
    # Compute lightweight hash: mean RGB + mean luminance at 64x64
    if not force_recompute and _cached_person_mask is not None:
        small = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
        frame_hash = small.mean(dim=(2, 3)).cpu()  # [1, 3] average color
        if _cached_person_frame_hash is not None:
            diff = (frame_hash - _cached_person_frame_hash).abs().sum().item()
            if diff < 0.08:  # Very similar frame — reuse cached mask
                if _cached_person_mask.shape[2:] == (h, w):
                    return _cached_person_mask
                return F.interpolate(_cached_person_mask, size=(h, w),
                                     mode='bilinear', align_corners=False).clamp(0, 1)

    # Multi-scale inference
    masks = []
    for seg_size in (512, 768):
        resized = F.interpolate(tensor, size=(seg_size, seg_size),
                                mode='bilinear', align_corners=False)
        normed = (resized - mean) / std
        with torch.no_grad():
            output = model(normed)['out']
        probs = torch.softmax(output, dim=1)
        person = probs[:, 15:16, :, :]
        masks.append(F.interpolate(person, size=(h, w),
                                   mode='bilinear', align_corners=False))

    merged = torch.max(masks[0], masks[1])

    # ─── Edge Refinement (alpha matte) ─────────────────────────
    # Inner mask (high confidence) stays hard
    inner = (merged > 0.7).float()
    # Outer mask (lower confidence) for feathered edges
    outer = (merged > 0.25).float()

    # Morphological close on inner mask (fill tiny holes)
    inner = KF.gaussian_blur2d(inner, (5, 5), (1.2, 1.2))
    inner = (inner > 0.4).float()

    # Feathered edge: smooth transition between inner and outer
    inner_soft = KF.gaussian_blur2d(inner, (5, 5), (1.5, 1.5))
    outer_soft = KF.gaussian_blur2d(outer, (9, 9), (3.0, 3.0))

    # Composite: hard center + soft edge
    refined = inner_soft * 0.8 + outer_soft * 0.2

    # ─── Image-Guided Edge Sharpening ─────────────────────────
    # Where the image has strong edges (clothing/background boundary),
    # sharpen the mask transition. Where edges are weak, keep soft.
    gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    sobel = KF.sobel(gray)  # [1, 1, H, W] edge magnitude
    edge_mag = sobel.clamp(0, 1)

    # Normalize edge magnitude
    e_max = edge_mag.max()
    if e_max > 0.01:
        edge_mag = (edge_mag / e_max).clamp(0, 1)

    # Only apply at the mask boundary (transition zone 0.1-0.9)
    boundary = ((refined > 0.1) & (refined < 0.9)).float()

    # At boundary: where image edges are strong → hard threshold the mask
    # Where edges are weak → keep the soft feathered version
    hard_at_edges = (merged > 0.5).float()  # Binary threshold version
    edge_weight = (edge_mag * boundary).clamp(0, 1)
    edge_weight = edge_weight.pow(0.7)  # Boost weak edges

    refined = refined * (1.0 - edge_weight) + hard_at_edges * edge_weight

    # ─── Depth-Guided Refinement (if depth available from prev frame) ───
    # Near objects (low depth) are more likely to be the person.
    # At mask boundary: boost confidence for near pixels, reduce for far.
    depth = get_cached_depth()
    if depth is not None and depth.shape[2:] == refined.shape[2:]:
        # Only apply at boundary (transition zone)
        boundary_d = ((refined > 0.15) & (refined < 0.85)).float()
        if boundary_d.sum() > 0:
            # Near pixels (depth < 0.4) → boost mask, far pixels (depth > 0.6) → reduce
            depth_boost = (0.4 - depth).clamp(0, 0.4) * 2.5  # 0-1, near=1
            depth_reduce = (depth - 0.6).clamp(0, 0.4) * 2.5  # 0-1, far=1
            refined = refined + boundary_d * depth_boost * 0.15
            refined = refined - boundary_d * depth_reduce * 0.10

    result = refined.clamp(0, 1)

    # Cache for frame-similarity reuse
    _cached_person_mask = result.detach().clone()
    small = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
    _cached_person_frame_hash = small.mean(dim=(2, 3)).cpu()

    return result


class SegmentationEffect:
    """Semantic segmentation with audio-reactive foreground/background effects.

    Params:
        mode: str — 'blur_bg' | 'highlight_fg' | 'stylize_regions' | 'isolate' (default: 'blur_bg')
        blur_strength: float — background blur sigma (default: 8.0)
        feather: float — mask edge feathering radius (default: 3.0)
        fg_brightness: float — foreground brightness boost (default: 0.1)
        bg_darken: float — background darkening factor (default: 0.3)
    """

    def __init__(self, device):
        self.device = device
        self._model = None
        self._audio = AudioReactiveEngine()

    def _ensure_model(self):
        if self._model is not None:
            return
        sys.stderr.write("kornia-server: loading DeepLabV3-MobileNetV3...\n")
        sys.stderr.flush()
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self._model = deeplabv3_mobilenet_v3_large(weights=weights).to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._preprocess = weights.transforms()
        sys.stderr.write("kornia-server: DeepLabV3-MobileNetV3 ready\n")
        sys.stderr.flush()

    def _get_mask(self, tensor):
        """Get foreground probability mask (person class + common foreground objects)."""
        self._ensure_model()
        _, _, h, w = tensor.shape

        # DeepLabV3 expects ImageNet-normalized input at 520x520
        proc_size = 256  # Smaller for speed
        resized = F.interpolate(tensor, size=(proc_size, proc_size), mode='bilinear', align_corners=False)

        # Normalize for ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        normed = (resized - mean) / std

        with torch.no_grad():
            output = self._model(normed)['out']  # (1, 21, H, W)

        # Foreground = person(15) + common objects: cat(8), dog(12), car(7), bike(2), bird(3)
        fg_classes = [15, 8, 12, 7, 2, 3, 14, 13, 6, 1]  # person, animals, vehicles, etc.
        probs = torch.softmax(output, dim=1)
        fg_prob = sum(probs[:, c:c+1, :, :] for c in fg_classes).clamp(0, 1)

        # Upscale mask back to original resolution
        mask = F.interpolate(fg_prob, size=(h, w), mode='bilinear', align_corners=False)
        return mask

    def _feather_mask(self, mask, radius):
        """Soften mask edges with gaussian blur."""
        if radius <= 0:
            return mask
        k = int(radius) * 2 + 1
        sigma = float(radius)
        return KF.gaussian_blur2d(mask, (k, k), (sigma, sigma))

    def process(self, tensor, params):
        mode = params.get('mode', 'blur_bg')
        blur_strength = float(params.get('blur_strength', 8.0))
        feather = float(params.get('feather', 3.0))
        fg_brightness = float(params.get('fg_brightness', 0.1))
        bg_darken = float(params.get('bg_darken', 0.3))

        # Audio-reactive: bass drives bg blur, beats flash fg
        bands = get_6bands(params)
        beat_pulse = float(params.get('beat_pulse', 0))
        bass = self._audio.smooth('bass', bands['bass'] + bands['sub_bass'])
        beat = self._audio.smooth('beat', beat_pulse, attack=0.5, release=0.08)

        blur_strength += bass * 6.0  # Bass widens background blur
        fg_brightness += beat * 0.15  # Beats flash foreground

        # Get foreground mask
        mask = self._get_mask(tensor)
        mask = self._feather_mask(mask, feather)

        if mode == 'blur_bg':
            # Blur background, keep foreground sharp
            ks = max(3, int(blur_strength) * 2 + 1)
            bg_blurred = KF.gaussian_blur2d(tensor, (ks, ks), (blur_strength, blur_strength))
            result = tensor * mask + bg_blurred * (1 - mask)

        elif mode == 'highlight_fg':
            # Brighten foreground, darken background
            fg = (tensor + fg_brightness).clamp(0, 1)
            bg = tensor * (1.0 - bg_darken)
            result = fg * mask + bg * (1 - mask)

        elif mode == 'stylize_regions':
            # Different color grading for fg vs bg
            import kornia.enhance as KE
            fg = KE.adjust_saturation(tensor, 1.3 + beat * 0.3)
            bg = KE.adjust_saturation(tensor, 0.3)
            bg = KE.adjust_brightness(bg, -0.1)
            result = fg * mask + bg * (1 - mask)

        elif mode == 'isolate':
            # Black background, foreground only
            result = tensor * mask

        else:
            result = tensor

        return result.clamp(0, 1)
