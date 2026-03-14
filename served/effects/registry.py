"""Effect registry — dispatches effect IDs to implementations."""

import json
import sys
import numpy as np
import torch


EFFECT_PASSTHROUGH = 0x00
EFFECT_DEEPDREAM = 0x01
EFFECT_STYLE_TRANSFER = 0x02
EFFECT_EDGE_GLOW = 0x03
EFFECT_DEPTH_BLUR = 0x04
EFFECT_SEGMENTATION = 0x05
EFFECT_SUPER_RES = 0x06
EFFECT_COLOR_GRADE = 0x07
EFFECT_MORPHOLOGY = 0x08
EFFECT_DREAM_SEQUENCE = 0x09
EFFECT_GLITCH = 0x0A
EFFECT_WARP = 0x0B
EFFECT_HALFTONE = 0x0C
EFFECT_DREAMWAVE = 0x0D
EFFECT_RECOLOR = 0x0E
EFFECT_PROJECTION = 0x0F
EFFECT_COSMIC = 0x10
EFFECT_WATERMARK_REMOVE = 0x11


class EffectRegistry:
    """Lazy-loading effect registry with temporal coherence.

    Effects initialize on first use. Temporal blending smooths output
    across consecutive frames when momentum > 0.
    """

    def __init__(self, device):
        self.device = device
        self._effects = {}
        self._temporal = None

    def _ensure_temporal(self, momentum):
        """Lazy-init temporal blender."""
        if self._temporal is None and momentum > 0:
            from .temporal import TemporalBlender
            self._temporal = TemporalBlender(self.device, momentum)
            sys.stderr.write(f"kornia-server: temporal blending enabled (momentum={momentum})\n")
            sys.stderr.flush()
        elif self._temporal is not None and momentum > 0:
            self._temporal.momentum = momentum

    def _get_effect(self, effect_id):
        if effect_id in self._effects:
            return self._effects[effect_id]

        if effect_id == EFFECT_DEEPDREAM:
            from .deepdream import DeepDreamEffect
            eff = DeepDreamEffect(self.device)
        elif effect_id == EFFECT_EDGE_GLOW:
            from .edge_glow import EdgeGlowEffect
            eff = EdgeGlowEffect(self.device)
        elif effect_id == EFFECT_COLOR_GRADE:
            from .color_grade import ColorGradeEffect
            eff = ColorGradeEffect(self.device)
        elif effect_id == EFFECT_STYLE_TRANSFER:
            from .style_transfer import StyleTransferEffect
            eff = StyleTransferEffect(self.device)
        elif effect_id == EFFECT_DEPTH_BLUR:
            from .depth_blur import DepthBlurEffect
            eff = DepthBlurEffect(self.device)
        elif effect_id == EFFECT_MORPHOLOGY:
            from .morphology_effect import MorphologyEffect
            eff = MorphologyEffect(self.device)
        elif effect_id == EFFECT_SEGMENTATION:
            from .segmentation import SegmentationEffect
            eff = SegmentationEffect(self.device)
        elif effect_id == EFFECT_SUPER_RES:
            from .super_resolution import SuperResEffect
            eff = SuperResEffect(self.device)
        elif effect_id == EFFECT_GLITCH:
            from .glitch import GlitchEffect
            eff = GlitchEffect(self.device)
        elif effect_id == EFFECT_WARP:
            from .warp import WarpEffect
            eff = WarpEffect(self.device)
        elif effect_id == EFFECT_HALFTONE:
            from .halftone import HalftoneEffect
            eff = HalftoneEffect(self.device)
        elif effect_id == EFFECT_DREAMWAVE:
            from .dreamwave import DreamWaveEffect
            eff = DreamWaveEffect(self.device)
        elif effect_id == EFFECT_RECOLOR:
            from .recolor import ClothingRecolorEffect
            eff = ClothingRecolorEffect(self.device)
        elif effect_id == EFFECT_PROJECTION:
            from .projection import ProjectionEffect
            eff = ProjectionEffect(self.device)
        elif effect_id == EFFECT_COSMIC:
            from .cosmic import CosmicEffect
            eff = CosmicEffect(self.device)
        elif effect_id == EFFECT_WATERMARK_REMOVE:
            from .watermark_removal import WatermarkRemovalEffect
            eff = WatermarkRemovalEffect(self.device)
        elif effect_id == EFFECT_DREAM_SEQUENCE:
            # Chain: deepdream + edge_glow + color_grade
            from .deepdream import DeepDreamEffect
            from .edge_glow import EdgeGlowEffect
            from .color_grade import ColorGradeEffect
            eff = ChainedEffect(self.device, [
                DeepDreamEffect(self.device),
                EdgeGlowEffect(self.device),
                ColorGradeEffect(self.device),
            ])
        else:
            return None

        self._effects[effect_id] = eff
        return eff

    def process(self, effect_id, rgba_bytes, w, h, params_json):
        """Process a single RGBA frame. Returns RGBA bytes."""
        if effect_id == EFFECT_PASSTHROUGH:
            return rgba_bytes

        params = json.loads(params_json) if params_json else {}

        # RGBA bytes → numpy (H, W, 4) → torch (1, 3, H, W) float [0,1]
        arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(h, w, 4).copy()
        rgb = arr[:, :, :3]  # Drop alpha
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        tensor = tensor.to(self.device)

        # Pre-brightness boost for dark frames (helps DeepDream on dark environments)
        pre_boost = float(params.get('pre_brightness', 0))
        if pre_boost > 0:
            tensor = (tensor + pre_boost).clamp(0, 1)

        # Resolution scaling: process at lower resolution for speed
        process_scale = float(params.get('process_scale', 1.0))
        orig_h, orig_w = tensor.shape[2], tensor.shape[3]
        if 0 < process_scale < 1.0:
            new_h = max(64, int(orig_h * process_scale))
            new_w = max(64, int(orig_w * process_scale))
            tensor = torch.nn.functional.interpolate(
                tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        effect = self._get_effect(effect_id)
        if effect is None:
            return rgba_bytes

        result = effect.process(tensor, params)

        # Upscale: delegate to effect if it handles upscale, else bicubic+unsharp for DreamWave, bilinear for others
        if 0 < process_scale < 1.0:
            if getattr(effect, 'handles_upscale', False):
                result = effect.upscale(result, orig_h, orig_w)
            elif effect_id == EFFECT_DREAMWAVE:
                # DreamWave: bicubic + unsharp mask for sharper upscale
                result = torch.nn.functional.interpolate(
                    result, size=(orig_h, orig_w), mode='bicubic', align_corners=False).clamp(0, 1)
                try:
                    import kornia.filters as KF
                    blurred = KF.gaussian_blur2d(result, (5, 5), (1.5, 1.5))
                    result = (result + 0.4 * (result - blurred)).clamp(0, 1)
                except Exception:
                    pass
            else:
                result = torch.nn.functional.interpolate(
                    result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        # Temporal blending — smooth across frames
        momentum = float(params.get('temporal_momentum', 0))
        if momentum > 0:
            self._ensure_temporal(momentum)
            # Reset temporal blender on scene changes to avoid ghosting
            if params.get('_scene_change', False) and self._temporal is not None:
                self._temporal.reset()
            result = self._temporal.blend(result)

        # torch (1, 3, H, W) → numpy (H, W, 4) RGBA bytes
        out_rgb = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte()
        out_rgb = out_rgb.cpu().numpy()
        out_rgba = np.dstack([out_rgb, arr[:, :, 3:4]])  # Re-attach alpha
        return out_rgba.tobytes()


class ChainedEffect:
    """Run multiple effects in sequence."""

    def __init__(self, device, effects):
        self.device = device
        self.effects = effects

    def process(self, tensor, params):
        for eff in self.effects:
            tensor = eff.process(tensor, params)
        return tensor
