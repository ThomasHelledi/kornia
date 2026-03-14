"""DreamWave — Atlas' signature audio-reactive neural effect pipeline.

The core insight: neural effects (DeepDream, EdgeGlow, etc.) need visual content
to amplify. Dark procedural frames give them nothing to work with. DreamWave
solves this with intelligent adaptive pre-processing + layered effect chains.

Pipeline:
  1. Adaptive luminance boost — analyze frame brightness, lift dark frames
  2. Color grade warm-up — establish color palette before neural processing
  3. Edge glow — extract and bloom structural detail (Canny + HSV)
  4. DeepDream — hallucinate features from the now-visible content
  5. Depth blur — focus attention with radial/depth blur
  6. Final grade — contrast/saturation polish

Each stage is audio-reactive. Bass drives dream intensity, beats trigger
edge glow pulses, spectral centroid shifts color temperature.
"""

import torch
import torch.nn.functional as F
import kornia.enhance as KE
import kornia.filters as KF
import kornia.color as KC

from .audio_reactive import AudioReactiveEngine, get_6bands, get_spectral
from .cinema_styles import get_cinema_style
from .scene_detect import SceneDetector


# DreamWave intensity presets
DREAMWAVE_PRESETS = {
    'subtle': {
        'pre_boost': 0.06, 'dream_lr': 0.005, 'dream_iter': 8,
        'edge_blend': 0.3, 'depth_strength': 2.0, 'saturation': 1.1,
        'temporal_momentum': 0.25,
    },
    'medium': {
        'pre_boost': 0.10, 'dream_lr': 0.008, 'dream_iter': 12,
        'edge_blend': 0.5, 'depth_strength': 3.0, 'saturation': 1.2,
        'temporal_momentum': 0.3,
    },
    'intense': {
        'pre_boost': 0.15, 'dream_lr': 0.012, 'dream_iter': 18,
        'edge_blend': 0.7, 'depth_strength': 4.0, 'saturation': 1.4,
        'temporal_momentum': 0.35,
    },
    'transcendent': {
        'pre_boost': 0.20, 'dream_lr': 0.015, 'dream_iter': 25,
        'edge_blend': 0.9, 'depth_strength': 5.0, 'saturation': 1.6,
        'temporal_momentum': 0.4,
    },
    # Lite: skip DeepDream entirely for ~10x faster processing (~0.05s/frame).
    # Perfect for video input where source material already has rich texture.
    'lite': {
        'pre_boost': 0.08, 'dream_lr': 0, 'dream_iter': 0,
        'edge_blend': 0.4, 'depth_strength': 2.5, 'saturation': 1.15,
        'temporal_momentum': 0.2,
    },
    # Lite-Style: skip DeepDream but enable VGG19 style transfer.
    # Ideal for video-input where you want style transfer without hallucination.
    # ~2-4s/frame depending on style_ref complexity.
    'lite-style': {
        'pre_boost': 0.06, 'dream_lr': 0, 'dream_iter': 0,
        'edge_blend': 0.35, 'depth_strength': 2.0, 'saturation': 1.1,
        'chain_mix': 0.7,
        'temporal_momentum': 0.25,
    },
}


class DreamWaveEffect:
    """Atlas' signature multi-layer audio-reactive neural pipeline.

    Params:
        preset: str — 'subtle' | 'medium' | 'intense' | 'transcendent' (default: 'medium')
        dream_layer: str — DeepDream layer preset (default: 'soft')
        adapt_brightness: bool — auto-boost dark frames (default: true)
        chain_mix: float — 0=clean, 1=full DreamWave (default: 0.85)
        edge_color: [R,G,B] — edge glow tint (default: [100, 180, 255])
        focus_mode: str — 'radial' | 'vertical' | 'none' (default: 'radial')
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()
        self._dream = None
        self._edge = None
        self._depth = None
        self._color = None
        self._style_transfer = None
        self._glitch = None
        self._warp = None
        self._recolor = None
        self._bg_replace = None
        self._depth_midas = None
        self._scene = SceneDetector()
        self._prev_result = None  # Cached previous result for smart frame skip
        self._edge_guide_cache = {}  # path → tensor cache for edge guidance
        self._prev_grain = None  # Previous grain pattern for temporal grain coherence

    def _ensure_effects(self):
        """Lazy-load sub-effects."""
        if self._dream is None:
            from .deepdream import DeepDreamEffect
            from .edge_glow import EdgeGlowEffect
            from .depth_blur import DepthBlurEffect
            from .color_grade import ColorGradeEffect
            self._dream = DeepDreamEffect(self.device)
            self._edge = EdgeGlowEffect(self.device)
            self._depth = DepthBlurEffect(self.device)
            self._color = ColorGradeEffect(self.device)

    def _adaptive_brightness(self, tensor, target_mean=0.35, max_boost=0.25):
        """Intelligently boost dark frames so neural effects have content to work with.

        Analyzes frame luminance and applies graduated boost. Brighter frames
        get less/no boost. Dark frames get significant lift.

        Args:
            tensor: (1, 3, H, W) input
            target_mean: desired mean luminance (0-1)
            max_boost: maximum additive brightness

        Returns:
            (boosted_tensor, boost_amount)
        """
        gray = KC.rgb_to_grayscale(tensor)
        current_mean = gray.mean().item()

        if current_mean >= target_mean:
            return tensor, 0.0

        # Graduated boost: more boost for darker frames
        deficit = target_mean - current_mean
        boost = min(deficit * 1.5, max_boost)

        # Apply gamma correction + additive boost for natural-looking lift
        # Gamma < 1.0 lifts shadows, > 1.0 crushes them
        gamma = max(0.5, 1.0 - boost * 2.0)
        boosted = KE.adjust_gamma(tensor, gamma)
        boosted = (boosted + boost * 0.5).clamp(0, 1)

        return boosted, boost

    def should_process_frame(self, tensor, params):
        """Smart frame skip — decide whether this frame needs full neural processing.

        Uses audio energy, beat detection, and scene change to decide.
        Call before process() to implement audio-adaptive frame skip.

        Args:
            tensor: (1, 3, H, W) current frame
            params: dict with audio features

        Returns:
            (should_process: bool, reason: str)
        """
        bands = get_6bands(params)
        spectral = get_spectral(params)
        beat_pulse = float(params.get('beat_pulse', 0))

        # Scene change — always process (don't interpolate across cuts)
        is_scene_change, change_score = self._scene.check(tensor)
        if is_scene_change:
            return True, 'scene_change'

        # Beat/onset — process for visual impact
        if beat_pulse > 0.3:
            return True, 'beat'
        if spectral['flux'] > 0.4:
            return True, 'onset'

        # Audio energy — skip when nothing interesting
        total_energy = (bands['sub_bass'] + bands['bass'] + bands['low_mid'] +
                        bands['mid'] + bands['high_mid'] + bands['treble']) / 6.0
        if total_energy < 0.08:
            return False, 'low_energy'

        # Default: process
        return True, 'default'

    def _content_edge_density(self, tensor):
        """Compute edge density for content-aware intensity modulation.

        High edges (faces, buildings) → preserve detail (reduce dream).
        Low edges (sky, water) → more hallucination (increase dream).

        Args:
            tensor: (1, 3, H, W) float tensor

        Returns:
            density: float 0-1 (fraction of pixels that are edges)
        """
        gray = KC.rgb_to_grayscale(tensor)
        # Canny-like edge detection via Sobel magnitude
        sobel = KF.sobel(gray)
        magnitude = sobel.squeeze()
        # Threshold at 0.1 to count "edge pixels"
        edge_count = (magnitude > 0.1).float().mean().item()
        return edge_count

    def process(self, tensor, params):
        self._ensure_effects()

        # Load preset
        preset_name = params.get('preset', 'medium')
        preset = dict(DREAMWAVE_PRESETS.get(preset_name, DREAMWAVE_PRESETS['medium']))

        # Cinema style: merge style params on top of preset defaults
        cinema_style_name = params.get('cinema_style', '')
        if cinema_style_name:
            style_params = get_cinema_style(cinema_style_name)
            if style_params:
                preset.update({k: v for k, v in style_params.items()
                               if k != 'description'})

        # Smart frame skip (F2): return blended previous result on skip
        smart_skip = params.get('smart_skip', False)
        if smart_skip and self._prev_result is not None:
            should, reason = self.should_process_frame(tensor, params)
            if not should:
                # Blend 85% cached + 15% current for smooth transitions
                blended = self._prev_result * 0.85 + tensor * 0.15
                return blended.clamp(0, 1)

        # Inject temporal_momentum from preset into params for registry
        if 'temporal_momentum' not in params:
            params['temporal_momentum'] = preset.get('temporal_momentum', 0.3)

        # Scene change detection: signal reset for temporal blender
        is_scene_change, _ = self._scene.check(tensor)
        if is_scene_change:
            params['_scene_change'] = True

        # Explicit params override everything
        chain_mix = float(params.get('chain_mix', preset.get('chain_mix', 0.85)))
        adapt = params.get('adapt_brightness', True)
        focus_mode = params.get('focus_mode', preset.get('focus_mode', 'radial'))
        dream_layer = params.get('dream_layer', preset.get('dream_layer', 'soft'))
        edge_color = params.get('edge_color', preset.get('edge_color', [100, 180, 255]))

        # Audio features
        bands = get_6bands(params)
        spectral = get_spectral(params)
        beat_pulse = float(params.get('beat_pulse', 0))
        bpm = float(params.get('bpm', 0))
        time = float(params.get('time', 0))
        section = params.get('section', '')

        bass = self._audio.smooth('bass', bands['bass'] + bands['sub_bass'],
                                   attack=0.25, release=0.04)
        beat = self._audio.smooth('beat', beat_pulse, attack=0.5, release=0.06)
        centroid = self._audio.smooth('centroid', spectral['centroid'],
                                      attack=0.15, release=0.03)
        flux = spectral['flux']

        # ━━━ Stage 0.0: Logo/watermark removal (edge-fill inpainting) ━━━━━━
        logo_rect = params.get('logo_remove', None)
        if logo_rect:
            _, _, h, w = tensor.shape
            # Normalized coords [x1, y1, x2, y2] → pixel coords with margin
            margin = 4
            lx1 = max(0, int(float(logo_rect[0]) * w) - margin)
            ly1 = max(0, int(float(logo_rect[1]) * h) - margin)
            lx2 = min(w, int(float(logo_rect[2]) * w) + margin)
            ly2 = min(h, int(float(logo_rect[3]) * h) + margin)

            if lx2 > lx1 and ly2 > ly1:
                mask = torch.zeros(1, 1, h, w, device=self.device)
                mask[:, :, ly1:ly2, lx1:lx2] = 1.0

                # Use bg_replace's proven edge-fill inpainting (much cleaner than blur-fill)
                if self._bg_replace is None:
                    from .bg_replace import BackgroundReplaceEffect
                    self._bg_replace = BackgroundReplaceEffect(self.device)
                tensor = self._bg_replace._inpaint_edge_fill(tensor, mask, max_iterations=15)

        # ━━━ Stage 0.1: Clothing Recolor (before bg_replace for vivid ghost) ━━━
        recolor_active = params.get('recolor_hue', None) is not None or params.get('recolor_color', None) is not None
        if recolor_active:
            if self._recolor is None:
                from .recolor import ClothingRecolorEffect
                self._recolor = ClothingRecolorEffect(self.device)
            tensor = self._recolor.process(tensor, params)

        # ━━━ Stage 0.2: Background Replace (after recolor for vivid ghost) ━━━
        bg_mode = params.get('bg_mode', None)
        if bg_mode:
            if self._bg_replace is None:
                from .bg_replace import BackgroundReplaceEffect
                self._bg_replace = BackgroundReplaceEffect(self.device)
            tensor = self._bg_replace.process(tensor, params)

        # Preserve clean frame for final mix
        clean = tensor.clone()

        # ━━━ Stage 0.5: Edge Guidance (preserve reference composition) ━━━
        edge_ref = params.get('edge_ref', '')
        edge_strength = float(params.get('edge_strength', 0.3))
        if edge_ref:
            edge_map = self._load_edge_guide(edge_ref, tensor.shape[2], tensor.shape[3])
            if edge_map is not None:
                # Edges from reference suppress neural effects → preserve structure
                tensor = tensor * (1.0 - edge_strength) + tensor * edge_map * edge_strength

        # ━━━ Stage 1: Adaptive Brightness ━━━
        if adapt:
            tensor, boost_amount = self._adaptive_brightness(
                tensor,
                target_mean=0.30 + bass * 0.1,  # Bass lifts the shadows
                max_boost=preset['pre_boost'] * (1.0 + beat * 0.5)
            )

        # ━━━ Content-aware intensity modulation ━━━
        # High edges (faces, detail) → reduce dream. Low edges (sky, abstract) → more dream.
        edge_density = self._content_edge_density(tensor)
        # Map density to modifier: 0.0 edges → +0.3, 0.3+ edges → -0.3
        intensity_mod = 1.0 + 0.3 * (1.0 - min(edge_density / 0.15, 2.0))
        preset['dream_iter'] = int(preset['dream_iter'] * intensity_mod)
        preset['dream_lr'] = preset['dream_lr'] * intensity_mod

        # ━━━ Stage 2: Pre-grade (warm up colors before neural processing) ━━━
        if not preset.get('skip_color_grade', False):
            color_params = {
                'saturation': preset['saturation'] + bass * 0.15,
                'contrast': 1.05 + beat * 0.1,
                'brightness': beat * 0.03,
                # Centroid drives color temperature: low=warm, high=cool
                'hue': (centroid - 0.5) * 0.12,
                'beat_pulse': 0, 'bass_energy': 0, 'audio_intensity': 0,  # We handle audio ourselves
            }
            tensor = self._color.process(tensor, color_params)

        # ━━━ Stage 3: Edge Glow (structural detail before dream) ━━━
        if preset['edge_blend'] > 0:
            edge_params = {
                'threshold_low': 0.08,
                'threshold_high': 0.25,
                'glow_radius': 3 + int(bass * 5),
                'glow_color': edge_color,
                'blend': preset['edge_blend'] * (0.6 + beat * 0.8),
                'hsv_glow': True,
                'morph_cleanup': True,
                'beat_pulse': 0, 'bass_energy': 0, 'high_energy': 0,
                'spectral_centroid': centroid,
                'spectral_rolloff': spectral['rolloff'],
                'spectral_flux': flux,
                'spectral_flatness': spectral['flatness'],
            }
            tensor = self._edge.process(tensor, edge_params)

        # ━━━ Stage 3.2: Glitch (beat-synced digital corruption, optional) ━━━
        if params.get('glitch', False) and beat > 0.3:
            if self._glitch is None:
                from .glitch import GlitchEffect
                self._glitch = GlitchEffect(self.device)
            glitch_params = {
                'beat_pulse': beat_pulse, 'spectral_flux': flux,
                'spectral_centroid': centroid, 'spectral_rolloff': spectral['rolloff'],
                'spectral_flatness': spectral['flatness'],
                'channel_shift': 6.0 * beat, 'block_chance': 0.2 * beat,
                'time': time,
            }
            glitched = self._glitch.process(tensor, glitch_params)
            # Max 40% blend to keep it subtle
            glitch_blend = min(0.4, beat * 0.5)
            tensor = tensor * (1.0 - glitch_blend) + glitched * glitch_blend

        # ━━━ Stage 3.5: VGG19 Style Transfer (optional) ━━━
        style_ref = params.get('style_ref', '')
        if style_ref:
            if self._style_transfer is None:
                from .style_transfer import StyleTransferEffect
                self._style_transfer = StyleTransferEffect(self.device)
            style_weight = 5e5 * (1.0 + bass * 0.8)  # Bass increases style influence
            # Temporal coherence: high stability during quiet, responsive during beats
            # beat=0 → temporal_weight=1.5e4 (stable), beat=1 → temporal_weight=3e3 (responsive)
            temporal_weight = 1.5e4 * (1.0 - beat * 0.8)
            style_params = {
                'style_image': style_ref,
                'content_weight': 1.0,
                'style_weight': style_weight,
                'temporal_weight': temporal_weight,
                'iterations': 15,  # Lightweight: 15 not 50
                'lr': 0.04,
                # Resolution scaling: optimize at 320px max → ~4x faster
                # VGG19 cost ∝ H×W, so 320px vs 960px = ~9x fewer pixels
                'process_resolution': 320,
                # Smooth crossfade over 5 frames when style_ref changes between scenes
                'transition_frames': 5,
            }
            tensor = self._style_transfer.process(tensor, style_params)

        # ━━━ Stage 3.7: Warp (bass-driven geometry distortion, optional) ━━━
        warp_mode = params.get('warp_mode', preset.get('warp_mode', 'wave'))
        if params.get('warp', False):
            if self._warp is None:
                from .warp import WarpEffect
                self._warp = WarpEffect(self.device)
            warp_params = {
                'mode': warp_mode,
                'strength': 0.02 + bass * 0.04,
                'beat_pulse': beat_pulse, 'time': time,
                'sub_bass': bands['sub_bass'], 'bass': bands['bass'],
                'low_mid': bands['low_mid'], 'mid': bands['mid'],
                'high_mid': bands['high_mid'], 'treble': bands['treble'],
            }
            tensor = self._warp.process(tensor, warp_params)

        # ━━━ Stage 4: DeepDream (hallucinate on now-visible content) ━━━
        # Skip when dream_iter == 0 (lite preset) — ~10x faster
        if preset['dream_iter'] > 0:
            dream_params = {
                'layer': dream_layer,
                'iterations': int(preset['dream_iter'] * (0.7 + bass * 0.6)),
                'lr': preset['dream_lr'] * (1.0 + beat * 0.4),
                'octaves': 3,
                'octave_scale': 1.3,
                'jitter': 24,
                'loss_fn': 'norm',
                'beat_pulse': 0, 'audio_intensity': 0,  # We control it
                'bpm': bpm, 'time': time,
                'spectral_flux': flux,
            }
            # Save pre-dream tensor for person protection and edge-aware blending
            pre_dream = tensor.clone()

            # Resolution scaling: optimize dream at lower res for ~3-4x speedup
            dream_res = int(params.get('dream_resolution', 0))
            orig_h, orig_w = tensor.shape[2], tensor.shape[3]
            if dream_res > 0 and max(orig_h, orig_w) > dream_res:
                scale = dream_res / max(orig_h, orig_w)
                small_h = max(64, int(orig_h * scale) - int(orig_h * scale) % 2)
                small_w = max(64, int(orig_w * scale) - int(orig_w * scale) % 2)
                tensor_small = F.interpolate(tensor, size=(small_h, small_w),
                                             mode='bilinear', align_corners=False)
                dreamed_small = self._dream.process(tensor_small, dream_params)
                dreamed = F.interpolate(dreamed_small, size=(orig_h, orig_w),
                                        mode='bilinear', align_corners=False)
                # Blend: mostly dreamed + some original detail preservation
                tensor = dreamed * 0.85 + tensor * 0.15
            else:
                tensor = self._dream.process(tensor, dream_params)

            # ── Edge-Aware Dream Blending ──────────────────────────────
            # Reduce dream intensity in high-detail regions (edges, textures)
            # and preserve full intensity in uniform regions (walls, sky).
            # This naturally protects detailed objects while enhancing flat surfaces.
            edge_dream = float(params.get('edge_aware_dream', preset.get('edge_aware_dream', 0)))
            if edge_dream > 0:
                gray = 0.299 * pre_dream[:, 0:1] + 0.587 * pre_dream[:, 1:2] + 0.114 * pre_dream[:, 2:3]
                edges = KF.sobel(gray)
                # Normalize edge magnitude
                e_max = edges.max()
                if e_max > 0.01:
                    edges = (edges / e_max).clamp(0, 1)
                # Smooth edges to create broad "detail density" regions
                edges = KF.gaussian_blur2d(edges, (25, 25), (6.0, 6.0)).clamp(0, 1)
                # Invert: high edges → low dream, low edges → high dream
                # edge_dream controls the strength of this modulation
                dream_weight = (1.0 - edges * edge_dream).clamp(0.15, 1.0)
                tensor = tensor * dream_weight + pre_dream * (1.0 - dream_weight)

            # ── Person Protection: restore person region from pre-dream frame ──
            # Uses DeepLabV3 person mask to keep performer clean while background
            # gets full neural effects. Crucial for video-input with performers.
            person_protect = float(params.get('person_protect', preset.get('person_protect', 0)))
            if person_protect > 0 and pre_dream is not None:
                from .segmentation import person_mask_refined
                import kornia.morphology as KM
                pmask = person_mask_refined(pre_dream, self.device)
                # Dilate mask to expand person boundary buffer zone
                dilate_k = torch.ones(7, 7, device=self.device)
                pmask = KM.dilation(pmask, dilate_k)
                # Smooth edges for clean blending
                pmask = KF.gaussian_blur2d(pmask, (11, 11), (3.0, 3.0)).clamp(0, 1)
                # Blend: person region uses pre_dream (clean), background uses dreamed
                tensor = tensor * (1.0 - pmask * person_protect) + pre_dream * (pmask * person_protect)

        # ━━━ Stage 5: Depth Focus (guide viewer's eye) ━━━
        if focus_mode != 'none':
            depth_params = {
                'depth_mode': focus_mode,
                'focus_depth': 0.4 - bass * 0.15,  # Bass shifts focus
                'blur_strength': preset['depth_strength'],
                'sigmoid_falloff': 6.0,
                'bilateral': True,
                'kernel_size': 7,
                'chromatic': 0.02 + beat * 0.04,  # Subtle chromatic on beats
                'beat_pulse': 0, 'bass_energy': 0, 'audio_intensity': 0,
            }
            tensor = self._depth.process(tensor, depth_params)

        # ━━━ Stage 5.5: MiDaS Real Depth (optional, ML monocular depth) ━━━
        if params.get('depth_real', False) or params.get('depth_bokeh', 0) > 0 or params.get('depth_fog', 0) > 0:
            if self._depth_midas is None:
                from .depth_midas import DepthMidasEffect
                self._depth_midas = DepthMidasEffect(self.device)
            tensor = self._depth_midas.process(tensor, params)

        # ━━━ Stage 6: Final Polish (auto-exposure CLAHE) ━━━
        if not preset.get('skip_clahe', False):
            # Auto-exposure: adapt CLAHE clip_limit to scene brightness
            # Dark scenes → stronger CLAHE (2.5) to reveal detail
            # Bright scenes → gentle CLAHE (1.0) to avoid washing out
            gray_mean = KC.rgb_to_grayscale(tensor).mean().item()
            if gray_mean < 0.25:
                clip = 2.5  # Dark scene: strong local contrast
            elif gray_mean > 0.65:
                clip = 1.0  # Bright scene: gentle
            else:
                clip = 1.8 - (gray_mean - 0.25) * 2.0  # Gradient between
            clip = float(params.get('clahe_clip', clip))
            tensor = KE.equalize_clahe(tensor, clip_limit=clip)

        # ━━━ Stage 6.5: Film Post (cinema-grade finishing) ━━━
        film_grain = float(params.get('film_grain', preset.get('film_grain', 0)))
        vignette_strength = float(params.get('vignette', preset.get('vignette', 0)))
        halation_strength = float(params.get('halation', preset.get('halation', 0)))
        split_tone = params.get('split_tone', preset.get('split_tone', False))
        lift = params.get('lift', preset.get('lift', None))
        gain = params.get('gain', preset.get('gain', None))

        # Film grain: temporally coherent noise (50% new + 50% previous for natural grain movement)
        if film_grain > 0:
            h, w = tensor.shape[2], tensor.shape[3]
            noise = torch.randn(1, 1, h, w, device=self.device) * film_grain
            # Temporal grain coherence: blend with previous grain so it doesn't buzz
            if self._prev_grain is not None and self._prev_grain.shape == noise.shape:
                noise = noise * 0.6 + self._prev_grain * 0.4  # Mostly new but partially stable
            self._prev_grain = noise.detach().clone()
            # Luminance-aware: less grain in shadows, more in midtones
            luma = KC.rgb_to_grayscale(tensor)
            grain_mask = (luma * (1.0 - luma) * 4.0).clamp(0, 1)  # Peak at 0.5
            # Audio-reactive: beats increase grain slightly
            grain_intensity = 1.0 + beat * 0.3
            tensor = (tensor + noise * grain_mask * grain_intensity).clamp(0, 1)

        # S-curve contrast: cinematic tone curve (gentler than CLAHE)
        s_curve_strength = float(params.get('s_curve', preset.get('s_curve', 0)))
        if s_curve_strength > 0:
            # S-curve via smooth sigmoid: shadows darken, highlights brighten
            # Audio-reactive: beats slightly increase contrast
            curve_amount = s_curve_strength * (1.0 + beat * 0.15)
            midpoint = 0.5
            # Apply per-channel: x' = 1 / (1 + exp(-k * (x - mid))) normalized
            k = 4.0 + curve_amount * 8.0  # Steepness: 4-12
            curved = torch.sigmoid(k * (tensor - midpoint))
            # Normalize: sigmoid at 0 and 1 aren't exactly 0 and 1
            low = torch.sigmoid(torch.tensor(k * (0.0 - midpoint), device=self.device))
            high = torch.sigmoid(torch.tensor(k * (1.0 - midpoint), device=self.device))
            curved = (curved - low) / (high - low)
            tensor = curved.clamp(0, 1)

        # Vignette: warm-tinted radial darkening from edges, audio-reactive
        if vignette_strength > 0:
            h, w = tensor.shape[2], tensor.shape[3]
            y = torch.linspace(-1, 1, h, device=self.device).view(-1, 1)
            x = torch.linspace(-1, 1, w, device=self.device).view(1, -1)
            radius = (x * x + y * y).sqrt()
            # Smooth falloff: 1.0 at center, 0.0 at corners
            vignette_mask = (1.0 - (radius * 0.7).clamp(0, 1).pow(2.0))
            vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            # Audio-reactive: bass makes vignette breathe (slightly lighter on bass hits)
            vig_reactive = vignette_strength * (1.0 - bass * 0.2)
            # Warm tint: edges go slightly warm instead of pure black
            # Creates cinematic warmth in the vignette fall-off zone
            edge_zone = (1.0 - vignette_mask).clamp(0, 1)
            warm_tint = torch.zeros(1, 3, 1, 1, device=self.device)
            warm_tint[0, 0, 0, 0] = 0.03   # Subtle red warmth
            warm_tint[0, 1, 0, 0] = 0.01   # Minimal green
            warm_tint[0, 2, 0, 0] = -0.02  # Cool down blue in edges
            tensor = tensor * (1.0 - vig_reactive + vig_reactive * vignette_mask)
            tensor = (tensor + warm_tint * edge_zone * vig_reactive).clamp(0, 1)

        # Halation: scene-aware glow from bright areas (film stock light scatter)
        if halation_strength > 0:
            luma = KC.rgb_to_grayscale(tensor)
            bright_mask = (luma - 0.7).clamp(0, 1) * 3.0  # Threshold at 0.7

            # Anamorphic mode: horizontal streaks instead of circular bloom
            anamorphic = params.get('anamorphic', preset.get('anamorphic', False))
            if anamorphic:
                # Wide horizontal blur (anamorphic lens flare look)
                halo = KF.gaussian_blur2d(tensor * bright_mask, (7, 61), (1.5, 16.0))
            else:
                # Standard circular bloom
                halo = KF.gaussian_blur2d(tensor * bright_mask, (31, 31), (8.0, 8.0))

            # Dynamic halation tint: derive from scene's dominant bright color
            # This makes halation integrate naturally (warm stage → warm halo, blue stage → blue halo)
            bright_area = tensor * bright_mask
            bright_sum = bright_area.sum(dim=(2, 3), keepdim=True)
            bright_count = bright_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            avg_bright = bright_sum / bright_count  # [1, 3, 1, 1]
            # Boost saturation of the halation tint
            tint = (avg_bright * 1.3 + 0.1).clamp(0, 1.5)
            halo_tinted = halo * tint
            # Audio-reactive: bass pulses the halation
            tensor = (tensor + halo_tinted * halation_strength * (1.0 + bass * 0.5)).clamp(0, 1)

        # Split-toning: cool shadows + warm highlights (luminance-preserving)
        if split_tone:
            luma = KC.rgb_to_grayscale(tensor)
            # Shadow tint (blue-ish): below 0.4 luminance
            shadow_mask = (0.4 - luma).clamp(0, 0.4) / 0.4
            # Highlight tint (warm): above 0.6 luminance
            highlight_mask = (luma - 0.6).clamp(0, 0.4) / 0.4
            # Apply tints
            shadow_tint = torch.tensor([0.85, 0.9, 1.15], device=self.device).view(1, 3, 1, 1)
            highlight_tint = torch.tensor([1.12, 1.05, 0.88], device=self.device).view(1, 3, 1, 1)
            st_strength = 0.3  # Subtle
            tensor = tensor * (1.0 - shadow_mask * st_strength) + tensor * shadow_tint * shadow_mask * st_strength
            tensor = tensor * (1.0 - highlight_mask * st_strength) + tensor * highlight_tint * highlight_mask * st_strength
            # Restore original luminance (split-tone should only shift hue, not brightness)
            luma_after = KC.rgb_to_grayscale(tensor)
            luma_ratio = (luma / luma_after.clamp(min=1e-6)).clamp(0.8, 1.2)
            tensor = (tensor * luma_ratio).clamp(0, 1)

        # Lift/Gain: 3-way color correction (shadows lift, highlights gain)
        if lift is not None:
            lift_t = torch.tensor(lift, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            tensor = (tensor + lift_t * 0.1).clamp(0, 1)
        if gain is not None:
            gain_t = torch.tensor(gain, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            tensor = (tensor * (1.0 + gain_t * 0.15)).clamp(0, 1)

        # Shadow/highlight recovery: prevent crushed blacks and blown whites
        shadow_recovery = float(params.get('shadow_recovery', preset.get('shadow_recovery', 0)))
        highlight_recovery = float(params.get('highlight_recovery', preset.get('highlight_recovery', 0)))
        if shadow_recovery > 0 or highlight_recovery > 0:
            luma = KC.rgb_to_grayscale(tensor)
            if shadow_recovery > 0:
                # Lift only the darkest pixels (below 0.1) without affecting mid/highlights
                shadow_mask = (0.1 - luma).clamp(0, 0.1) * 10.0  # 1.0 at black, 0.0 at 0.1+
                tensor = (tensor + shadow_mask * shadow_recovery * 0.15).clamp(0, 1)
            if highlight_recovery > 0:
                # Pull down only the brightest pixels (above 0.9)
                highlight_mask = (luma - 0.9).clamp(0, 0.1) * 10.0  # 1.0 at white, 0.0 at 0.9-
                tensor = (tensor - highlight_mask * highlight_recovery * 0.15).clamp(0, 1)

        # Chromatic aberration: subtle RGB channel offset for film lens effect
        chroma_ab = float(params.get('chromatic_aberration', preset.get('chromatic_aberration', 0)))
        if chroma_ab > 0:
            h, w = tensor.shape[2], tensor.shape[3]
            # Offset red outward, blue inward from center
            shift_px = max(1, int(chroma_ab * min(h, w)))
            # Red channel: shift outward (scale up slightly)
            r_zoom = 1.0 + chroma_ab * 0.02
            r_h, r_w = int(h / r_zoom), int(w / r_zoom)
            r_y0, r_x0 = (h - r_h) // 2, (w - r_w) // 2
            r_ch = F.interpolate(tensor[:, 0:1, r_y0:r_y0 + r_h, r_x0:r_x0 + r_w],
                                 size=(h, w), mode='bilinear', align_corners=False)
            # Blue channel: shift inward (scale down slightly + pad)
            b_zoom = 1.0 - chroma_ab * 0.02
            b_h, b_w = int(h / b_zoom), int(w / b_zoom)
            b_y0, b_x0 = (b_h - h) // 2, (b_w - w) // 2
            b_resized = F.interpolate(tensor[:, 2:3], size=(b_h, b_w),
                                      mode='bilinear', align_corners=False)
            b_ch = b_resized[:, :, b_y0:b_y0 + h, b_x0:b_x0 + w]
            tensor = torch.cat([r_ch, tensor[:, 1:2], b_ch], dim=1).clamp(0, 1)

        # Anamorphic letterbox: cinema 2.39:1 crop with soft black bars
        letterbox = params.get('letterbox', preset.get('letterbox', False))
        if letterbox:
            h, w = tensor.shape[2], tensor.shape[3]
            target_ratio = float(params.get('letterbox_ratio', preset.get('letterbox_ratio', 2.39)))
            current_ratio = w / h
            if current_ratio < target_ratio:
                # Add bars top/bottom to reach wider ratio
                visible_h = int(w / target_ratio)
                bar_h = (h - visible_h) // 2
                if bar_h > 0:
                    # Soft fade into bars (8px gradient) instead of hard cut
                    fade_px = min(8, bar_h)
                    fade = torch.linspace(0, 1, fade_px, device=self.device)
                    # Top bar
                    tensor[:, :, :bar_h - fade_px, :] = 0
                    for i in range(fade_px):
                        tensor[:, :, bar_h - fade_px + i, :] *= fade[i]
                    # Bottom bar
                    tensor[:, :, h - bar_h + fade_px:, :] = 0
                    for i in range(fade_px):
                        tensor[:, :, h - bar_h + i, :] *= fade[fade_px - 1 - i]

        # ━━━ Stage 7: Breath/Pulse Zoom (BPM-synced subtle zoom) ━━━
        breath_amount = float(params.get('breath', preset.get('breath', 0)))
        if breath_amount > 0 and bpm > 0:
            import math
            # Oscillate at half BPM for breathing feel (inhale/exhale)
            phase = math.sin(2 * math.pi * (bpm / 120.0) * time)
            # Zoom range: 1.0 ± breath_amount (e.g. 0.02 = 2% zoom)
            zoom = 1.0 + phase * breath_amount * (1.0 + bass * 0.5)
            h, w = tensor.shape[2], tensor.shape[3]
            zh, zw = int(h / zoom), int(w / zoom)
            if zh > 0 and zw > 0 and zh < h and zw < w:
                # Center crop
                y0 = (h - zh) // 2
                x0 = (w - zw) // 2
                cropped = tensor[:, :, y0:y0 + zh, x0:x0 + zw]
                tensor = F.interpolate(cropped, size=(h, w),
                                       mode='bilinear', align_corners=False)

        # ━━━ Mix: blend DreamWave chain with clean frame ━━━
        # Audio-reactive mix: beats push toward full chain
        effective_mix = chain_mix * (0.8 + beat * 0.2)
        result = clean * (1.0 - effective_mix) + tensor * effective_mix

        result = result.clamp(0, 1)

        # Cache for smart frame skip
        self._prev_result = result.detach().clone()

        return result

    def _load_edge_guide(self, path, h, w):
        """Load and cache an edge guide map, resized to match frame dimensions.

        Args:
            path: path to grayscale edge map PNG
            h: target height
            w: target width

        Returns:
            (1, 1, H, W) float tensor or None
        """
        cache_key = (path, h, w)
        if cache_key in self._edge_guide_cache:
            return self._edge_guide_cache[cache_key]

        try:
            from PIL import Image
            import torchvision.transforms.functional as TF
            img = Image.open(path).convert('L')
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)  # (1, 1, H, W)
            tensor = F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
            self._edge_guide_cache[cache_key] = tensor
            return tensor
        except Exception:
            return None
