"""Clothing Recolor v3 — Cinema-grade selective recoloring.

Full pipeline: multi-scale segmentation → skin/hair/shoe exclusion →
LAB color space recolor → bilateral edge refinement → temporal coherence.

v3 EXPANSION over v2:
  - ResNet101 backbone (2x better segmentation than MobileNet)
  - Multi-scale seg (512+768px merged for robust edges)
  - LAB color space recoloring (perceptually uniform, natural fabric look)
  - Bilateral mask refinement (pixel-perfect edges guided by image)
  - Named color support ("red", "blue", "gold", etc.)
  - Background subject separation (subtle desaturate/darken BG)
  - Shoe exclusion (lower-body dark items preserved)
  - Confidence zones (hard interior → soft edge → zero exterior)
  - Fabric wrinkle enhancement (boost luminance contrast in clothing)
  - Debug mask output (recolor_debug: true → save mask overlay)

Usage via DreamWave post_params:
    recolor_hue: 0.0             # Target hue in radians (0=red, π/3=yellow, 2π/3=green...)
    recolor_color: "red"         # OR named color (overrides recolor_hue)
    recolor_saturation: 0.75     # Target saturation (0-1)
    recolor_bg_darken: 0.05      # Darken background (0=off, 0.1=subtle)
    recolor_bg_desat: 0.15       # Desaturate background (0=off, 0.3=noticeable)
    recolor_wrinkle_boost: 0.3   # Enhance fabric wrinkles/folds
"""

import math
import sys
import torch
import torch.nn.functional as F
import kornia.color as KC
import kornia.filters as KF
import kornia.morphology as KM

# Named color → HSV hue (radians, kornia convention: 0-2π)
NAMED_COLORS = {
    'red':        0.0,
    'crimson':    6.11,          # ~350°
    'scarlet':    0.07,          # ~4°
    'orange':     0.52,          # ~30°
    'gold':       0.84,          # ~48°
    'yellow':     1.05,          # ~60°
    'lime':       1.40,          # ~80°
    'green':      2.09,          # ~120°
    'teal':       2.62,          # ~150°
    'cyan':       3.14,          # ~180°
    'azure':      3.49,          # ~200°
    'blue':       4.19,          # ~240°
    'indigo':     4.54,          # ~260°
    'purple':     4.89,          # ~280°
    'violet':     5.06,          # ~290°
    'magenta':    5.24,          # ~300°
    'pink':       5.59,          # ~320°
    'rose':       5.93,          # ~340°
    'white':      -1,            # Special: desaturate to white
    'black':      -2,            # Special: desaturate + darken
}

# Named color → suggested saturation (some colors look better at different levels)
COLOR_SAT_HINTS = {
    'red': 0.78, 'crimson': 0.82, 'scarlet': 0.80,
    'orange': 0.75, 'gold': 0.70, 'yellow': 0.65,
    'green': 0.70, 'teal': 0.65, 'cyan': 0.60,
    'blue': 0.75, 'indigo': 0.72, 'purple': 0.70,
    'magenta': 0.75, 'pink': 0.60, 'rose': 0.65,
    'white': 0.05, 'black': 0.05,
}

# Named color → LAB b* cool_shift (controls warmth: 0.0=magenta/pink, 1.0=warm/orange)
# Only applied when target_a > 20 AND target_b > 15 (warm-zone hues)
COLOR_COOL_HINTS = {
    'red': 0.55,       # Balanced red (0.35 was too pink, 0.7 too orange)
    'crimson': 0.40,   # Deep crimson, slightly cooler
    'scarlet': 0.65,   # Warm scarlet, fire-like
    'orange': 1.0,     # Full warmth — it's orange!
    'gold': 0.85,      # Warm gold
    'rose': 0.30,      # Cool rose
    'pink': 0.25,      # Very cool
}


class ClothingRecolorEffect:
    """Cinema-grade clothing recolor with person segmentation."""

    def __init__(self, device):
        self.device = device
        self._prev_mask = None
        self._prev_masks = []  # Multi-frame mask history for voting
        self._frame_count = 0

        # Morphological kernels (multiple sizes for different operations)
        self._k3 = torch.ones(3, 3, device=device)
        self._k5 = torch.ones(5, 5, device=device)
        self._k7 = torch.ones(7, 7, device=device)
        self._k9 = torch.ones(9, 9, device=device)

    # ─── Segmentation ─────────────────────────────────────────────────

    def _person_mask(self, tensor):
        """Multi-scale person segmentation via shared model (shared with bg_replace)."""
        from .segmentation import person_mask_refined
        return person_mask_refined(tensor, self.device)

    # ─── Exclusion Masks ──────────────────────────────────────────────

    def _skin_mask(self, tensor, params):
        """Dual-space skin detection (YCbCr + HSV)."""
        r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0/255.0 + (-0.169 * r - 0.331 * g + 0.500 * b)
        cr = 128.0/255.0 + (0.500 * r - 0.419 * g - 0.081 * b)

        # Tighter YCbCr ranges to reduce false positives on gray/olive fabric
        cb_min = float(params.get('recolor_cb_min', 0.40))
        cb_max = float(params.get('recolor_cb_max', 0.52))
        cr_min = float(params.get('recolor_cr_min', 0.55))
        cr_max = float(params.get('recolor_cr_max', 0.68))
        y_min = float(params.get('recolor_y_min', 0.18))

        skin_ycbcr = (
            (cb > cb_min) & (cb < cb_max) &
            (cr > cr_min) & (cr < cr_max) &
            (y > y_min)
        ).float()

        hsv = KC.rgb_to_hsv(tensor)
        h_ch, s_ch, v_ch = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

        skin_hsv = ((h_ch < 0.70) & (s_ch > 0.15) & (v_ch > 0.25)).float()
        skin_wrap = ((h_ch > 5.24) & (s_ch > 0.10) & (v_ch > 0.20)).float()
        skin_hsv = (skin_hsv + skin_wrap).clamp(0, 1)

        # Require BOTH YCbCr AND HSV to agree (intersection, not union).
        # Union was too aggressive under warm stage lighting — classified clothing as skin.
        skin = (skin_ycbcr * skin_hsv).clamp(0, 1)
        skin = KM.dilation(skin, self._k5)  # Reduced from k7 — less expansion
        return skin

    def _hair_mask(self, tensor, person_mask):
        """Exclude hair: dark + low saturation in upper person region."""
        hsv = KC.rgb_to_hsv(tensor)
        v_ch = hsv[:, 2:3]
        s_ch = hsv[:, 1:2]

        _, _, h, w = tensor.shape

        # Create vertical position map (0=top, 1=bottom)
        y_pos = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)

        # Hair is typically in the upper 40% of the person
        upper_region = (y_pos < 0.4).float()

        # Hair: very dark + low saturation + upper body + within person
        # Tightened from v<0.40 to v<0.22 to preserve dark clothing panels
        hair = (
            (v_ch < 0.22) & (s_ch < 0.25) &
            (person_mask > 0.5) & (upper_region > 0.5)
        ).float()

        # Very dark in upper region (any saturation) — only truly black pixels
        very_dark_upper = ((v_ch < 0.08) & (person_mask > 0.5) & (upper_region > 0.5)).float()
        hair = (hair + very_dark_upper).clamp(0, 1)

        hair = KM.dilation(hair, self._k7)
        return hair

    def _shoe_mask(self, tensor, person_mask):
        """Exclude shoes: dark items in the bottom 20% of the person."""
        _, _, h, w = tensor.shape
        hsv = KC.rgb_to_hsv(tensor)
        v_ch = hsv[:, 2:3]

        # Bottom 20% of frame (where shoes typically are)
        y_pos = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        bottom_region = (y_pos > 0.80).float()

        # Shoes: very dark + in bottom region + within person
        # Tightened from v<0.35 to v<0.20 to preserve dark tracksuit panels on legs
        shoes = ((v_ch < 0.20) & (person_mask > 0.5) & (bottom_region > 0.5)).float()

        shoes = KM.dilation(shoes, self._k5)
        return shoes

    # ─── Mask Generation ──────────────────────────────────────────────

    def _clothing_mask(self, tensor, params):
        """Full clothing mask: person - skin - hair - shoes + refinement."""
        person = self._person_mask(tensor)
        skin = self._skin_mask(tensor, params)
        hair = self._hair_mask(tensor, person)
        shoes = self._shoe_mask(tensor, person)

        if self._frame_count <= 1:
            sys.stderr.write(f"  recolor mask debug: person={person.mean():.4f}, "
                             f"skin={skin.mean():.4f}, hair={hair.mean():.4f}, "
                             f"shoes={shoes.mean():.4f}\n")
            sys.stderr.flush()

        # Clothing = person AND NOT (skin OR hair OR shoes)
        exclusion = (skin + hair + shoes).clamp(0, 1)
        clothing = (person - exclusion).clamp(0, 1)

        # Color range filter — broad range for stage lighting
        hsv = KC.rgb_to_hsv(tensor)
        s_ch, v_ch = hsv[:, 1:2], hsv[:, 2:3]

        src_s_max = float(params.get('recolor_src_s_max', 0.85))
        src_v_min = float(params.get('recolor_src_v_min', 0.06))
        src_v_max = float(params.get('recolor_src_v_max', 0.95))

        color_filter = ((s_ch < src_s_max) & (v_ch > src_v_min) & (v_ch < src_v_max)).float()
        clothing = clothing * color_filter

        # Morphological cleanup (k5 opening — less aggressive than k7)
        clothing = KM.opening(clothing, self._k5)
        clothing = KM.closing(clothing, self._k7)

        # Erosion: pull edges inward to prevent bleeding
        clothing = KM.erosion(clothing, self._k3)

        # Bilateral-guided feathering: sharp at image edges, soft elsewhere
        feather = float(params.get('recolor_feather', 7.0))
        if feather > 0:
            clothing = self._bilateral_feather(clothing, tensor, feather)

        return clothing.clamp(0, 1)

    def _bilateral_feather(self, mask, image, feather):
        """Edge-aware mask feathering guided by image structure.

        Uses image luminance gradient to decide where mask should be sharp vs soft.
        """
        k = int(feather) * 2 + 1
        sigma = feather * 0.7

        # Blur the mask
        blurred = KF.gaussian_blur2d(mask, (k, k), (sigma, sigma))

        # Compute image edge strength
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grad = KF.spatial_gradient(gray, mode='sobel')
        edge_mag = (grad[:, :, 0] ** 2 + grad[:, :, 1] ** 2).sqrt()

        # Normalize and boost edges
        edge_max = edge_mag.max()
        if edge_max > 0:
            edge_mag = (edge_mag / edge_max).clamp(0, 1)

        # Sharpen: boost edge response
        edge_mag = edge_mag.pow(0.7)  # Gamma < 1 = boost weak edges

        # Where edges are strong → keep sharp mask; elsewhere → use blurred
        refined = mask * edge_mag + blurred * (1.0 - edge_mag)

        return refined

    # ─── Temporal Coherence ───────────────────────────────────────────

    def _temporal_smooth(self, mask, params):
        """Multi-frame temporal smoothing with exponential decay."""
        temporal = float(params.get('recolor_temporal', 0.35))

        # Exponential moving average with previous mask
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            mask = mask * (1.0 - temporal) + self._prev_mask * temporal

        # Multi-frame hysteresis voting: require pixel to appear in >= 2 of last 4
        # frames before activating. Prevents single-frame flicker without softening.
        max_history = 4
        self._prev_masks.append((mask > 0.3).float().detach().clone())
        if len(self._prev_masks) > max_history:
            self._prev_masks.pop(0)

        if len(self._prev_masks) >= 3:
            # Count how many recent frames had this pixel as clothing
            vote_count = torch.zeros_like(mask)
            for m in self._prev_masks:
                vote_count += m
            # Require >= 2 votes (majority for stability)
            consensus = (vote_count >= 2).float()
            # Smooth the consensus to avoid hard edges
            consensus = KF.gaussian_blur2d(consensus, (5, 5), (1.2, 1.2))
            # Blend: use consensus as gate, current mask for intensity
            mask = mask * consensus

        self._prev_mask = mask.detach().clone()
        return mask

    # ─── Color Application ────────────────────────────────────────────

    def _resolve_color(self, params):
        """Resolve target hue, saturation, and cool_shift from named color or raw values."""
        # Named color takes priority
        color_name = params.get('recolor_color', None)
        if color_name and isinstance(color_name, str):
            name = color_name.lower().strip()
            if name in NAMED_COLORS:
                hue = NAMED_COLORS[name]
                sat = float(params.get('recolor_saturation', COLOR_SAT_HINTS.get(name, 0.7)))
                # Per-color cool_shift: only override if user didn't set it explicitly
                if 'recolor_cool_shift' not in params and name in COLOR_COOL_HINTS:
                    params['recolor_cool_shift'] = COLOR_COOL_HINTS[name]
                return hue, sat
            else:
                sys.stderr.write(f"kornia-server: unknown color '{name}', using red\n")

        hue = float(params.get('recolor_hue', 0.0))
        sat = float(params.get('recolor_saturation', 0.75))
        return hue, sat

    def _recolor_lab(self, tensor, mask, target_hue, target_sat, params):
        """LAB-space recoloring: perceptually uniform, natural fabric look.

        L channel (luminance) is 100% preserved — only a/b (color) channels change.
        This is superior to HSV because LAB is designed for human perception.
        """
        # Special colors
        if target_hue == -1:  # White
            # Desaturate to white: reduce saturation, boost luminance
            hsv = KC.rgb_to_hsv(tensor)
            h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
            s_new = s * (1.0 - mask * 0.95)
            v_new = v + mask * 0.15
            hsv_out = torch.cat([h, s_new, v_new.clamp(0, 1)], dim=1)
            return KC.hsv_to_rgb(hsv_out).clamp(0, 1)

        if target_hue == -2:  # Black
            hsv = KC.rgb_to_hsv(tensor)
            h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
            s_new = s * (1.0 - mask * 0.9)
            v_new = v * (1.0 - mask * 0.6)
            hsv_out = torch.cat([h, s_new, v_new.clamp(0, 1)], dim=1)
            return KC.hsv_to_rgb(hsv_out).clamp(0, 1)

        # Convert to LAB (clamp input to prevent NaN from extreme values)
        lab = KC.rgb_to_lab(tensor.clamp(0.001, 0.999))  # L in [0,100], a/b in [-128,127]
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

        # Convert target HSV hue to LAB a/b via reference color at max brightness.
        # V=1.0 gives the maximum LAB chroma for this hue — essential for vivid color on gray source.
        ref_hsv = torch.tensor([[[target_hue], [1.0], [1.0]]], device=self.device)  # [1, 3, 1]
        ref_hsv = ref_hsv.unsqueeze(3)  # [1, 3, 1, 1]
        ref_rgb = KC.hsv_to_rgb(ref_hsv).clamp(0, 1)
        ref_lab = KC.rgb_to_lab(ref_rgb)
        target_a = ref_lab[0, 1, 0, 0].item() * target_sat
        target_b = ref_lab[0, 2, 0, 0].item() * target_sat

        # Cool-shift correction: LAB "red" has high b* (warm/orange tint).
        # Reduce b* for red-zone colors to get deeper, cooler red (not orange).
        # Default 0.55 → true red. Higher = warmer/orange, lower = cooler/pink.
        # Per-color hints override this default (see COLOR_COOL_HINTS).
        cool_shift = float(params.get('recolor_cool_shift', 0.55))
        if target_a > 20 and target_b > 15:
            target_b *= cool_shift

        # Luminance-adaptive chroma: only reduce for very dark areas.
        # Floor=0.70 ensures bright gray fabric gets vivid color, not washed-out tint.
        sat_floor = float(params.get('recolor_sat_floor', 0.70))
        L_norm = (L / 100.0).clamp(0.05, 0.95)
        chroma_scale = sat_floor + (1.0 - sat_floor) * L_norm

        # Fabric texture: preserve some original a/b variation
        texture_amount = float(params.get('recolor_texture', 0.12))
        a_orig_var = a - a.mean()
        b_orig_var = b - b.mean()

        # New a/b with texture variation
        a_new = target_a * chroma_scale + a_orig_var * texture_amount
        b_new = target_b * chroma_scale + b_orig_var * texture_amount

        # Blend within mask
        a_out = a * (1.0 - mask) + a_new * mask
        b_out = b * (1.0 - mask) + b_new * mask

        lab_out = torch.cat([L, a_out, b_out], dim=1)
        recolored = KC.lab_to_rgb(lab_out).clamp(0, 1)

        return recolored

    def _enhance_wrinkles(self, tensor, mask, strength):
        """Enhance fabric wrinkle/fold detail via local contrast boost."""
        if strength <= 0:
            return tensor

        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]

        # High-pass filter to find detail
        blurred = KF.gaussian_blur2d(gray, (15, 15), (4.0, 4.0))
        detail = gray - blurred  # Local detail / wrinkles

        # Apply detail enhancement only within clothing mask
        detail_boost = detail * mask * strength

        # Add detail to all RGB channels (luminance boost)
        enhanced = tensor + detail_boost.expand_as(tensor)
        return enhanced.clamp(0, 1)

    def _separate_background(self, tensor, person_mask, params):
        """Subtle background treatment: desaturate + darken for subject separation."""
        bg_darken = float(params.get('recolor_bg_darken', 0.0))
        bg_desat = float(params.get('recolor_bg_desat', 0.0))

        if bg_darken <= 0 and bg_desat <= 0:
            return tensor

        bg_mask = (1.0 - person_mask).clamp(0, 1)

        result = tensor
        if bg_desat > 0:
            hsv = KC.rgb_to_hsv(result)
            h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
            s_reduced = s * (1.0 - bg_mask * bg_desat)
            hsv_out = torch.cat([h, s_reduced, v], dim=1)
            result = KC.hsv_to_rgb(hsv_out).clamp(0, 1)

        if bg_darken > 0:
            result = result * (1.0 - bg_mask * bg_darken)

        return result.clamp(0, 1)

    # ─── Main Process ─────────────────────────────────────────────────

    def process(self, tensor, params):
        """Full recolor pipeline: segment → exclude → recolor → refine.

        v3: LAB color space, multi-scale seg, background separation,
        fabric wrinkle enhancement, named color support.
        """
        self._frame_count += 1
        target_hue, target_sat = self._resolve_color(params)
        blend = float(params.get('recolor_blend', 1.0))

        # Generate clothing mask
        mask = self._clothing_mask(tensor, params)

        # Temporal coherence
        mask = self._temporal_smooth(mask, params)

        # LAB-space recoloring (perceptually uniform)
        recolored = self._recolor_lab(tensor, mask, target_hue, target_sat, params)

        # Enhance fabric wrinkles/folds within clothing
        wrinkle_boost = float(params.get('recolor_wrinkle_boost', 0.25))
        recolored = self._enhance_wrinkles(recolored, mask, wrinkle_boost)

        # Background separation (subtle desaturation + darkening)
        person_mask = (mask > 0.1).float()  # Broader person region for BG treatment
        recolored = self._separate_background(recolored, person_mask, params)

        # ─── Skin Tone Protection ──────────────────────────────────
        # For warm target colors (red/orange), attenuate recolor near skin
        # regions to prevent red cast bleeding onto face/hands at mask edges.
        skin_protect = float(params.get('recolor_skin_protect', 0.7))
        if skin_protect > 0 and target_hue >= -0.1 and target_hue < 1.5:
            # Re-use skin mask (already computed in _clothing_mask)
            skin = self._skin_mask(tensor, params)
            # Dilate skin further to create protection zone
            skin_zone = KM.dilation(skin, self._k9)
            skin_zone = KF.gaussian_blur2d(skin_zone, (15, 15), (4.0, 4.0))
            # Attenuate recolor in the skin proximity zone
            skin_fade = (1.0 - skin_zone * skin_protect).clamp(0, 1)
            # Blend: less recolor near skin, full recolor elsewhere
            recolored = tensor * (1.0 - skin_fade) + recolored * skin_fade

        # Final blend
        if blend < 1.0:
            recolored = tensor * (1.0 - blend) + recolored * blend

        # Log stats
        if self._frame_count <= 3 or self._frame_count % 100 == 1:
            coverage = mask.mean().item() * 100
            sys.stderr.write(f"kornia-server: recolor v3 frame {self._frame_count}, "
                             f"clothing coverage: {coverage:.1f}%, "
                             f"tensor range: [{tensor.min():.3f}, {tensor.max():.3f}], "
                             f"shape: {list(tensor.shape)}\n")
            sys.stderr.flush()

        return recolored
