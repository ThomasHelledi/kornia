"""Background Replace — Person removal + infinite pattern generation.

Remove humans from video frames via iterative inpainting, then optionally
create infinite repeating patterns from the clean background.

Modes:
    remove      — Remove person, fill with inpainted background
    infinite    — Remove person + create infinite pattern from background
    ghost       — Person at reduced opacity over pattern background

Patterns:
    mirror      — Horizontal mirror (infinite corridor/tunnel effect)
    kaleidoscope — 4-way symmetry (abstract mandala)
    zoom        — Recursive zoom (fractal depth illusion)

Usage via DreamWave post_params:
    bg_mode: "remove"          # remove | infinite | ghost
    bg_pattern: "mirror"       # mirror | kaleidoscope | zoom
    bg_person_alpha: 0.0       # 0=removed, 1=full, 0.3=ghost
    bg_inpaint_iter: 25        # Quality iterations (more=smoother)
    bg_temporal: 0.3           # Temporal smoothing (0=none, 0.5=heavy)
"""

import sys
import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.morphology as KM


class BackgroundReplaceEffect:
    """Remove person and/or replace background with infinite pattern."""

    def __init__(self, device):
        self.device = device
        self._prev_bg = None
        self._frame_count = 0

        self._k3 = torch.ones(3, 3, device=device)
        self._k5 = torch.ones(5, 5, device=device)
        self._k7 = torch.ones(7, 7, device=device)
        self._k11 = torch.ones(11, 11, device=device)
        self._k21 = torch.ones(21, 21, device=device)

    def _person_mask(self, tensor):
        """Multi-scale person segmentation → refined soft mask [0,1].

        Uses shared DeepLabV3 model (shared with recolor effect).
        """
        from .segmentation import person_mask_refined
        return person_mask_refined(tensor, self.device)

    # ─── Inpainting ─────────────────────────────────────────────────────

    def _inpaint_edge_fill(self, tensor, mask, max_iterations=80):
        """Progressive edge-fill: propagate real pixels inward from mask boundary.

        Unlike blur-fill (which averages garbage inside the mask), this method:
        1. Identifies boundary pixels (unknown pixels with known neighbors)
        2. Fills boundary using weighted average of known neighbors ONLY
        3. Marks boundary as known, repeats until mask is fully filled
        4. Adds texture from surrounding background to avoid smooth blob

        Result: clean fill that propagates actual scene content, not blur.
        """
        result = tensor.clone()
        unknown = (mask > 0.5).float()
        original_mask = unknown.clone()

        # Phase 1: Progressive edge-fill (structure propagation)
        for i in range(max_iterations):
            if unknown.sum() == 0:
                break

            # Find boundary: unknown pixels that have known neighbors
            known = 1.0 - unknown
            # Dilate known region by 3px per iteration for speed
            dilated_known = KM.dilation(known, self._k7)
            boundary = ((dilated_known > 0.5) & (unknown > 0.5)).float()

            if boundary.sum() == 0:
                # Remaining unknown pixels have no known neighbors — fill with blur
                if unknown.sum() > 0:
                    blurred = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
                    result = result * (1.0 - unknown) + blurred * unknown
                break

            # Weighted average of known neighbors only
            # Weight = known mask, so unknown pixels contribute 0
            weighted = result * known
            k_size = 7
            sigma = 2.5
            blurred_weighted = KF.gaussian_blur2d(weighted, (k_size, k_size), (sigma, sigma))
            blurred_known = KF.gaussian_blur2d(known, (k_size, k_size), (sigma, sigma))
            blurred_known = blurred_known.clamp(min=1e-6)

            fill_values = blurred_weighted / blurred_known

            # Fill only boundary pixels
            result = result * (1.0 - boundary) + fill_values * boundary
            unknown = unknown * (1.0 - boundary)

        # Phase 2: Texture injection — add high-frequency detail from background
        # Extract texture from the unmasked region and inject into filled region
        bg_only = tensor * (1.0 - original_mask)
        bg_lowfreq = KF.gaussian_blur2d(bg_only, (15, 15), (4.0, 4.0))
        bg_known_w = KF.gaussian_blur2d(1.0 - original_mask, (15, 15), (4.0, 4.0)).clamp(min=1e-6)
        bg_lowfreq = bg_lowfreq / bg_known_w
        texture = (tensor - bg_lowfreq) * (1.0 - original_mask)

        # Multi-directional texture sampling: average 4 shifted copies to avoid
        # copying the person's shape outline into the fill
        _, _, h, w = tensor.shape
        shifts = [(h // 7, w // 5), (-h // 5, w // 3), (h // 4, -w // 6), (-h // 3, -w // 4)]
        combined_texture = torch.zeros_like(texture)
        for sy, sx in shifts:
            combined_texture += torch.roll(texture, shifts=(sy, sx), dims=(2, 3))
        combined_texture = combined_texture / len(shifts)

        # Fade texture near mask edges (avoid hard texture boundaries)
        edge_fade = KF.gaussian_blur2d(original_mask, (21, 21), (6.0, 6.0)).clamp(0, 1)
        inner_mask = (edge_fade > 0.3).float() * original_mask

        # Apply texture at reduced strength
        result = result + combined_texture * inner_mask * 0.5

        # Phase 3: Color/brightness matching — match filled region to surroundings
        # Compute local mean of surrounding (non-mask) region
        surround_zone = KF.gaussian_blur2d(original_mask, (31, 31), (10.0, 10.0))
        surround_ring = ((surround_zone > 0.05) & (surround_zone < 0.5)).float()
        if surround_ring.sum() > 100:
            # Mean color of surrounding ring
            ring_sum = (result * surround_ring).sum(dim=(2, 3), keepdim=True)
            ring_count = surround_ring.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            ring_mean = ring_sum / ring_count

            # Mean color of filled region
            fill_sum = (result * original_mask).sum(dim=(2, 3), keepdim=True)
            fill_count = original_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            fill_mean = fill_sum / fill_count

            # Shift filled region to match surrounding color balance
            color_shift = (ring_mean - fill_mean) * 0.6  # Partial correction
            result = result + color_shift * original_mask

        # Phase 4: Edge-aware seam blending
        # Instead of uniform Gaussian (which softens sharp edges), use the image's
        # own edges to decide where to smooth. Sharp boundaries stay sharp.
        filled_smooth = KF.gaussian_blur2d(result, (5, 5), (1.2, 1.2))
        boundary_zone = KF.gaussian_blur2d(original_mask, (11, 11), (3.0, 3.0))
        boundary_thin = (boundary_zone > 0.1).float() * (boundary_zone < 0.9).float()

        # Edge-aware: detect strong edges and reduce smoothing there
        gray = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
        edge_mag = KF.sobel(gray).clamp(0, 1)
        e_max = edge_mag.max()
        if e_max > 0.01:
            edge_mag = (edge_mag / e_max).clamp(0, 1)
        # Where edges exist → less smoothing (preserve structure)
        smooth_weight = boundary_thin * (1.0 - edge_mag * 0.7)
        result = result * (1.0 - smooth_weight) + filled_smooth * smooth_weight

        return result.clamp(0, 1)

    # ─── Infinite Patterns ──────────────────────────────────────────────

    def _pattern_mirror(self, bg):
        """Horizontal mirror — infinite corridor/tunnel effect."""
        _, _, h, w = bg.shape
        flipped = torch.flip(bg, dims=[3])

        # Wide gradient blend at center for invisible seam
        blend_w = w // 3  # Wide blend zone (was w//5)
        ramp = torch.linspace(0, 1, blend_w, device=self.device).view(1, 1, 1, blend_w)
        # Smooth S-curve for natural blending (linear ramps show as brightness shift)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # Hermite smoothstep
        center = w // 2
        start = center - blend_w // 2

        result = bg.clone()
        sl = slice(start, start + blend_w)
        result[:, :, :, sl] = result[:, :, :, sl] * (1.0 - ramp) + flipped[:, :, :, sl] * ramp
        return result

    def _pattern_kaleidoscope(self, bg):
        """4-way symmetry — abstract mandala from architecture."""
        _, _, h, w = bg.shape
        # Take top-left quadrant, mirror into all 4
        tl = bg[:, :, :h // 2, :w // 2]
        tr = torch.flip(tl, dims=[3])
        bl = torch.flip(tl, dims=[2])
        br = torch.flip(tl, dims=[2, 3])

        top = torch.cat([tl, tr], dim=3)
        bottom = torch.cat([bl, br], dim=3)
        result = torch.cat([top, bottom], dim=2)

        return F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False)

    def _pattern_zoom(self, bg):
        """Recursive zoom — fractal depth illusion."""
        _, _, h, w = bg.shape
        result = bg.clone()

        # Layer scaled copies centered on the image
        for scale, alpha in [(0.75, 0.5), (0.55, 0.3), (0.35, 0.2)]:
            sh, sw = int(h * scale), int(w * scale)
            scaled = F.interpolate(bg, size=(sh, sw), mode='bilinear', align_corners=False)
            pad_h, pad_w = (h - sh) // 2, (w - sw) // 2

            # Blend scaled layer
            region = result[:, :, pad_h:pad_h + sh, pad_w:pad_w + sw]
            result[:, :, pad_h:pad_h + sh, pad_w:pad_w + sw] = (
                region * (1.0 - alpha) + scaled * alpha
            )

        return result

    def _pattern_tile(self, bg):
        """Tiled grid — repeating mosaic from center crop of background."""
        _, _, h, w = bg.shape
        # Extract center quarter as the tile
        th, tw = h // 2, w // 2
        y0, x0 = h // 4, w // 4
        tile = bg[:, :, y0:y0 + th, x0:x0 + tw]

        # Build 2x2 grid
        top = torch.cat([tile, torch.flip(tile, dims=[3])], dim=3)  # Mirror horizontally
        bottom = torch.cat([torch.flip(tile, dims=[2]), torch.flip(tile, dims=[2, 3])], dim=3)
        grid = torch.cat([top, bottom], dim=2)

        return F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)

    def _pattern_radial(self, bg):
        """Radial symmetry — polar coordinate mirror for mandala effect."""
        _, _, h, w = bg.shape

        # Take left half, mirror right
        half_w = w // 2
        left = bg[:, :, :, :half_w]
        right = torch.flip(left, dims=[3])
        mirrored_h = torch.cat([left, right], dim=3)

        # Take top half, mirror bottom
        half_h = h // 2
        top = mirrored_h[:, :, :half_h, :]
        bottom = torch.flip(top, dims=[2])
        result = torch.cat([top, bottom], dim=2)

        # Gentle radial blur for smooth center
        center_mask = torch.zeros(1, 1, h, w, device=self.device)
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=self.device).float().view(-1, 1) - cy
        x = torch.arange(w, device=self.device).float().view(1, -1) - cx
        dist = (y * y + x * x).sqrt() / max(h, w)
        center_mask[0, 0] = (1.0 - dist * 2.0).clamp(0, 0.3)

        blurred = KF.gaussian_blur2d(result, (15, 15), (4.0, 4.0))
        result = result * (1.0 - center_mask) + blurred * center_mask

        return F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False)

    # ─── Main Process ───────────────────────────────────────────────────

    def process(self, tensor, params):
        """Background replacement pipeline.

        Returns the processed tensor with person removed/replaced.
        """
        self._frame_count += 1
        mode = str(params.get('bg_mode', 'remove'))
        pattern_name = str(params.get('bg_pattern', 'mirror'))
        person_alpha = float(params.get('bg_person_alpha', 0.0))
        inpaint_iter = int(params.get('bg_inpaint_iter', 25))
        temporal = float(params.get('bg_temporal', 0.3))

        # Segment person
        person = self._person_mask(tensor)

        # Dilate mask generously for clean removal (no fringing)
        person_dilated = KM.dilation(person, self._k21)
        person_dilated = KF.gaussian_blur2d(person_dilated, (11, 11), (3.0, 3.0))
        person_dilated = person_dilated.clamp(0, 1)

        if self._frame_count <= 2:
            coverage = person.mean().item() * 100
            sys.stderr.write(f"kornia-server: bg_replace frame {self._frame_count}, "
                             f"person: {coverage:.1f}%, mode={mode}, pattern={pattern_name}\n")
            sys.stderr.flush()

        # Adaptive iteration count: smaller masks converge faster
        coverage = person_dilated.mean().item()
        if inpaint_iter > 0:
            if coverage < 0.10:
                inpaint_iter = min(inpaint_iter, 15)  # Small person: fast fill
            elif coverage > 0.35:
                inpaint_iter = max(inpaint_iter, 40)  # Large occlusion: more iterations

        # Inpaint person region (progressive edge-fill)
        bg_clean = self._inpaint_edge_fill(tensor, person_dilated, max_iterations=inpaint_iter)

        # Temporal smoothing for stable background
        if self._prev_bg is not None and self._prev_bg.shape == bg_clean.shape:
            bg_clean = bg_clean * (1.0 - temporal) + self._prev_bg * temporal
        self._prev_bg = bg_clean.detach().clone()

        # Apply pattern if infinite mode
        if mode == 'infinite':
            patterns = {
                'mirror': self._pattern_mirror,
                'kaleidoscope': self._pattern_kaleidoscope,
                'zoom': self._pattern_zoom,
                'tile': self._pattern_tile,
                'radial': self._pattern_radial,
            }
            fn = patterns.get(pattern_name, self._pattern_mirror)
            patterned = fn(bg_clean)
            # Pattern blend: 0=clean bg, 1=full pattern (default 1)
            pattern_blend = float(params.get('bg_pattern_blend', 1.0))
            bg_final = bg_clean * (1.0 - pattern_blend) + patterned * pattern_blend
        else:
            bg_final = bg_clean

        # Composite person back at desired opacity
        if person_alpha > 0.01:
            # Use soft person mask (not dilated) for compositing
            soft_mask = KF.gaussian_blur2d(person, (7, 7), (2.0, 2.0))

            # Ghost mode enhancement: luminance-weighted alpha
            # Brighter parts (face, highlights) more visible, darker (edges) fade
            ghost_3d = params.get('bg_ghost_3d', mode == 'ghost')
            if ghost_3d and person_alpha < 0.95:
                import kornia.color as KC
                luma = KC.rgb_to_grayscale(tensor)  # [1,1,H,W]
                # Map luminance to alpha multiplier: dark→0.5, mid→1.0, bright→1.2
                alpha_mod = (0.5 + luma * 0.7).clamp(0.4, 1.2)
                effective_alpha = soft_mask * person_alpha * alpha_mod
            else:
                effective_alpha = soft_mask * person_alpha

            result = bg_final * (1.0 - effective_alpha) + tensor * effective_alpha

            # Ghost glow: ethereal bloom around person silhouette
            ghost_glow = float(params.get('bg_ghost_glow', 0.15 if mode == 'ghost' else 0))
            if ghost_glow > 0:
                # Audio-reactive: beats pulse the glow intensity
                beat_pulse = float(params.get('beat_pulse', 0))
                glow_reactive = ghost_glow * (1.0 + beat_pulse * 0.8)

                # Wide gaussian of the person mask = soft glow halo
                glow_mask = KF.gaussian_blur2d(person, (31, 31), (8.0, 8.0))
                # Subtract the solid person area → only the halo ring
                glow_ring = (glow_mask - soft_mask * 0.8).clamp(0, 1)
                # Tint glow with average person color
                person_color = (tensor * soft_mask).sum(dim=(2, 3), keepdim=True)
                person_area = soft_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
                avg_color = person_color / person_area
                # Brighten the glow color
                glow_color = (avg_color * 1.5 + 0.2).clamp(0, 1)
                result = result + glow_ring * glow_color * glow_reactive
        else:
            result = bg_final

        return result.clamp(0, 1)
