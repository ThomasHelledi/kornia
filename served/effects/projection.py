"""Projection — animated geometric patterns projected onto architecture.

Extracts architectural edges from the video background, generates procedural
animated geometry that follows the structure, and composites with person mask.

Modes split into two families:

  PROJECTION (additive overlay on light backgrounds):
    - edge_waves: Light waves propagate along architectural edges
    - surface_geo: Animated geometric tiling on flat surfaces between edges
    - pulse_grid: Grid of pulsing dots/circles on surfaces, audio-reactive
    - edge_flow: Light streaming along edge tangent directions (streamlines)
    - reaction_diffusion: Gray-Scott organic patterns growing from architecture
    - sdf_morph: SDF procedural shapes morphing between geometric forms
    - combo: edge_flow on edges + sdf_morph on surfaces

  ENHANCEMENT (respect dark/detailed source material):
    - neon_enhance: Bloom extraction + light amplification + color grading
    - light_flow: Animated light streaming along existing bright paths
    - cyberpunk: Full cinematic enhancement (bloom + flow + atmosphere + scan)

Enhancement modes are designed for dark, detailed scenes (cyberpunk cities,
concert footage, night shots) where the goal is to AMPLIFY existing beauty
rather than overlay new patterns.
"""

import math
import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.color as KC

from .audio_reactive import AudioReactiveEngine, get_6bands, get_spectral


class ProjectionEffect:
    """Animated geometric projection onto architectural surfaces.

    Modes:
        edge_waves     — Traveling sine waves along edges
        surface_geo    — Hexagonal tiling breathing on surfaces
        pulse_grid     — Pulsing dot matrix, audio-reactive
        edge_flow      — Light streaming along edge tangent directions (hero mode)
        reaction_diffusion — Gray-Scott organic patterns growing from edges
        sdf_morph      — Morphing SDF shapes (circle/box/star/cross)
        combo          — edge_flow on edges + sdf_morph on surfaces
        neon_enhance   — Bloom + light amplification for dark scenes
        light_flow     — Animated light along existing bright paths
        cyberpunk      — Full cinematic enhancement suite

    Params:
        mode: str — one of the modes above (default: 'edge_waves')
        intensity: float — overall pattern strength (default: 0.5)
        speed: float — animation speed multiplier (default: 1.0)
        color: [R, G, B] — pattern tint 0-255 (default: [200, 220, 255] cool white)
        color2: [R, G, B] — optional secondary color for surfaces (dual-color mode)
        person_protect: float — person mask strength (default: 1.0)
        edge_threshold: float — Canny edge sensitivity (default: 0.08)
        wave_count: int — wave sources for edge_waves (default: 3)
        tile_scale: float — scale of geometric tiling (default: 1.0)
        bloom_threshold: float — brightness threshold for bloom extraction (default: 0.35)
        bloom_intensity: float — bloom glow strength (default: 0.6)
        color_push: float — color grading strength (default: 0.4)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()
        self._frame_count = 0
        self._prev_pattern = None
        self._person_model = None
        # Reaction-diffusion persistent state
        self._rd_A = None
        self._rd_B = None

    def _get_person_mask(self, tensor):
        """Get person mask using shared DeepLabV3 model."""
        from .segmentation import person_mask_refined
        return person_mask_refined(tensor, self.device)

    def _get_background_edges(self, tensor, person_mask, threshold=0.08):
        """Extract architectural edges from background only."""
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        # Canny edge detection
        _, edges = KF.canny(gray, low_threshold=threshold,
                            high_threshold=threshold * 3)
        # Mask out person region — only keep background edges
        bg_mask = (1.0 - person_mask).clamp(0, 1)
        edges = edges * bg_mask
        return edges

    def _get_surface_mask(self, edges, person_mask):
        """Get flat surface areas (background minus edges minus person).

        These are the wall/ceiling surfaces where we project patterns.
        """
        bg_mask = (1.0 - person_mask).clamp(0, 1)
        # Dilate edges slightly so surfaces don't touch edges
        dilated = KF.gaussian_blur2d(edges, (7, 7), (2.0, 2.0))
        dilated = (dilated > 0.1).float()
        # Surface = background AND NOT near-edge
        surface = bg_mask * (1.0 - dilated)
        return surface

    def _edge_waves(self, edges, surface, t, audio, params):
        """Animated waves propagating along architectural edges.

        Creates the illusion of light flowing through the structure.
        Waves emanate from beat hits and travel along edge paths.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        wave_count = int(params.get('wave_count', 3))

        bass = audio['bass']
        beat = audio['beat']
        centroid = audio['centroid']

        # Create coordinate grids
        yy = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(0, 1, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Multiple wave sources that travel along edges
        pattern = torch.zeros(1, 1, h, w, device=self.device)
        for i in range(wave_count):
            # Each wave has different direction and phase
            angle = (i / wave_count) * math.pi + t * speed * 0.3
            dx = math.cos(angle)
            dy = math.sin(angle)
            # Traveling wave: distance along wave direction
            dist = xx * dx + yy * dy
            # Sine wave with time-varying frequency
            freq = 12.0 + bass * 8.0 + i * 4.0
            phase = t * speed * (1.5 + i * 0.5) + beat * 2.0
            wave = torch.sin(dist * freq * math.pi + phase)
            # Normalize to 0-1
            wave = (wave + 1.0) * 0.5
            pattern = pattern + wave

        pattern = pattern / wave_count

        # Edge proximity glow: waves are brightest near edges
        # Distance transform approximation via gaussian blur of edges
        edge_proximity = KF.gaussian_blur2d(edges, (31, 31), (8.0, 8.0))
        edge_prox_max = edge_proximity.max()
        if edge_prox_max > 0.01:
            edge_proximity = (edge_proximity / edge_prox_max).clamp(0, 1)

        # Combine: pattern visible near edges + faint on surfaces
        near_edge_strength = edge_proximity * 0.8  # Strong near edges
        surface_strength = surface * 0.25  # Subtle on surfaces
        visibility = (near_edge_strength + surface_strength).clamp(0, 1)

        # Beat pulse: flash on beat
        visibility = visibility * (0.6 + beat * 0.4)

        return pattern * visibility

    def _surface_geo(self, edges, surface, t, audio, params):
        """Animated geometric tiling on flat surfaces.

        Generates hexagonal/triangular patterns that breathe and rotate.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        tile_scale = float(params.get('tile_scale', 1.0))

        bass = audio['bass']
        beat = audio['beat']

        # Coordinate grid in aspect-corrected space
        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Slowly rotating coordinate system
        rot_angle = t * speed * 0.15
        cos_r, sin_r = math.cos(rot_angle), math.sin(rot_angle)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # Hexagonal grid pattern
        scale = (6.0 + bass * 3.0) * tile_scale
        # Two offset triangular grids create hexagons
        gx = rx * scale
        gy = ry * scale * 1.1547  # 2/sqrt(3) for hex ratio

        # Hex cell coordinates
        row = torch.floor(gy)
        col = torch.floor(gx - (row % 2) * 0.5)

        # Distance from hex center
        cy = (gy - row - 0.5)
        cx = (gx - col - 0.5 - (row % 2) * 0.5)
        hex_dist = (cx.abs() + cy.abs() * 0.866).clamp(0, 1)

        # Pulsing hex borders
        border_width = 0.08 + beat * 0.06
        hex_border = ((hex_dist > (0.45 - border_width)) & (hex_dist < 0.45)).float()

        # Breathing effect: hex cells expand/contract with bass
        breath = 0.35 + bass * 0.15 + 0.05 * math.sin(t * speed * 2.0)
        hex_fill = (hex_dist < breath).float() * 0.3

        # Combine border + subtle fill
        pattern = hex_border * 0.9 + hex_fill

        # Traveling highlight wave across the grid
        wave_phase = t * speed * 0.8
        highlight = torch.sin(gx * 0.5 + wave_phase)
        highlight = ((highlight + 1.0) * 0.5).pow(3)  # Sharp peaks
        pattern = pattern + hex_border * highlight * 0.5

        # Only on surfaces (not on edges or person)
        return pattern.clamp(0, 1) * surface

    def _pulse_grid(self, edges, surface, t, audio, params):
        """Grid of pulsing circles on surfaces, audio-reactive.

        Creates a dot-matrix projection look.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        tile_scale = float(params.get('tile_scale', 1.0))

        bass = audio['bass']
        beat = audio['beat']

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Grid scale
        scale = (10.0 + bass * 5.0) * tile_scale

        # Grid cell position
        gx = xx * scale
        gy = yy * scale

        # Fractional position within cell (distance from center)
        fx = (gx - torch.floor(gx) - 0.5)
        fy = (gy - torch.floor(gy) - 0.5)
        dist_sq = fx * fx + fy * fy

        # Pulsing radius: each cell pulses with offset
        cell_id = (torch.floor(gx) * 7.0 + torch.floor(gy) * 13.0)
        cell_phase = cell_id * 0.1 + t * speed * 3.0
        radius = 0.08 + 0.12 * (torch.sin(cell_phase) * 0.5 + 0.5) + beat * 0.08
        radius_sq = radius * radius

        # Circle: bright inside, falloff at edge
        dots = (1.0 - (dist_sq / radius_sq).clamp(0, 1)).clamp(0, 1)
        dots = dots.pow(2)  # Sharper falloff

        # Ripple wave across the grid
        ripple_phase = t * speed * 1.5
        cx_grid = torch.floor(gx) / scale
        cy_grid = torch.floor(gy) / scale
        ripple = torch.sin((cx_grid + cy_grid) * 20.0 + ripple_phase)
        ripple = (ripple * 0.5 + 0.5).clamp(0, 1)

        # Modulate dot brightness by ripple
        pattern = dots * (0.5 + ripple * 0.5)

        # Edge glow: add subtle glow near edges
        edge_glow = KF.gaussian_blur2d(edges, (15, 15), (4.0, 4.0))
        e_max = edge_glow.max()
        if e_max > 0.01:
            edge_glow = (edge_glow / e_max).clamp(0, 1)
        pattern = pattern * surface + edge_glow * 0.4 * (1.0 - surface)

        return pattern.clamp(0, 1)

    def _edge_flow(self, edges, surface, t, audio, params):
        """Light streaming along architectural edge tangent directions.

        Computes Sobel gradients on edges to find edge direction, then creates
        animated streamlines that follow the architecture — like luminous veins.
        Enhanced with bloom glow and beat flash for cinematic projection feel.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        # Blur edges for smooth gradient field
        edge_blur = KF.gaussian_blur2d(edges, (9, 9), (3.0, 3.0))
        # Spatial gradient → (B, C, 2, H, W)
        grads = KF.spatial_gradient(edge_blur)
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]

        # Edge tangent = perpendicular to gradient = (-gy, gx)
        mag = (gx ** 2 + gy ** 2).sqrt().clamp(1e-6)
        tx = -gy / mag
        ty = gx / mag

        # Coordinate grids
        yy = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(0, 1, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Streamline pattern: dot product of position with flow direction
        flow_phase = (xx * tx + yy * ty) * 20.0
        flow_anim = t * speed * 3.0 + beat * 1.5

        # Primary streamline
        stream1 = torch.sin(flow_phase + flow_anim)
        stream1 = (stream1 + 1.0) * 0.5

        # Secondary harmonic for richer texture
        stream2 = torch.sin(flow_phase * 2.3 + flow_anim * 0.7 + 0.94)
        stream2 = (stream2 + 1.0) * 0.5

        # Third harmonic — fine detail layer
        stream3 = torch.sin(flow_phase * 4.7 + flow_anim * 1.3 + 2.1)
        stream3 = (stream3 + 1.0) * 0.5

        pattern = stream1 * 0.55 + stream2 * 0.3 + stream3 * 0.15

        # Wide edge proximity for extended glow reach
        edge_proximity = KF.gaussian_blur2d(edges, (31, 31), (9.0, 9.0))
        ep_max = edge_proximity.max()
        if ep_max > 0.01:
            edge_proximity = (edge_proximity / ep_max).clamp(0, 1)

        # Tight edge proximity for sharp core
        edge_core = KF.gaussian_blur2d(edges, (11, 11), (3.0, 3.0))
        ec_max = edge_core.max()
        if ec_max > 0.01:
            edge_core = (edge_core / ec_max).clamp(0, 1)

        # Dual-layer glow: bright core + soft halo
        glow = edge_core.pow(0.4) * 0.6 + edge_proximity.pow(0.6) * 0.4
        glow = glow * (0.5 + bass * 0.3 + beat * 0.2)
        pattern = pattern * glow

        # Bright leading edge that "flows" along the edges
        lead = torch.sin(flow_phase * 0.5 + t * speed * 5.0)
        lead = lead.clamp(0, 1).pow(4)
        pattern = pattern + lead * edge_core * 0.6

        # Bloom: soft gaussian blur added back for cinematic glow
        bloom = KF.gaussian_blur2d(pattern, (15, 15), (5.0, 5.0))
        pattern = pattern + bloom * 0.4

        # Beat flash: on strong beats, flash the architecture
        if beat > 0.3:
            flash = edge_proximity * beat * 0.5
            pattern = pattern + flash

        return pattern.clamp(0, 1)

    def _reaction_diffusion(self, edges, surface, t, audio, params):
        """Gray-Scott reaction-diffusion seeded from architectural edges.

        Organic coral-like patterns grow from edge structures and evolve
        over time. Persistent state across frames for continuous growth.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        # Gray-Scott parameters — worm/coral regime for fast visible growth
        f = 0.062 + bass * 0.008  # Feed rate (audio modulated)
        k = 0.061 + beat * 0.002  # Kill rate

        # Initialize or re-init on resolution change
        if self._rd_A is None or self._rd_A.shape[2:] != (h, w):
            self._rd_A = torch.ones(1, 1, h, w, device=self.device)
            self._rd_B = torch.zeros(1, 1, h, w, device=self.device)
            # Strong initial seed from edges — wider blur for visible patterns
            seed = KF.gaussian_blur2d(edges, (11, 11), (3.0, 3.0))
            s_max = seed.max()
            if s_max > 0.01:
                seed = seed / s_max
            # Add noise near edges for organic variation
            noise = torch.rand_like(seed) * 0.3
            self._rd_B = (seed * 0.8 + seed * noise).clamp(0, 1)

        # Laplacian convolution kernel (sums to 0)
        laplacian = torch.tensor(
            [[0.05, 0.2, 0.05],
             [0.2, -1.0, 0.2],
             [0.05, 0.2, 0.05]],
            device=self.device).view(1, 1, 3, 3)

        # Diffusion rates — higher for faster pattern development
        Da, Db = 1.0, 0.5
        iters = int(12 + bass * 8)  # More iterations for faster evolution

        A = self._rd_A
        B = self._rd_B

        for _ in range(iters):
            LA = F.conv2d(A, laplacian, padding=1)
            LB = F.conv2d(B, laplacian, padding=1)
            ABB = A * B * B
            A = (A + Da * LA - ABB + f * (1.0 - A)).clamp(0, 1)
            B = (B + Db * LB + ABB - (k + f) * B).clamp(0, 1)

        # Continuous edge re-seeding (light) + beat pulse seeding
        seed = KF.gaussian_blur2d(edges, (5, 5), (1.5, 1.5))
        s_max = seed.max()
        if s_max > 0.01:
            seed = (seed / s_max).clamp(0, 1)
        B = (B + seed * (0.02 + beat * 0.08)).clamp(0, 1)

        self._rd_A = A.detach()
        self._rd_B = B.detach()

        # B chemical is the visible organic pattern
        pattern = B

        # Apply to background (surfaces + near edges)
        edge_glow = KF.gaussian_blur2d(edges, (15, 15), (4.0, 4.0)).clamp(0, 1)
        bg_visibility = (surface + edge_glow * 0.5).clamp(0, 1)
        pattern = pattern * bg_visibility

        return pattern.clamp(0, 1)

    def _sdf_morph(self, edges, surface, t, audio, params):
        """SDF procedural shapes that morph between geometric forms.

        Tiled signed distance field rendering with smooth transitions
        between circle, rounded box, and star shapes.
        """
        _, _, h, w = edges.shape
        speed = float(params.get('speed', 1.0))
        tile_scale = float(params.get('tile_scale', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Tile the space
        scale = (4.0 + bass * 2.0) * tile_scale
        tx = (xx * scale) % 2.0 - 1.0
        ty = (yy * scale) % 2.0 - 1.0

        # Slowly rotating coordinates
        rot = t * speed * 0.5
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = tx * cos_r - ty * sin_r
        ry = tx * sin_r + ty * cos_r

        # SDF: Circle
        d_circle = (rx ** 2 + ry ** 2).sqrt() - 0.4

        # SDF: Rounded box
        bx, by = rx.abs() - 0.35, ry.abs() - 0.35
        d_box = (bx.clamp(min=0) ** 2 + by.clamp(min=0) ** 2).sqrt() \
                + torch.max(bx, by).clamp(max=0) - 0.05

        # SDF: 6-point star
        angle = torch.atan2(ry, rx)
        radius = (rx ** 2 + ry ** 2).sqrt()
        star_r = 0.3 + 0.12 * torch.cos(angle * 6.0 + t * speed * 1.5)
        d_star = radius - star_r

        # SDF: Cross (union of horizontal + vertical bars)
        d_hbar_x, d_hbar_y = rx.abs() - 0.4, ry.abs() - 0.1
        d_hbar = (d_hbar_x.clamp(min=0) ** 2 + d_hbar_y.clamp(min=0) ** 2).sqrt() \
                 + torch.max(d_hbar_x, d_hbar_y).clamp(max=0)
        d_vbar_x, d_vbar_y = rx.abs() - 0.1, ry.abs() - 0.4
        d_vbar = (d_vbar_x.clamp(min=0) ** 2 + d_vbar_y.clamp(min=0) ** 2).sqrt() \
                 + torch.max(d_vbar_x, d_vbar_y).clamp(max=0)
        d_cross = torch.min(d_hbar, d_vbar)

        # Morph between shapes based on time + beat
        phase = t * speed * 0.3
        m1 = (math.sin(phase) * 0.5 + 0.5)
        m2 = (math.sin(phase * 0.7 + 1.0) * 0.5 + 0.5)
        d = d_circle * m1 + d_box * (1.0 - m1)
        d = d * m2 + d_star * (1.0 - m2)

        # Beat: snap in cross shape
        if beat > 0.6:
            d = d * (1.0 - beat * 0.5) + d_cross * beat * 0.5

        # Render SDF: edge glow + subtle fill
        edge_w = 0.03 + beat * 0.02
        sdf_edge = (1.0 - (d.abs() / edge_w).clamp(0, 1))
        sdf_fill = (1.0 - (d / 0.15).clamp(0, 1)).clamp(0, 1) * 0.2

        pattern = sdf_edge * 0.8 + sdf_fill

        return pattern.clamp(0, 1) * surface

    # ── ENHANCEMENT MODES ──────────────────────────────────────────────
    # These modes ENHANCE existing image content rather than overlaying.
    # Designed for dark, detailed scenes (cities, concerts, night shots).

    def _extract_bloom(self, tensor, threshold=0.35):
        """Extract bright regions for HDR-style bloom.

        Multi-scale bloom simulates real camera lens behavior.
        Separates warm lights (windows) from cool lights (neon) for
        independent color treatment.

        Returns:
            bright: thresholded bright pixels (soft mask)
            bloom_s/m/l: 3-scale bloom (tight/medium/atmospheric)
            color_bloom: bloom preserving original light color
            warm_bloom: bloom of warm-colored lights only
        """
        lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        # Soft threshold
        bright = ((lum - threshold) / (1.0 - threshold)).clamp(0, 1)

        # Multi-scale bloom — much larger kernels for cinematic spread
        bloom_s = KF.gaussian_blur2d(bright, (15, 15), (5.0, 5.0))
        bloom_m = KF.gaussian_blur2d(bright, (51, 51), (16.0, 16.0))
        bloom_l = KF.gaussian_blur2d(bright, (101, 101), (35.0, 35.0))

        # Color-preserving bloom: bright pixels keep their color as they spread
        bright_rgb = tensor * bright
        color_bloom = KF.gaussian_blur2d(bright_rgb, (41, 41), (14.0, 14.0))

        # Warm light detection: R channel dominance
        r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
        warm_mask = ((r - b) * 4.0 + (r - g) * 2.0).clamp(0, 1) * bright
        warm_bloom = KF.gaussian_blur2d(warm_mask, (21, 21), (7.0, 7.0))

        # Local bright spots: pixels brighter than their neighborhood
        # Catches small windows, signs, point lights regardless of color
        lum_blur = KF.gaussian_blur2d(lum, (21, 21), (7.0, 7.0))
        local_bright = ((lum - lum_blur) * 5.0).clamp(0, 1)
        # Bloom these local maxima for window glow effect
        spot_bloom = KF.gaussian_blur2d(local_bright, (15, 15), (5.0, 5.0))

        return bright, bloom_s, bloom_m, bloom_l, color_bloom, warm_bloom, spot_bloom

    def _color_grade_cyberpunk(self, tensor, push=0.4):
        """Cyberpunk color grading: S-curve contrast + blue shadows + teal highlights.

        Uses a true S-curve: shadows darker, midtones preserved, highlights boosted.
        Not a simple gamma — maintains building detail while deepening mood.
        """
        lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]

        # S-curve contrast: steepness controlled by push
        # Formula: 0.5 + 0.5 * sign(x-0.5) * |2*(x-0.5)|^gamma
        # This darkens darks AND brightens brights while preserving midtones
        centered = lum - 0.5
        steepness = 1.0 + push * 1.5  # push=0.5 → steepness 1.75
        s_curve = 0.5 + centered.sign() * (2.0 * centered.abs()).pow(steepness) * 0.5
        s_curve = s_curve.clamp(0, 1)

        # Apply S-curve to all channels proportionally
        ratio = (s_curve / lum.clamp(1e-6)).clamp(0.2, 3.0)
        tensor = tensor * ratio

        # Shadow tint: deep navy-blue in dark regions
        shadow_mask = (1.0 - lum * 3.0).clamp(0, 1)  # Only truly dark areas
        tensor[:, 0:1] = tensor[:, 0:1] + shadow_mask * 0.01 * push
        tensor[:, 1:2] = tensor[:, 1:2] + shadow_mask * 0.03 * push
        tensor[:, 2:3] = tensor[:, 2:3] + shadow_mask * 0.08 * push

        # Highlight tint: push bright areas toward teal/cyan
        highlight_mask = ((lum - 0.3) / 0.7).clamp(0, 1)
        tensor[:, 1:2] = tensor[:, 1:2] + highlight_mask * 0.04 * push
        tensor[:, 2:3] = tensor[:, 2:3] + highlight_mask * 0.07 * push

        return tensor.clamp(0, 1)

    def _neon_enhance(self, tensor, t, audio, params):
        """HDR bloom + light amplification + cinematic color grading.

        Designed for dark, detailed scenes. Finds existing bright pixels
        (windows, neon, streetlights), creates massive multi-scale bloom,
        separates warm/cool lights, and applies aggressive contrast.
        """
        bass = audio['bass']
        beat = audio['beat']
        bloom_thresh = float(params.get('bloom_threshold', 0.25))
        bloom_int = float(params.get('bloom_intensity', 0.8))
        color_push = float(params.get('color_push', 0.5))

        # Step 1: Color grading (crush shadows, tint, contrast)
        result = self._color_grade_cyberpunk(tensor.clone(), color_push)

        # Step 2: Multi-scale bloom extraction
        bright, bloom_s, bloom_m, bloom_l, color_bloom, warm_bloom, spot_bloom = \
            self._extract_bloom(tensor, bloom_thresh)

        # Audio-reactive bloom strength
        bloom_str = bloom_int * (0.7 + bass * 0.2 + beat * 0.1)

        # Tight bloom: sharp glow directly around light sources
        result = result + bloom_s * bloom_str * 0.5
        # Medium bloom: atmospheric halo around light clusters
        result = result + bloom_m * bloom_str * 0.45
        # Wide bloom: overall scene atmosphere
        result = result + bloom_l * bloom_str * 0.3

        # Step 3: Color-preserving bloom — lights glow their actual color
        result = result + color_bloom * bloom_str * 0.7

        # Step 4: Warm light glow (windows, lamps) — orange/amber warmth
        result[:, 0:1] = result[:, 0:1] + warm_bloom * bloom_str * 0.35
        result[:, 1:2] = result[:, 1:2] + warm_bloom * bloom_str * 0.15

        # Step 4b: Local bright spot glow (windows, signs, point lights)
        # These are small lights that might not hit the global bloom threshold
        spot_color = tensor * spot_bloom  # Glow in original color
        result = result + spot_color * bloom_str * 0.5
        # Extra warm glow for spots — most window lights are warm
        result[:, 0:1] = result[:, 0:1] + spot_bloom * bloom_str * 0.15
        result[:, 1:2] = result[:, 1:2] + spot_bloom * bloom_str * 0.08

        # Step 5: Anamorphic horizontal bloom (cinematic lens streak)
        very_bright = ((bright - 0.3) / 0.7).clamp(0, 1)
        if very_bright.max() > 0.01:
            h_bloom = KF.gaussian_blur2d(very_bright, (1, 101), (0.1, 35.0))
            result[:, 0:1] = result[:, 0:1] + h_bloom * bloom_str * 0.08
            result[:, 1:2] = result[:, 1:2] + h_bloom * bloom_str * 0.15
            result[:, 2:3] = result[:, 2:3] + h_bloom * bloom_str * 0.22

        # Step 6: Selective bright edge accent (very subtle)
        lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        bright_mask = (lum > 0.2).float()
        _, edges = KF.canny(lum, low_threshold=0.1, high_threshold=0.3)
        bright_edges = edges * bright_mask
        edge_glow = KF.gaussian_blur2d(bright_edges, (7, 7), (2.0, 2.0))
        eg_max = edge_glow.max()
        if eg_max > 0.01:
            edge_glow = (edge_glow / eg_max).clamp(0, 1)
        # Very thin teal accent
        result[:, 1:2] = result[:, 1:2] + edge_glow * 0.06
        result[:, 2:3] = result[:, 2:3] + edge_glow * 0.08

        # Step 7: Beat pulse
        if beat > 0.3:
            result = result * (1.0 + beat * 0.08)

        # Step 8: Vignette
        _, _, h, w = tensor.shape
        vy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1)
        vx = torch.linspace(-1, 1, w, device=self.device).view(1, 1, 1, w)
        vignette = 1.0 - (vx ** 2 + vy ** 2).sqrt() * 0.3
        vignette = vignette.clamp(0.5, 1.0)
        result = result * vignette

        return result.clamp(0, 1)

    def _light_flow(self, tensor, t, audio, params):
        """Animated light streaming along existing bright paths.

        Builds on neon_enhance, then adds flow animation masked to
        existing bright linear structures (highways, light trails, neon).
        """
        _, _, h, w = tensor.shape
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        bloom_thresh = float(params.get('bloom_threshold', 0.25))

        # Start with neon_enhance as base
        result = self._neon_enhance(tensor, t, audio, params)

        # Extract bright path mask from ORIGINAL (not graded) tensor
        lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        bright = ((lum - bloom_thresh) / (1.0 - bloom_thresh)).clamp(0, 1)

        # Flow direction along bright structures
        bright_blur = KF.gaussian_blur2d(bright, (9, 9), (3.0, 3.0))
        grads = KF.spatial_gradient(bright_blur)
        gx, gy = grads[:, :, 0], grads[:, :, 1]
        mag = (gx ** 2 + gy ** 2).sqrt().clamp(1e-6)
        tx, ty = -gy / mag, gx / mag

        # Coordinate grids
        yy = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(0, 1, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # Flow animation — light particles streaming along paths
        flow_phase = (xx * tx + yy * ty) * 30.0
        flow_anim = t * speed * 4.0

        stream = torch.sin(flow_phase + flow_anim).clamp(0, 1).pow(3)
        stream2 = torch.sin(flow_phase * 0.4 + flow_anim * 0.7 + 1.5).clamp(0, 1).pow(2)
        flow_pattern = stream * 0.6 + stream2 * 0.4

        # Mask: only on existing bright structures
        flow_visible = flow_pattern * bright.pow(0.5)

        # Add flow in the COLOR of the original light source
        flow_color = tensor * flow_visible * 0.35 * (0.7 + bass * 0.3)
        result = result + flow_color

        # Beat flash on existing lights
        if beat > 0.4:
            result = result + bright * beat * 0.2

        return result.clamp(0, 1)

    def _cyberpunk(self, tensor, t, audio, params):
        """Full cinematic cyberpunk enhancement suite.

        Combines all enhancement techniques:
        1. Neon bloom + color grading (neon_enhance base)
        2. Light flow animation along bright paths
        3. Holographic scan lines on bright structures
        4. Film grain for cinematic texture
        5. Atmospheric depth haze
        """
        _, _, h, w = tensor.shape
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        # Base: light_flow (which includes neon_enhance)
        result = self._light_flow(tensor, t, audio, params)

        # Holographic scan lines on bright structures
        lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        bright = (lum > 0.3).float()

        yy = torch.linspace(0, h, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        # Scan lines: horizontal lines that slowly drift
        scan_freq = 2.0 + bass * 1.0  # pixels between scan lines
        scan_phase = t * speed * 0.5
        scan = torch.sin(yy * scan_freq + scan_phase)
        scan = scan.clamp(0, 1).pow(8)  # Very thin bright lines
        # Only on bright structures, very subtle
        scan_visible = scan * bright * 0.08
        # Chromatic aberration on scan lines: slight R/B shift
        result[:, 0:1] = result[:, 0:1] + scan_visible * 0.5  # Red channel
        result[:, 2:3] = result[:, 2:3] + scan_visible  # Blue channel

        # Atmospheric depth: top of frame slightly hazier (sky glow)
        depth_y = torch.linspace(0, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        # Invert: top=1, bottom=0 — sky has more atmosphere
        sky_mask = (1.0 - depth_y).pow(2) * 0.06
        # Blue-teal atmospheric tint
        result[:, 0:1] = result[:, 0:1] + sky_mask * 0.3
        result[:, 1:2] = result[:, 1:2] + sky_mask * 0.5
        result[:, 2:3] = result[:, 2:3] + sky_mask * 0.8

        # Film grain: very subtle noise for cinematic texture
        grain = torch.randn(1, 1, h, w, device=self.device) * 0.015
        result = result + grain

        return result.clamp(0, 1)

    # ── PROJECTION MODES (original) ─────────────────────────────────

    def _combo(self, edges, surface, t, audio, params):
        """Combined mode: edge_flow on edges + sdf_morph on surfaces.

        The best of both worlds — luminous streaming along architecture
        with geometric shapes projected onto flat walls.
        """
        # Edge flow for the arches and structural lines
        flow = self._edge_flow(edges, surface, t, audio, params)
        # SDF shapes on flat surfaces
        sdf = self._sdf_morph(edges, surface, t, audio, params)
        # Combine: flow dominates near edges, SDF on surfaces
        edge_proximity = KF.gaussian_blur2d(edges, (21, 21), (6.0, 6.0))
        ep_max = edge_proximity.max()
        if ep_max > 0.01:
            edge_proximity = (edge_proximity / ep_max).clamp(0, 1)
        # Crossfade: near edges → flow, on surfaces → SDF
        pattern = flow * edge_proximity + sdf * (1.0 - edge_proximity) * surface
        # Add subtle flow bleed onto surfaces for cohesion
        pattern = pattern + flow * surface * 0.15
        return pattern.clamp(0, 1)

    def process(self, tensor, params):
        self._frame_count += 1
        mode = params.get('mode', 'edge_waves')
        intensity = float(params.get('intensity', 0.5))
        color = params.get('color', [200, 220, 255])
        color2 = params.get('color2', None)  # Optional secondary color
        person_protect = float(params.get('person_protect', 1.0))
        edge_threshold = float(params.get('edge_threshold', 0.08))
        t = float(params.get('time', self._frame_count / 30.0))

        # Audio features
        bands = get_6bands(params)
        spectral = get_spectral(params)
        beat_pulse = float(params.get('beat_pulse', 0))
        bass = self._audio.smooth('bass', bands['bass'] + bands['sub_bass'])
        beat = self._audio.smooth('beat', beat_pulse, attack=0.5, release=0.08)
        audio = {
            'bass': bass,
            'beat': beat,
            'centroid': spectral['centroid'],
        }

        # ── Enhancement modes: skip person mask/edge detection ──
        # These work on the full image, enhancing existing content
        if mode in ('neon_enhance', 'light_flow', 'cyberpunk'):
            if mode == 'neon_enhance':
                return self._neon_enhance(tensor, t, audio, params)
            elif mode == 'light_flow':
                return self._light_flow(tensor, t, audio, params)
            else:  # cyberpunk
                return self._cyberpunk(tensor, t, audio, params)

        # ── Projection modes: use person mask + edge extraction ──
        # Person mask (cached in segmentation module across frames)
        pmask = self._get_person_mask(tensor)

        # Background edges and surfaces
        edges = self._get_background_edges(tensor, pmask, edge_threshold)
        surface = self._get_surface_mask(edges, pmask)

        # Generate animated pattern based on mode
        if mode == 'surface_geo':
            pattern = self._surface_geo(edges, surface, t, audio, params)
        elif mode == 'pulse_grid':
            pattern = self._pulse_grid(edges, surface, t, audio, params)
        elif mode == 'edge_flow':
            pattern = self._edge_flow(edges, surface, t, audio, params)
        elif mode == 'reaction_diffusion':
            pattern = self._reaction_diffusion(edges, surface, t, audio, params)
        elif mode == 'sdf_morph':
            pattern = self._sdf_morph(edges, surface, t, audio, params)
        elif mode == 'combo':
            pattern = self._combo(edges, surface, t, audio, params)
        else:  # edge_waves
            pattern = self._edge_waves(edges, surface, t, audio, params)

        # Temporal smoothing: blend with previous pattern for stability
        if self._prev_pattern is not None and self._prev_pattern.shape == pattern.shape:
            momentum = 0.3
            pattern = pattern * (1.0 - momentum) + self._prev_pattern * momentum
        self._prev_pattern = pattern.detach().clone()

        # Colorize pattern — supports dual color (color for edges, color2 for surfaces)
        cr, cg, cb = [c / 255.0 for c in color]
        color_pattern = torch.zeros_like(tensor)
        if color2 is not None:
            # Dual-color: color on edges, color2 on surfaces
            cr2, cg2, cb2 = [c / 255.0 for c in color2]
            edge_proximity = KF.gaussian_blur2d(edges, (21, 21), (6.0, 6.0))
            ep_max = edge_proximity.max()
            if ep_max > 0.01:
                edge_proximity = (edge_proximity / ep_max).clamp(0, 1)
            # Blend between two colors based on edge proximity
            r = pattern * (cr * edge_proximity + cr2 * (1.0 - edge_proximity))
            g = pattern * (cg * edge_proximity + cg2 * (1.0 - edge_proximity))
            b = pattern * (cb * edge_proximity + cb2 * (1.0 - edge_proximity))
            color_pattern[:, 0:1] = r
            color_pattern[:, 1:2] = g
            color_pattern[:, 2:3] = b
        else:
            color_pattern[:, 0:1] = pattern * cr
            color_pattern[:, 1:2] = pattern * cg
            color_pattern[:, 2:3] = pattern * cb

        # Composite: additive blend onto background, person stays clean
        bg_mask = (1.0 - pmask * person_protect).clamp(0, 1)
        result = tensor + color_pattern * intensity * bg_mask

        return result.clamp(0, 1)
