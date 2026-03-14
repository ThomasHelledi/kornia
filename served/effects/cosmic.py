"""Cosmic — procedural environments for 8D sound visualization.

Generates full-frame procedural visuals inspired by cosmic imagery:
nebulae, particle clouds, DNA helixes. These REPLACE the input frame
entirely with procedural content driven by audio analysis.

Modes:
  - nebula: Gas cloud eye with warm/cool split, central light, star particles
  - cosmic_dust: Multi-depth particle layers with parallax drift
  - dna_helix: Double helix strand with glowing nucleotide spheres

Reference: "Life, the Universe & Everything" — Garret John (UPRISER)
"""

import math
import torch
import torch.nn.functional as F
import kornia.filters as KF

from .audio_reactive import AudioReactiveEngine, get_6bands, get_spectral


def _value_noise(h, w, scale, device, seed_offset=0.0):
    """Smooth value noise via upsampled random grid."""
    sh = max(2, h // scale)
    sw = max(2, w // scale)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed_offset * 1000) % (2 ** 31))
    grid = torch.rand(1, 1, sh, sw, generator=gen).to(device)
    return F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)


def _fractal_noise(h, w, octaves, device, seed=0.0, base_scale=4):
    """Multi-octave fractal noise (value noise sum).

    base_scale: starting grid divisions (lower = larger features).
    """
    noise = torch.zeros(1, 1, h, w, device=device)
    amp = 1.0
    total = 0.0
    for i in range(octaves):
        scale = base_scale * (2 ** i)
        noise = noise + _value_noise(h, w, scale, device, seed + i * 17.3) * amp
        total += amp
        amp *= 0.5
    return noise / total


def _warped_noise(h, w, octaves, device, seed=0.0, base_scale=4, warp_strength=0.3):
    """Domain-warped fractal noise — organic, flowing results.

    Feeds SIGNED noise into coordinates to create swirling patterns:
      final(x,y) = fbm(x + warp_strength * (fbm_a - 0.5), y + warp_strength * (fbm_b - 0.5))
    """
    # Warp field: SIGNED noise [-0.5, 0.5] for bidirectional displacement
    warp_x = _fractal_noise(h, w, max(3, octaves - 1), device, seed + 50.0, base_scale) - 0.5
    warp_y = _fractal_noise(h, w, max(3, octaves - 1), device, seed + 83.0, base_scale) - 0.5

    # Build coordinate grid in [-1, 1] for grid_sample
    grid_y = torch.linspace(-1, 1, h, device=device).view(1, h, 1, 1).expand(1, h, w, 1)
    grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, w, 1).expand(1, h, w, 1)
    base_grid = torch.cat([grid_x, grid_y], dim=-1)  # (1, H, W, 2)

    # Warp coordinates: signed displacement scaled by warp_strength
    warp = torch.stack([
        warp_x.squeeze(0).squeeze(0) * warp_strength * 2.0,
        warp_y.squeeze(0).squeeze(0) * warp_strength * 2.0,
    ], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    warped_grid = (base_grid + warp).clamp(-1, 1)

    # Sample fractal noise through warped coordinates
    raw = _fractal_noise(h, w, octaves, device, seed, base_scale)
    warped = F.grid_sample(raw, warped_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return warped


def _turbulence(h, w, octaves, device, seed=0.0, base_scale=4):
    """Turbulence noise — absolute value of signed noise for wispy edges."""
    noise = torch.zeros(1, 1, h, w, device=device)
    amp = 1.0
    total = 0.0
    for i in range(octaves):
        scale = base_scale * (2 ** i)
        layer = _value_noise(h, w, scale, device, seed + i * 17.3)
        noise = noise + (layer * 2.0 - 1.0).abs() * amp  # abs(signed) = sharp edges
        total += amp
        amp *= 0.5
    return noise / total


def _curl_noise(h, w, device, seed=0.0, base_scale=5, octaves=4):
    """Curl noise — divergence-free flow field from potential field.

    Computes curl of a scalar potential field via finite differences.
    Produces filamentary, wispy displacement vectors (dx, dy).
    Returns (curl_x, curl_y) each (1, 1, H, W), normalized to [-1, 1].
    """
    # Scalar potential field
    potential = _fractal_noise(h, w, octaves, device, seed, base_scale)

    # Finite differences via torch.roll (1-pixel shift)
    dN_dy = torch.roll(potential, -1, 2) - torch.roll(potential, 1, 2)  # dP/dy
    dN_dx = torch.roll(potential, -1, 3) - torch.roll(potential, 1, 3)  # dP/dx

    # Curl: (dP/dy, -dP/dx) — perpendicular to gradient = divergence-free
    curl_x = dN_dy
    curl_y = -dN_dx

    # Normalize to [-1, 1]
    max_val = max(curl_x.abs().max().item(), curl_y.abs().max().item(), 1e-6)
    curl_x = curl_x / max_val
    curl_y = curl_y / max_val

    return curl_x, curl_y


def _curl_warp(h, w, device, density, seed=0.0, strength=0.15, base_scale=5):
    """Apply curl noise displacement to a density field.

    Warps the density field along divergence-free flow lines,
    creating filamentary wispy structures from blobby FBM input.
    """
    curl_x, curl_y = _curl_noise(h, w, device, seed, base_scale)

    # Build sampling grid
    grid_y = torch.linspace(-1, 1, h, device=device).view(1, h, 1, 1).expand(1, h, w, 1)
    grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, w, 1).expand(1, h, w, 1)
    base_grid = torch.cat([grid_x, grid_y], dim=-1)  # (1, H, W, 2)

    # Displacement from curl field
    disp = torch.stack([
        curl_x.squeeze(0).squeeze(0) * strength,
        curl_y.squeeze(0).squeeze(0) * strength,
    ], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    warped_grid = (base_grid + disp).clamp(-1, 1)
    return F.grid_sample(density, warped_grid, mode='bilinear',
                         padding_mode='reflection', align_corners=True)


def _worley_noise(h, w, device, seed=0.0, num_cells=8):
    """Worley/Voronoi noise — F2-F1 ridge pattern for shock-front filaments.

    Grid-based: places one random point per cell, computes distance to nearest
    two points. F2-F1 = ridges along cell boundaries (Voronoi edges).
    Returns (1, 1, H, W) in [0, 1].
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed * 1000) % (2 ** 31))

    # Random point per cell
    cell_h = num_cells
    cell_w = int(num_cells * (w / h))
    if cell_w < 2:
        cell_w = 2
    num_points = cell_h * cell_w

    # Cell centers + jitter
    points_y = torch.zeros(num_points)
    points_x = torch.zeros(num_points)
    idx = 0
    for cy in range(cell_h):
        for cx in range(cell_w):
            jy = torch.rand(1, generator=gen).item()
            jx = torch.rand(1, generator=gen).item()
            points_y[idx] = (cy + jy) / cell_h
            points_x[idx] = (cx + jx) / cell_w
            idx += 1

    points_y = points_y.to(device)
    points_x = points_x.to(device)

    # Pixel coordinates normalized [0, 1]
    py = torch.linspace(0, 1, h, device=device).view(h, 1)
    px = torch.linspace(0, 1, w, device=device).view(1, w)

    # Distance to each point: (H, W, N)
    dy = py.unsqueeze(2) - points_y.view(1, 1, -1)  # (H, 1, N) - (1, 1, N)
    dx = px.unsqueeze(2) - points_x.view(1, 1, -1)  # (1, W, N) - (1, 1, N)
    # Expand for broadcasting
    dy = dy.expand(h, w, num_points)
    dx = dx.expand(h, w, num_points)
    dists = (dy ** 2 + dx ** 2).sqrt()  # (H, W, N)

    # F1 = nearest, F2 = second nearest
    sorted_dists, _ = dists.sort(dim=2)
    f1 = sorted_dists[:, :, 0]
    f2 = sorted_dists[:, :, 1]

    # F2 - F1 = ridge pattern (bright at Voronoi edges)
    ridges = (f2 - f1)
    # Normalize to [0, 1]
    ridges = ridges / (ridges.max() + 1e-6)

    return ridges.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def _double_warped_noise(h, w, octaves, device, seed=0.0, base_scale=4, warp_strength=0.3):
    """Double domain warping — nested FBM warp for ultra-organic folds.

    Inspired by Inigo Quilez: f(p + 4*r) where r = fbm(p + 4*q) and q = fbm(p).
    Two levels of coordinate distortion create deeply folded, flowing patterns
    that look like real gas dynamics rather than simple noise.
    """
    # First warp field (q): signed displacement
    q_x = _fractal_noise(h, w, max(2, octaves - 1), device, seed + 50.0, base_scale) - 0.5
    q_y = _fractal_noise(h, w, max(2, octaves - 1), device, seed + 83.0, base_scale) - 0.5

    # Build coordinate grid
    grid_y = torch.linspace(-1, 1, h, device=device).view(1, h, 1, 1).expand(1, h, w, 1)
    grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, w, 1).expand(1, h, w, 1)
    base_grid = torch.cat([grid_x, grid_y], dim=-1)

    # First displacement: warp by q
    d1 = torch.stack([
        q_x.squeeze(0).squeeze(0) * warp_strength * 2.0,
        q_y.squeeze(0).squeeze(0) * warp_strength * 2.0,
    ], dim=-1).unsqueeze(0)
    mid_grid = (base_grid + d1).clamp(-1, 1)

    # Second warp field (r): sample through warped coordinates
    r_raw_x = _fractal_noise(h, w, max(2, octaves - 1), device, seed + 17.0, base_scale)
    r_raw_y = _fractal_noise(h, w, max(2, octaves - 1), device, seed + 42.0, base_scale)
    r_x = F.grid_sample(r_raw_x, mid_grid, mode='bilinear',
                         padding_mode='reflection', align_corners=True) - 0.5
    r_y = F.grid_sample(r_raw_y, mid_grid, mode='bilinear',
                         padding_mode='reflection', align_corners=True) - 0.5

    # Second displacement: warp again by r (slightly weaker)
    d2 = torch.stack([
        r_x.squeeze(0).squeeze(0) * warp_strength * 1.4,
        r_y.squeeze(0).squeeze(0) * warp_strength * 1.4,
    ], dim=-1).unsqueeze(0)
    final_grid = (mid_grid + d2).clamp(-1, 1)

    # Sample final noise through doubly-warped coordinates
    raw = _fractal_noise(h, w, octaves, device, seed, base_scale)
    return F.grid_sample(raw, final_grid, mode='bilinear',
                         padding_mode='reflection', align_corners=True)


def _advect_frame(prev_frame, h, w, device, t, strength=0.02, rot_strength=0.012):
    """Advect a frame along curl noise + rotational flow field.

    Two components:
    1. Curl noise: organic, divergence-free turbulence (filamentary streams)
    2. Rotational flow: tangential velocity around center (visible gas rotation)
    Combined creates the illusion of gas swirling around the nebula center
    with organic turbulent variation.
    """
    curl_x, curl_y = _curl_noise(h, w, device, seed=t * 0.1 + 777.0,
                                  base_scale=6, octaves=3)

    # All coordinates in [-1, 1] normalized space (grid_sample space)
    gy = torch.linspace(-1, 1, h, device=device).view(h, 1).expand(h, w)
    gx = torch.linspace(-1, 1, w, device=device).view(1, w).expand(h, w)

    # Base sampling grid
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Rotational flow: tangential velocity (-y, x) / r
    radius = (gx ** 2 + gy ** 2).sqrt() + 1e-4
    # Rotation strongest near ring (r~0.5), decays outward and inward
    rot_profile = (1.0 - ((radius - 0.5) / 0.6).clamp(0, 1)).pow(0.5)
    rot_vx = (-gy / radius) * rot_strength * rot_profile
    rot_vy = (gx / radius) * rot_strength * rot_profile

    # Combine curl noise + rotation
    disp = torch.stack([
        curl_x.squeeze(0).squeeze(0) * strength + rot_vx,
        curl_y.squeeze(0).squeeze(0) * strength + rot_vy,
    ], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    advected_grid = (base_grid + disp).clamp(-1, 1)
    return F.grid_sample(prev_frame, advected_grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


def _emission_absorption(density, thin_color, thick_color, absorption_strength=2.0):
    """Emission + Absorption model for physically-inspired nebula coloring.

    Thin gas emits blue (OIII), thick gas emits red (H-alpha),
    very dense gas absorbs (darkens).

    Args:
        density: (1, 1, H, W) gas density [0, 1]
        thin_color: (R, G, B) tuple for thin/low-density emission
        thick_color: (R, G, B) tuple for thick/high-density emission
        absorption_strength: how quickly dense gas absorbs light

    Returns:
        (1, 3, H, W) colored emission with absorption
    """
    # Emission color blends from thin to thick based on density
    d = density.clamp(0, 1)
    blend = (1.0 - torch.exp(-d * 3.0))  # Smoothstep-like: 0→thin_color, 1→thick_color

    h, w = density.shape[2], density.shape[3]
    result = torch.zeros(1, 3, h, w, device=density.device)

    for ch in range(3):
        thin_c = thin_color[ch]
        thick_c = thick_color[ch]
        result[:, ch:ch+1] = thin_c * (1.0 - blend) + thick_c * blend

    # Emission brightness = density (more gas = more light, up to a point)
    emission = d.pow(0.7) * 1.5  # Sub-linear: diminishing returns at high density

    # Absorption: very dense regions self-absorb (darken)
    transmittance = torch.exp(-d * absorption_strength)

    # Final = emission * absorption
    result = result * emission * (0.3 + 0.7 * transmittance)

    return result


class CosmicEffect:
    """Procedural cosmic environments for 8D sound visualization.

    Params:
        mode: str — 'nebula' | 'cosmic_dust' | 'dna_helix' (default: 'nebula')
        speed: float — animation speed (default: 1.0)
        texture: str — path to nebula photograph for photographic gas quality
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()
        self._frame_count = 0
        self._star_field = None
        self._star_phases = None
        self._cached_h = 0
        self._cached_w = 0
        # Temporal feedback: previous frame for fluid-like persistence
        self._prev_frame = None
        self._prev_h = 0
        self._prev_w = 0
        # Texture-based mode: use real nebula photograph
        self._texture = None
        self._texture_path = None

    def _load_texture(self, path):
        """Load a nebula photograph as (1, 3, H, W) tensor in [0, 1]."""
        if self._texture is not None and self._texture_path == path:
            return self._texture
        import PIL.Image
        import numpy as np
        img = PIL.Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self._texture = t
        self._texture_path = path
        return t

    def _sample_texture(self, texture, grid):
        """Sample texture using grid_sample. grid: (1, H, W, 2) in [-1, 1]."""
        return F.grid_sample(texture, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def _ensure_stars(self, h, w, count=200):
        """Pre-render star fields with size variation — fully vectorized (no Python loops).

        Creates 3 star layers: large bright stars, medium, and small dim points.
        Each layer is a sparse tensor with gaussian-blurred glow.
        """
        if self._star_field is not None and self._cached_h == h and self._cached_w == w:
            return

        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        positions = torch.rand(count, 2, generator=gen)
        brightness = torch.rand(count, generator=gen)
        phases = torch.rand(count, generator=gen) * 6.28

        # Build star field vectorized: place all stars as single pixels, then blur for glow
        stars = torch.zeros(1, 1, h, w, device=self.device)
        # Convert positions to pixel coords on CPU, then scatter
        ys = (positions[:, 0] * (h - 1)).long().clamp(0, h - 1)
        xs = (positions[:, 1] * (w - 1)).long().clamp(0, w - 1)
        for i in range(count):
            stars[0, 0, ys[i], xs[i]] = brightness[i].item()

        # Large stars: bright ones get gaussian glow
        large_mask = (brightness > 0.65).float()
        large_stars = torch.zeros(1, 1, h, w, device=self.device)
        for i in range(count):
            if large_mask[i] > 0:
                large_stars[0, 0, ys[i], xs[i]] = brightness[i].item()
        if large_stars.max() > 0:
            large_stars = KF.gaussian_blur2d(large_stars, (5, 5), (1.2, 1.2))
            large_stars = large_stars * 3.0  # Boost after blur spreads it

        # Medium stars: gaussian glow with smaller kernel
        med_mask = ((brightness > 0.3) & (brightness <= 0.65)).float()
        med_stars = torch.zeros(1, 1, h, w, device=self.device)
        for i in range(count):
            if med_mask[i] > 0:
                med_stars[0, 0, ys[i], xs[i]] = brightness[i].item()
        if med_stars.max() > 0:
            med_stars = KF.gaussian_blur2d(med_stars, (3, 3), (0.6, 0.6))
            med_stars = med_stars * 2.0

        # Combine: large glow + medium glow + all point stars
        combined = torch.max(stars, torch.max(large_stars, med_stars))

        self._star_field = combined
        self._star_phases = phases
        self._star_brightness = brightness
        self._star_positions = positions
        self._cached_h = h
        self._cached_w = w

    def _nebula(self, h, w, t, audio, params):
        """Cosmic nebula v34 — texture-based UPRISER recreation.

        When 'texture' param points to a nebula photograph, uses that as the gas source
        with animated zoom, rotation, curl noise warping, temporal feedback, and audio reactivity.
        Falls back to procedural v33 when no texture is provided.

        Reference: "Life, the Universe & Everything" — Garret John (UPRISER)
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        texture_path = params.get('texture', '')
        if texture_path:
            tex_mode = params.get('tex_mode', 'nebula')
            if tex_mode == 'dna':
                return self._dna_textured(h, w, t, audio, params)
            elif tex_mode == 'bigbang':
                return self._bigbang_textured(h, w, t, audio, params)
            elif tex_mode == 'solar':
                return self._solar_textured(h, w, t, audio, params)
            elif tex_mode == 'clouds':
                return self._clouds_textured(h, w, t, audio, params)
            elif tex_mode == 'energy':
                return self._energy_textured(h, w, t, audio, params)
            elif tex_mode == 'water':
                return self._water_textured(h, w, t, audio, params)
            elif tex_mode == 'synapses':
                return self._synapses_textured(h, w, t, audio, params)
            else:
                return self._nebula_textured(h, w, t, audio, params)


        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # ── ZOOM ANIMATION: start far, zoom in over first ~10s ──
        zoom_duration = float(params.get('zoom_duration', 10.0))
        zoom_start = float(params.get('zoom_start', 2.2))   # Far: nebula is small
        zoom_end = float(params.get('zoom_end', 1.0))       # Close: fills frame
        zoom_t = min(t * speed / zoom_duration, 1.0)
        # Smooth ease-out curve (fast start, gentle settle)
        zoom_t = 1.0 - (1.0 - zoom_t) ** 2.0
        zoom = zoom_start + (zoom_end - zoom_start) * zoom_t
        # Apply zoom to coordinates
        xx_z = xx * zoom
        yy_z = yy * zoom
        radius = (xx_z ** 2 + yy_z ** 2).sqrt()

        # Slowly rotating coordinates
        rot = t * speed * 0.06
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx_z * cos_r - yy_z * sin_r
        ry = xx_z * sin_r + yy_z * cos_r
        angle = torch.atan2(ry, rx)

        # ── Asymmetric ring shape: 3-lobe modulation (UPRISER: very organic, not circular) ──
        asym_phase = t * speed * 0.015  # Slow drift of asymmetry axis
        lobe_2 = torch.cos(angle - asym_phase) * 0.24          # Primary lobe: +/-0.24
        lobe_3 = torch.cos(angle * 2.0 - asym_phase * 0.7 + 1.2) * 0.14  # Secondary: +/-0.14
        lobe_4 = torch.cos(angle * 3.0 + asym_phase * 0.4 - 0.8) * 0.07  # Tertiary: fine irregularity
        asymmetry = lobe_2 + lobe_3 + lobe_4  # Total: ~0.30 variation around circumference

        # Fine-scale organic edge variation
        ang_noise = torch.sin(angle * 7.0 + t * speed * 0.06) * 0.015 \
                  + torch.sin(angle * 13.0 + t * speed * 0.04 + 2.1) * 0.010 \
                  + torch.sin(angle * 19.0 - t * speed * 0.03 + 4.5) * 0.006

        # Inner edge: very close to core (UPRISER: no dark gap, gas meets core bloom)
        ring_inner = 0.08 + bass * 0.02 + asymmetry * 0.06
        # Outer edge: extends BEYOND visible area — vignette creates the fade
        ring_outer = 1.45 + bass * 0.05 + asymmetry + ang_noise

        ring_pos = ((radius - ring_inner) / (ring_outer - ring_inner + 1e-4)).clamp(0, 1)
        # DONUT shape: density concentrated at ring_pos 0.15-0.70 (not 0-1)
        # Inner edge ramps up slowly → visible blue void inside
        ring_inner_ramp = ((ring_pos - 0.0) / 0.18).clamp(0, 1).pow(1.5)
        ring_outer_ramp = (1.0 - ((ring_pos - 0.70) / 0.30).clamp(0, 1)).pow(0.8)
        ring_density = (ring_inner_ramp * ring_outer_ramp * 2.0).clamp(0, 2.0)

        # Gentle inner fade to prevent hard edge
        inner_fade_r = 0.06 + asymmetry * 0.04
        inner_fade = ((radius - inner_fade_r) / 0.10).clamp(0, 1).pow(1.0)
        ring_density = ring_density * inner_fade

        # Outer fade: modulated by gas for organic irregular edge (not a circle)
        outer_fade_base = (1.0 - ((radius - ring_outer) / 0.20).clamp(0, 1)).pow(1.0)
        # Gas texture breaks the circular edge — thick gas extends further
        outer_fade = outer_fade_base

        # ── Void kill: VERY small — core bloom fills into gas smoothly ──
        void_kill_r = 0.06 + asymmetry * 0.03  # Tiny void (UPRISER: core merges into gas)
        void_kill = ((radius - void_kill_r) / 0.08).clamp(0, 1).pow(1.0)  # Linear fade

        # ── Gas texture: double domain warped for organic folds ──
        # Faster seed evolution — temporal feedback smooths it into fluid motion
        gas_seed = t * speed * 0.08

        # Primary gas: large flowing clouds — less blurred for visible texture
        gas_clouds = _double_warped_noise(h, w, 3, self.device, gas_seed,
                                          base_scale=45, warp_strength=0.95)
        gas_clouds = KF.gaussian_blur2d(gas_clouds, (9, 9), (3.0, 3.0))  # Less blur → more texture
        gas_clouds = gas_clouds.pow(0.45)

        # Secondary gas: medium detail for density variation
        gas_detail = _double_warped_noise(h, w, 3, self.device, gas_seed + 150.0,
                                          base_scale=18, warp_strength=0.55)
        gas_detail = KF.gaussian_blur2d(gas_detail, (5, 5), (1.5, 1.5))

        # Turbulence layer: wispy edges for filamentary quality
        gas_turb = _turbulence(h, w, 3, self.device, gas_seed + 100.0, base_scale=14)
        gas_turb = _curl_warp(h, w, self.device, gas_turb, seed=gas_seed + 650.0,
                              strength=0.25, base_scale=10)
        gas_turb = KF.gaussian_blur2d(gas_turb, (5, 5), (1.5, 1.5))

        # Ridge filaments: large-scale, curl-warped
        ridge_base = _fractal_noise(h, w, 2, self.device, gas_seed + 200.0, base_scale=18)
        ridges = (ridge_base * 2.0 - 1.0).abs()
        ridges = _curl_warp(h, w, self.device, ridges, seed=gas_seed + 600.0,
                            strength=0.35, base_scale=12)
        ridges = (1.0 - ridges).pow(1.2)
        ridges = ((ridges - 0.20) * 2.0).clamp(0, 1)
        ridges = KF.gaussian_blur2d(ridges, (5, 5), (1.2, 1.2))

        # Dark absorption lanes: flowing but VISIBLE (UPRISER: clear dark structure in gas)
        lane_noise = _warped_noise(h, w, 2, self.device, gas_seed + 500.0,
                                   base_scale=35, warp_strength=0.85)
        lane_noise = _curl_warp(h, w, self.device, lane_noise, seed=gas_seed + 550.0,
                                strength=0.25, base_scale=15)
        dark_lanes = (1.0 - lane_noise).pow(1.8)  # Flowing dark structure
        dark_lanes = KF.gaussian_blur2d(dark_lanes, (11, 11), (3.5, 3.5))  # Smooth flowing lanes

        # Worley filaments: web-like gas threads (Voronoi cell boundaries)
        # Computed at 1/4 resolution for speed (filaments are large-scale structures)
        wh4, ww4 = max(32, h // 4), max(32, w // 4)
        worley = _worley_noise(wh4, ww4, self.device, seed=gas_seed + 700.0, num_cells=5)
        worley = F.interpolate(worley, size=(h, w), mode='bilinear', align_corners=False)
        worley = KF.gaussian_blur2d(worley, (3, 3), (0.8, 0.8))

        # Combined gas: clouds + turbulence for filamentary texture
        gas_combined = (gas_clouds * 0.38 + gas_detail * 0.16 + gas_turb * 0.14
                        + ridges * 0.18 + worley * 0.14).clamp(0, 1)
        # Moderate contrast for visible structure without harsh patches
        gas_combined = ((gas_combined - 0.46) * 2.4 + 0.5).clamp(0, 1)
        # Bass breathing: gas density pulses with the bass
        gas_combined = gas_combined * (0.85 + bass * 0.35)
        gas_combined = gas_combined.clamp(0, 1)
        gas_combined = KF.gaussian_blur2d(gas_combined, (7, 7), (2.2, 2.2))  # Balance: texture + flow

        # Modulate outer_fade with gas texture for organic irregular edge
        outer_fade = outer_fade * (0.25 + gas_combined * 0.75)

        # ── Tendrils: curl-warped density extending beyond ring into space ──
        tendril_noise = _warped_noise(h, w, 3, self.device, gas_seed + 800.0,
                                      base_scale=10, warp_strength=0.8)
        tendril_noise = _curl_warp(h, w, self.device, tendril_noise, seed=gas_seed + 900.0,
                                   strength=0.45, base_scale=8)
        # Ridge tendrils: bright at zero-crossings for filamentary look
        tendril_ridges = (tendril_noise * 2.0 - 1.0).abs()
        tendril_ridges = (1.0 - tendril_ridges).pow(0.8)
        tendrils = ((tendril_ridges - 0.30) * 2.5).clamp(0, 1).pow(0.5)
        tendrils = KF.gaussian_blur2d(tendrils, (5, 5), (1.5, 1.5))
        # Tendrils in outer zone — beyond main ring into deep space
        tendril_zone = ((radius - 1.10) / 0.20).clamp(0, 1) * \
                       (1.0 - ((radius - 2.00) / 0.40).clamp(0, 1))
        tendrils = tendrils * tendril_zone * 0.85

        # ── Angular color variation: UPRISER-balanced warm/cool ──
        # UPRISER: ~40% warm (red/orange), ~45% cool (deep blue), ~15% transition
        ang_raw = torch.sin(angle + asym_phase * 0.5) * 0.5 + 0.5  # [0, 1]
        ang_color = ang_raw.pow(1.3)  # Push toward 0 → more cool/blue sectors
        ang_color2 = (torch.cos(angle * 1.5 + asym_phase * 0.3 + 0.8) * 0.5 + 0.5)

        # ── Multi-layer volumetric compositing ──
        result = torch.zeros(1, 3, h, w, device=self.device)
        accumulated_alpha = torch.zeros(1, 1, h, w, device=self.device)

        # Background: deep blue-purple space (UPRISER: never pure black)
        bg = (1.0 - (radius / (2.5 * zoom)).clamp(0, 1)).pow(2.0)  # Scale with zoom
        result[:, 0:1] += bg * 0.012
        result[:, 1:2] += bg * 0.010
        result[:, 2:3] += bg * 0.055

        # ── Deep space dust: colorful nebula clouds in corners (UPRISER ref) ──
        dust_seed = t * speed * 0.02  # Very slow drift
        dust_clouds = _warped_noise(h, w, 3, self.device, dust_seed + 1200.0,
                                     base_scale=12, warp_strength=0.5)
        dust_clouds = KF.gaussian_blur2d(dust_clouds, (15, 15), (5.0, 5.0))
        dust_clouds2 = _warped_noise(h, w, 2, self.device, dust_seed + 1300.0,
                                      base_scale=18, warp_strength=0.4)
        dust_clouds2 = KF.gaussian_blur2d(dust_clouds2, (11, 11), (4.0, 4.0))
        # Dust visible only in deep space (OUTSIDE the ring edge)
        dust_zone = ((radius - 1.10) / 0.30).clamp(0, 1).pow(0.7)
        # Blue-teal dust (UPRISER: cool deep space, NOT pink/magenta)
        dust_vis = dust_clouds * dust_zone * 0.15
        result[:, 0:1] += dust_vis * 0.04  # Minimal red
        result[:, 1:2] += dust_vis * 0.10  # Teal component
        result[:, 2:3] += dust_vis * 0.50  # Blue dominant
        # Secondary deep blue dust
        dust_vis2 = dust_clouds2 * dust_zone * 0.10
        result[:, 0:1] += dust_vis2 * 0.06  # Very little red
        result[:, 1:2] += dust_vis2 * 0.08  # Slight teal
        result[:, 2:3] += dust_vis2 * 0.40  # Strong blue

        # ── 2 volumetric layers ──
        for layer_i in range(2):
            depth = float(layer_i)

            layer_density = ring_density * gas_combined * (0.7 + depth * 0.3)
            # Dark lanes: VISIBLE flowing structure (UPRISER: dramatic gas contrast)
            lane_strength = 0.65 + depth * 0.15
            layer_density = layer_density * (1.0 - dark_lanes * lane_strength)
            # Add tendrils to density (extends beyond ring)
            layer_density = layer_density + tendrils * (0.6 + depth * 0.4)
            layer_density = layer_density * void_kill * outer_fade

            # Base emission: RICH RADIAL GRADIENT — inner=bright gold, outer=DEEP CRIMSON
            radial_t = ring_pos.clamp(0, 1)
            # Thin gas (low density): blue-teal for inner, purple for outer
            thin_r = 0.06 + depth * 0.08 + radial_t * 0.12
            thin_g = 0.12 + depth * 0.06 - radial_t * 0.08
            thin_b = 0.65 - depth * 0.22 - radial_t * 0.25
            thin_color = (thin_r.mean().item(), thin_g.mean().item(), thin_b.mean().item())
            # Thick gas: inner=bright gold → mid=orange → outer=DEEP CRIMSON
            thick_r = 0.58 + depth * 0.15 + radial_t * 0.35  # Even redder at outer
            thick_g = 0.35 + depth * 0.08 - radial_t * 0.28  # Stronger green kill
            thick_b = 0.03 + depth * 0.02 - radial_t * 0.02
            thick_color = (thick_r.mean().item(), thick_g.mean().item(), thick_b.mean().item())

            layer_color = _emission_absorption(
                layer_density, thin_color, thick_color,
                absorption_strength=1.2 + depth * 1.0)

            # Per-pixel radial color adjustment — stronger for rich gradients
            # Inner ring → warmer/lighter, outer ring → deeper/richer
            layer_color[:, 0:1] += radial_t * 0.18 * layer_density  # More red at edges
            layer_color[:, 1:2] -= radial_t * 0.14 * layer_density  # Kill green at edges

            # PER-PIXEL angular color shift — warm vs cool (UPRISER ref)
            dens_mask = layer_density.clamp(0, 1)
            inner_ring_mask = ((ring_density.clamp(0, 1) - 0.05) * 3.0).clamp(0, 1)
            ang_mask = dens_mask * inner_ring_mask
            # Warm sectors: RICH red-orange-gold (UPRISER: saturated warm side)
            layer_color[:, 0:1] += ang_color * 0.70 * ang_mask * (0.5 + depth * 0.5)
            layer_color[:, 1:2] += ang_color * 0.15 * ang_mask  # Slight warmth
            # Cool sectors: DEEP blue-purple (UPRISER: deep navy-blue, NOT pink)
            cool_mask = (1.0 - ang_color)
            layer_color[:, 0:1] *= (1.0 - cool_mask * 0.88 * ang_mask)  # Very aggressive red kill
            layer_color[:, 1:2] *= (1.0 - cool_mask * 0.45 * ang_mask)  # Strong green kill → navy (not pink)
            layer_color[:, 2:3] += cool_mask * 0.75 * ang_mask  # Strong blue

            # Warm color variation: gold vs amber vs deep red (not uniform orange)
            warm_var = _warped_noise(h, w, 2, self.device, gas_seed + 1500.0 + depth * 50,
                                     base_scale=25, warp_strength=0.5)
            warm_var = KF.gaussian_blur2d(warm_var, (9, 9), (3.0, 3.0))
            warm_zone = ang_color * ang_mask  # Only in warm sectors
            # Deep red patches (less green where warm_var is high)
            layer_color[:, 1:2] -= warm_var * warm_zone * 0.12
            # Gold highlights (more green where warm_var is low)
            layer_color[:, 1:2] += (1.0 - warm_var) * warm_zone * 0.08

            # Self-illumination: dense gas GLOWS (UPRISER: luminous quality)
            glow_boost = (layer_density * gas_combined).clamp(0, 1).pow(0.6) * 0.18
            layer_color = layer_color + glow_boost.expand_as(layer_color) * layer_color

            transmittance = 1.0 - accumulated_alpha
            layer_opacity = (1.0 - torch.exp(-layer_density * (1.0 + depth * 0.8))).clamp(0, 1)
            result = result + layer_color * transmittance
            accumulated_alpha = accumulated_alpha + layer_opacity * transmittance
            accumulated_alpha = accumulated_alpha.clamp(0, 1)

        # ── RAINBOW RING BAND: wide spectral ring at inner gas edge ──
        # UPRISER: vivid spectral transition — yellow/green/blue band around inner ring
        # Wider region: ring_pos 0.01–0.30 (was 0.02–0.22)
        rb_t = ((ring_pos - 0.01) / 0.28).clamp(0, 1)
        # Wide band — gentle rise, gentle fall (UPRISER: subtle blending, not neon)
        rb_mask = ((ring_pos - 0.005) / 0.04).clamp(0, 1) * \
                  (1.0 - ((ring_pos - 0.28) / 0.06).clamp(0, 1))
        # Mask: visible where there's ring density
        rb_ring = (ring_density.clamp(0, 0.5) * 2.0).clamp(0, 1)
        rb_mask = rb_mask * (rb_ring * 0.5 + 0.5) * outer_fade * void_kill
        # Spectral colors (UPRISER: rich yellow/green inner edge):
        # rb_t 0=inner (blue) → 0.25=teal → 0.45=yellow-green → 0.7=orange → 1.0=red
        rb_r = ((rb_t - 0.35) * 2.2).clamp(0, 1)     # Red from 0.35 outward
        rb_g = (1.0 - (rb_t - 0.42).abs() * 3.5).clamp(0, 1)  # Green peaks at 0.42
        rb_b = (1.0 - rb_t * 1.8).clamp(0, 1)        # Blue fades through inner half
        # Strong brightness
        rainbow_strength = rb_mask * 1.40
        result[:, 0:1] += rb_r * rainbow_strength
        result[:, 1:2] += rb_g * rainbow_strength * 0.95
        result[:, 2:3] += rb_b * rainbow_strength

        # ── Tangential streaks: visible gas flow direction ──
        # 1D noise in angular direction → appears as circular gas streams swept by rotation
        streak_seed = t * speed * 0.12  # Faster evolution for visible flow
        streak_1 = (torch.sin(angle * 14.0 + streak_seed * 4.0) * 0.5 + 0.5)
        streak_2 = (torch.sin(angle * 23.0 + streak_seed * 2.5 + 1.7) * 0.5 + 0.5)
        streak_3 = (torch.sin(angle * 8.0 - streak_seed * 1.5 + 3.2) * 0.5 + 0.5)
        streaks = (streak_1 * 0.5 + streak_2 * 0.3 + streak_3 * 0.2).clamp(0, 1)
        # Streaks modulate ring brightness — bright/dark bands along circumference
        streak_ring_mask = (ring_density.clamp(0, 1) * outer_fade * void_kill)
        streak_mod = (0.70 + streaks * 0.30) * streak_ring_mask + (1.0 - streak_ring_mask)
        result = result * streak_mod.expand_as(result)

        # ── Direct gas modulation for DRAMATIC cloud structure ──
        ring_vis = ring_density.clamp(0, 1) * outer_fade * void_kill
        # Lower floor for deeper darks → dramatic cloud contrast (UPRISER quality)
        gas_mod = (0.12 + gas_combined * 0.88)  # Range: 0.12 → 1.0
        ring_mod_mask = ring_vis.clamp(0, 1)
        modulation = 1.0 - ring_mod_mask * (1.0 - gas_mod)
        result = result * modulation.expand_as(result)

        # Gold wisps at density peaks — WARM SECTORS ONLY (angular-masked)
        wisps = _warped_noise(h, w, 3, self.device, gas_seed + 300.0,
                              base_scale=16, warp_strength=0.4)
        wisps = ((wisps - 0.40) * 2.5).clamp(0, 1)
        wisps = KF.gaussian_blur2d(wisps, (5, 5), (1.2, 1.2))
        gold_wisps = ring_vis * wisps * gas_clouds * ang_color * 0.35  # Angular mask!
        result[:, 0:1] += gold_wisps * 0.52
        result[:, 1:2] += gold_wisps * 0.40
        result[:, 2:3] += gold_wisps * 0.06

        # Hot spots — bright energy points (warm-biased)
        hotspots = (gas_clouds * ridges).clamp(0, 1).pow(0.6) * ring_vis * 0.30
        hot_warm_bias = (0.3 + ang_color * 0.7)  # Warm sectors get full, cool get 30%
        result[:, 0:1] += hotspots * 0.45 * hot_warm_bias
        result[:, 1:2] += hotspots * 0.30 * hot_warm_bias
        # Cool sectors get blue hotspots instead
        result[:, 2:3] += hotspots * 0.35 * (1.0 - ang_color * 0.7)

        # (Pink rim removed in v28 — UPRISER has no visible boundary line)

        # ── Outer teal-blue ring: UPRISER has visible blue halo at ring edge ──
        outer_ring_start = 0.75 + asymmetry * 0.3
        outer_ring_end = 1.15 + asymmetry * 0.3
        outer_ring = ((radius - outer_ring_start) / (outer_ring_end - outer_ring_start)).clamp(0, 1)
        outer_ring = outer_ring * (1.0 - ((radius - outer_ring_end) / 0.15).clamp(0, 1))
        outer_ring = outer_ring * gas_combined * 0.35  # Gas-modulated for organic look
        outer_ring = KF.gaussian_blur2d(outer_ring, (9, 9), (3.0, 3.0))
        # Yellow-teal halo on warm side, blue on cool side (UPRISER: warm outer halo)
        result[:, 0:1] += outer_ring * (0.03 + ang_color * 0.20)  # Warm side gets yellow
        result[:, 1:2] += outer_ring * (0.15 + ang_color * 0.12)  # Green-teal
        result[:, 2:3] += outer_ring * (0.50 - ang_color * 0.15)  # Blue dominant on cool

        # ── Deep red outer glow: UPRISER has rich crimson gas extending into dark space ──
        outer_gas_zone = ((radius - 0.75) / 0.45).clamp(0, 1) * \
                         (1.0 - ((radius - 1.50) / 0.25).clamp(0, 1))
        outer_gas_vis = gas_clouds * outer_gas_zone * 0.22
        # Deep crimson-red extending into space (warm side dominant)
        warm_outer = ang_color.clamp(0.15, 1.0)
        result[:, 0:1] += outer_gas_vis * warm_outer * 0.65  # Rich red
        result[:, 1:2] += outer_gas_vis * warm_outer * 0.10
        result[:, 2:3] += outer_gas_vis * (1.0 - warm_outer) * 0.25  # Blue on cool side

        # ── Deep space darkening FIRST: kill ring gas beyond edge ──
        space_darken = (1.0 - ((radius - 1.15) / 0.25).clamp(0, 1)).pow(1.0)
        space_darken = space_darken * 0.90 + 0.10  # Slightly higher floor
        result = result * space_darken.expand_as(result)

        # ── Cinematic bloom AFTER darkening: luminous glow + softened transitions ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.24  # Strong bloom for luminous UPRISER quality

        # ── Edge vignette: darken ALL channels at edges → clean dark space ──
        edge_vignette = ((radius - 0.95) / 0.30).clamp(0, 1)
        result *= (1.0 - edge_vignette * 0.55).expand_as(result)  # Overall darken
        result[:, 0:1] *= (1.0 - edge_vignette * 0.25)  # Extra red kill → cooler edge

        # Tendril color: blue-teal wisps into deep space (NO magenta)
        tendril_vis = tendrils * 0.35
        result[:, 0:1] += tendril_vis * 0.04  # Almost no red
        result[:, 1:2] += tendril_vis * 0.12  # Teal
        result[:, 2:3] += tendril_vis * 0.40  # Blue dominant

        # ── Pink/magenta wisps: UPRISER has pink clouds in transition sectors ──
        # Pink appears at the boundary between warm and cool angular sectors
        pink_ang = (torch.sin(angle * 1.0 + asym_phase * 0.3 + 1.5) * 0.5 + 0.5)
        pink_ang = (pink_ang * (1.0 - pink_ang) * 4.0).pow(0.5)  # Peaks at boundary
        pink_gas = gas_clouds * pink_ang * ring_density.clamp(0, 1) * outer_fade * void_kill
        pink_zone_r = ((ring_pos - 0.10) / 0.25).clamp(0, 1) * \
                      (1.0 - ((ring_pos - 0.50) / 0.20).clamp(0, 1))
        pink_vis = pink_gas * pink_zone_r * 0.15  # Subtle, not dominant
        result[:, 0:1] += pink_vis * 0.45  # Moderate pink red
        result[:, 1:2] += pink_vis * 0.08
        result[:, 2:3] += pink_vis * 0.50  # More blue in pink → purple-pink

        # ── Inner transition: angular-dependent glow bridging core → ring gas ──
        inner_band_inner = 0.06
        inner_band_outer = 0.25
        inner_pos = ((radius - inner_band_inner) / (inner_band_outer - inner_band_inner)).clamp(0, 1)
        transition_vis = ((radius - 0.04) / 0.06).clamp(0, 1) * \
                         (1.0 - ((radius - 0.28) / 0.06).clamp(0, 1))
        gas_mod_fade = ((radius - 0.10) / 0.10).clamp(0, 1)
        transition_gas = transition_vis * (0.40 + gas_clouds * 0.30 * gas_mod_fade) * 0.45
        transition_gas = transition_gas * void_kill
        # Warm sectors: golden fill. Cool sectors: blue-purple fill
        trans_warm = ang_color.clamp(0, 1)
        result[:, 0:1] += transition_gas * (0.45 + inner_pos * 0.15) * (0.2 + trans_warm * 0.8)
        result[:, 1:2] += transition_gas * (0.25 + inner_pos * 0.05) * (0.2 + trans_warm * 0.8)
        result[:, 2:3] += transition_gas * (0.12 + (1.0 - trans_warm) * 0.30)

        # ── Inner void — compute masks for later application ──
        void_radius = 0.50 + asymmetry * 0.14
        void_mask = (1.0 - (radius / void_radius).clamp(0, 1)).pow(0.5)
        inner_sparse = (1.0 - ring_density.clamp(0, 0.25) * 4.0).clamp(0, 1)

        # ── Core: soft warm star + wide bloom (UPRISER: large soft golden glow) ──
        core_size = 0.06 + beat * 0.015
        core = (1.0 - (radius / core_size).clamp(0, 1)).pow(2.0)

        # 3-cascade bloom: tight warm glow, doesn't flood void (UPRISER: compact core glow)
        core_bloom_1 = (1.0 - (radius / 0.10).clamp(0, 1)).pow(1.0) * 1.0   # Hot inner
        core_bloom_2 = (1.0 - (radius / 0.20).clamp(0, 1)).pow(1.3) * 0.50  # Medium spread
        core_bloom_3 = (1.0 - (radius / 0.35).clamp(0, 1)).pow(1.8) * 0.20  # Gentle bridge

        all_bloom = core_bloom_1 + core_bloom_2 + core_bloom_3
        # White core → golden-amber outer bloom
        bloom_warmth = (radius / 0.25).clamp(0, 1)
        result[:, 0:1] += core * 1.0 + all_bloom * (0.90 + bloom_warmth * 0.10)
        result[:, 1:2] += core * 0.92 + all_bloom * (0.68 - bloom_warmth * 0.14)
        result[:, 2:3] += core * 0.60 + all_bloom * (0.28 - bloom_warmth * 0.16)

        # ── AFTER core bloom: force deep navy blue in void ──
        # This kills warm bloom bleed, making the void dark navy (UPRISER: deep space between core and ring)
        inner_vis = void_mask * inner_sparse * 0.18
        result[:, 0:1] += inner_vis * 0.02
        result[:, 1:2] += inner_vis * 0.02
        result[:, 2:3] += inner_vis * 0.50
        # Force navy: kill warm tones in void (core bloom spillage)
        inner_blue_force = void_mask * (1.0 - ring_pos.clamp(0, 0.30) / 0.30).clamp(0, 1) * 0.80
        # Protect the core point (r<0.10) — it should stay white-gold
        core_protect = (radius / 0.10).clamp(0, 1)
        inner_blue_force = inner_blue_force * core_protect
        result[:, 0:1] *= (1.0 - inner_blue_force * 0.85)  # Kill red → navy
        result[:, 1:2] *= (1.0 - inner_blue_force * 0.60)  # Kill green → navy (not teal)
        result[:, 2:3] += inner_blue_force * 0.10  # Add a little blue

        # ── Stars: MORE visible, especially inside the blue void (UPRISER) ──
        self._ensure_stars(h, w, count=1200)  # More stars
        # Stars visible where gas is dim OR in inner void
        brightness_mask = result.max(dim=1, keepdim=True)[0]
        dark_mask = (1.0 - brightness_mask * 1.2).clamp(0, 1)
        # Boost stars inside ring void (blue area has lots of visible stars)
        void_star_boost = void_mask * inner_sparse * 1.5
        twinkle_mod = 0.7 + 0.3 * math.sin(t * speed * 1.5)
        star_vis = self._star_field * (dark_mask + void_star_boost).clamp(0, 1) * twinkle_mod * 1.8
        result[:, 0:1] += star_vis * 0.85
        result[:, 1:2] += star_vis * 0.90
        result[:, 2:3] += star_vis * 1.0

        # ── Beat flash ──
        if beat > 0.25:
            flash = core_bloom_3 * beat * 0.4
            result += flash.expand_as(result)

        # (bloom applied before space darkening, not here)

        # ── Temporal feedback: spatially-aware — strong in ring, weak in space ──
        # Ring region: high persistence (gas flows organically between frames)
        # Deep space: low persistence (stays crisp and dark, no brightness accumulation)
        feedback = float(params.get('feedback', 0.75))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t, strength=0.035)
            # Slight diffusion on trail (gas spreads as it flows)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            # Spatial mask: ring_density determines where feedback is strong
            # Inside ring: full feedback → smooth flowing gas
            # Outside ring: minimal feedback → crisp dark space
            fb_mask = (ring_density * outer_fade * void_kill).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0))  # Smooth transition
            fb_mask = fb_mask * 0.85 + 0.08  # Range [0.08, 0.93] — some feedback everywhere
            # Beat breaks feedback for energy bursts
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.3)
            effective_fb = effective_fb.expand_as(result)
            result = advected * effective_fb + result * (1.0 - effective_fb)
        # Cache current frame for next iteration
        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w

        return result.clamp(0, 1)

    def _nebula_textured(self, h, w, t, audio, params):
        """Texture-based nebula v4 — evolving gas, sweeping scan-lines, color drift.

        v4 improvements over v3:
        - Warp evolution: 0.06 → 0.22 over 30s (gas becomes more turbulent)
        - Reduced feedback: 0.65 → 0.30 (lets changes show through)
        - Dual sweep lines: horizontal brightness bars that scan vertically
        - Faster curl noise: seed rate 0.08 → 0.18 for more visible motion
        - Color drift: subtle hue rotation per region over time
        - Radial ripples: expanding concentric brightness rings
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture_path = params.get('texture', '')

        # Load texture (cached)
        texture = self._load_texture(texture_path)

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)

        # ── ZOOM ANIMATION — continuous camera push into the nebula ──
        zoom_duration = float(params.get('zoom_duration', 45.0))
        zoom_start = float(params.get('zoom_start', 1.0))
        zoom_end = float(params.get('zoom_end', 1.55))
        zoom_t = min(t * speed / zoom_duration, 1.0)
        zoom_t = 1.0 - (1.0 - zoom_t) ** 2.0  # Ease-out
        zoom = zoom_start + (zoom_end - zoom_start) * zoom_t
        zoom += bass * 0.02 + 0.015 * math.sin(t * speed * 0.25)

        # ── SLOW ROTATION ──
        rot = t * speed * 0.035
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        radius = (rx ** 2 + ry ** 2).sqrt()

        # ── CURL NOISE — faster seed evolution for visible gas motion ──
        warp_seed = t * speed * 0.18  # was 0.08
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=6, octaves=4)
        tex_radius = radius / zoom
        ring_zone = (1.0 - ((tex_radius - 0.35) / 0.45).abs().clamp(0, 1)).pow(0.4)

        # ── EVOLVING WARP — gas becomes more turbulent over time ──
        warp_t = min(t * speed / 30.0, 1.0)
        warp_base = 0.06 + warp_t * 0.16  # 0.06 → 0.22 over 30s
        warp_str = warp_base * (0.4 + bass * 0.5) * ring_zone

        # Second warp layer: larger, slower swirls
        curl_x2, curl_y2 = _curl_noise(h, w, self.device, seed=warp_seed * 0.3 + 100,
                                         base_scale=3, octaves=2)
        warp_str2 = 0.06 * (0.5 + bass * 0.3)

        # ── Build UV grid ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + curl_x2.squeeze(0).squeeze(0) * warp_str2
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + curl_y2.squeeze(0).squeeze(0) * warp_str2
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS with progressive boost ──
        bright_t = min(t * speed / 30.0, 1.0)
        brightness_arc = 0.80 + bright_t * 0.25  # 0.80 → 1.05
        brightness_mod = brightness_arc + bass * 0.30 + beat * 0.12
        result = sampled * brightness_mod

        # ── DUAL SWEEP LINES — horizontal bars that scan vertically ──
        sweep_speed_1 = t * speed * 0.12
        sweep_speed_2 = t * speed * 0.08
        sweep_y1 = math.sin(sweep_speed_1) * 0.6  # Oscillates -0.6 to +0.6
        sweep_y2 = math.cos(sweep_speed_2 + 1.5) * 0.5
        sweep_width = 0.12
        sweep_1 = (1.0 - ((yy - sweep_y1).abs() / sweep_width).clamp(0, 1)).pow(1.5)
        sweep_2 = (1.0 - ((yy - sweep_y2).abs() / sweep_width).clamp(0, 1)).pow(1.5)
        sweep_mask = (sweep_1 * 0.20 + sweep_2 * 0.15).unsqueeze(0).unsqueeze(0)
        result = result + result * sweep_mask

        # ── RADIAL RIPPLES — expanding concentric rings ──
        ripple_phase = t * speed * 0.5
        radius_1d = radius.unsqueeze(0).unsqueeze(0)
        ripple = torch.sin(radius_1d * 25.0 - ripple_phase * 6.0) * 0.5 + 0.5
        ripple_fade = (1.0 - (radius_1d / 0.8).clamp(0, 1)).pow(0.8)  # Fade at edges
        ripple_strength = 0.08 * ripple_fade * (0.6 + bass * 0.4)
        result = result + result * ripple * ripple_strength

        # ── COLOR DRIFT — subtle hue shift by region over time ──
        color_phase = t * speed * 0.04
        # Warm shift in upper half, cool shift in lower half
        y_factor = yy.unsqueeze(0).unsqueeze(0)
        r_shift = 1.0 + 0.06 * torch.sin(torch.tensor(color_phase, device=self.device)) * y_factor
        b_shift = 1.0 - 0.06 * torch.sin(torch.tensor(color_phase + 1.0, device=self.device)) * y_factor
        result[:, 0:1] = result[:, 0:1] * r_shift
        result[:, 2:3] = result[:, 2:3] * b_shift

        # ── CORE: bright star + bloom ──
        core_size = 0.04 + beat * 0.010
        core = (1.0 - (radius_1d / core_size).clamp(0, 1)).pow(2.5)
        core_bloom_1 = (1.0 - (radius_1d / 0.08).clamp(0, 1)).pow(1.2) * 0.70
        core_bloom_2 = (1.0 - (radius_1d / 0.18).clamp(0, 1)).pow(1.5) * 0.30
        core_bloom_3 = (1.0 - (radius_1d / 0.35).clamp(0, 1)).pow(2.0) * 0.10
        all_bloom = core_bloom_1 + core_bloom_2 + core_bloom_3
        bloom_warmth = (radius_1d / 0.20).clamp(0, 1)
        result[:, 0:1] += core * 0.90 + all_bloom * (0.85 + bloom_warmth * 0.10)
        result[:, 1:2] += core * 0.85 + all_bloom * (0.60 - bloom_warmth * 0.10)
        result[:, 2:3] += core * 0.55 + all_bloom * (0.20 - bloom_warmth * 0.10)

        # ── STARS ──
        self._ensure_stars(h, w, count=1200)
        brightness_mask = result.max(dim=1, keepdim=True)[0]
        dark_mask = (1.0 - brightness_mask * 1.5).clamp(0, 1)
        twinkle = 0.7 + 0.3 * math.sin(t * speed * 1.5)
        star_vis = self._star_field * dark_mask * twinkle * 1.5
        result[:, 0:1] += star_vis * 0.85
        result[:, 1:2] += star_vis * 0.90
        result[:, 2:3] += star_vis * 1.0

        # ── VIGNETTE ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.6) / 0.4).clamp(0, 1).pow(1.2) * 0.7)
        result = result * vignette

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.15

        # ── Beat flash ──
        if beat > 0.25:
            flash = core_bloom_2 * beat * 0.30
            result += flash.expand_as(result)

        # ── TEMPORAL FEEDBACK — reduced to let changes show ──
        feedback = float(params.get('feedback', 0.30))  # was 0.65
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.025, rot_strength=0.008)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            sampled_brightness = sampled.mean(dim=1, keepdim=True)
            fb_mask = (sampled_brightness * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0))
            fb_mask = fb_mask * 0.75 + 0.08
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.3)
            effective_fb = effective_fb.expand_as(result)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w

        return result.clamp(0, 1)

    def _dna_textured(self, h, w, t, audio, params):
        """DNA helix v5 — zoom IN (works WITH brightness arc), dual scan-lines, evolving warp.

        v5 fixes v4's brightness conflict (zoom-OUT canceled brightness arc):
        - Zoom IN (1.0→1.40) so brightness arc is visible (dark→bright)
        - Dual scan-lines: main (downward) + secondary (upward, teal-tinted)
        - Evolving warp: curl strength grows 0.04→0.10 over time
        - Stronger diagonal drift for "traveling along helix" feel
        - Feedback reduced to 0.40 for more visible frame change
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture_path = params.get('texture', '')
        texture = self._load_texture(texture_path)

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)

        # ── ZOOM IN — works WITH brightness arc (close-up gets brighter) ──
        zoom_duration = float(params.get('zoom_duration', 40.0))
        zoom_start = float(params.get('zoom_start', 1.0))
        zoom_end = float(params.get('zoom_end', 1.40))
        zoom_t = min(t * speed / zoom_duration, 1.0)
        zoom_t = 1.0 - (1.0 - zoom_t) ** 2.0
        zoom = zoom_start + (zoom_end - zoom_start) * zoom_t
        zoom += 0.025 * math.sin(t * speed * 0.18) + bass * 0.020

        # ── SPIRAL CAMERA — rotation accelerates ──
        rot_speed = 0.025 + min(t * speed / 60.0, 0.015)
        rot = t * speed * rot_speed
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── SPIRAL DRIFT — stronger diagonal travel along helix axis ──
        v_offset = 0.12 * math.sin(t * speed * 0.11)
        h_offset = 0.08 * math.sin(t * speed * 0.08 + 0.7)
        diag_drift = t * speed * 0.010  # Stronger drift (was 0.006)

        # ── CURL NOISE — aggressive evolution (barely visible → heavy warp) ──
        warp_seed = t * speed * 0.12  # Faster seed
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=6, octaves=4)
        warp_evolution = min(t * speed / 25.0, 1.0)
        warp_str = (0.02 + warp_evolution * 0.10) * (0.5 + bass * 0.6)  # 0.02→0.12

        # ── Build UV grid ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + h_offset + diag_drift
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + v_offset + diag_drift * 0.5
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS ARC — dark reveal → full power (now works with zoom-in) ──
        time_arc = min(t * speed / 28.0, 1.0)
        time_arc = time_arc ** 0.7
        brightness_base = 0.55 + time_arc * 0.50  # 0.55 → 1.05
        brightness_mod = brightness_base + bass * 0.30 + beat * 0.18
        result = sampled * brightness_mod

        # ── DUAL SCAN-LINES — two beams reading the helix ──
        # Primary: green, travels downward
        scan_pos = -0.8 + (t * speed / 30.0) * 1.6
        scan_pos += 0.12 * math.sin(t * speed * 0.25)
        scan_width = 0.10 + bass * 0.04
        scan_dist = (yy - scan_pos).abs()
        scan_band = (1.0 - (scan_dist / scan_width).clamp(0, 1)).pow(2.0)
        scan_band = scan_band.unsqueeze(0).unsqueeze(0)
        scan_intensity = 0.30 + beat * 0.25  # Brighter (was 0.20)
        result[:, 1:2] += scan_band * scan_intensity
        result[:, 2:3] += scan_band * scan_intensity * 0.3
        result[:, 0:1] += scan_band * scan_intensity * 0.05

        # Secondary: teal, travels UPWARD (opposite direction)
        scan_pos2 = 0.6 - (t * speed / 30.0) * 1.2
        scan_pos2 += 0.10 * math.sin(t * speed * 0.20 + 2.0)
        scan_dist2 = (yy - scan_pos2).abs()
        scan_band2 = (1.0 - (scan_dist2 / (scan_width * 0.8)).clamp(0, 1)).pow(2.0)
        scan_band2 = scan_band2.unsqueeze(0).unsqueeze(0)
        scan2_intensity = 0.15 + beat * 0.15
        result[:, 1:2] += scan_band2 * scan2_intensity * 0.6  # Less green
        result[:, 2:3] += scan_band2 * scan2_intensity * 1.0  # More blue = teal
        result[:, 0:1] += scan_band2 * scan2_intensity * 0.15  # Slight warm

        # ── GREEN GLOW — multi-radius bloom, tripled evolution ──
        green_ch = sampled[:, 1:2]
        red_ch = sampled[:, 0:1]
        blue_ch = sampled[:, 2:3]
        green_dominant = ((green_ch - red_ch * 0.7 - blue_ch * 0.5) * 3.0).clamp(0, 1)
        green_bloom_inner = KF.gaussian_blur2d(green_dominant, (15, 15), (4.0, 4.0))
        green_bloom_outer = KF.gaussian_blur2d(green_dominant, (41, 41), (14.0, 14.0))
        glow_evolution = 0.3 + min(t * speed / 22.0, 1.0) * 1.5
        glow_pulse = glow_evolution + bass * 0.5
        result[:, 1:2] += green_bloom_inner * 0.35 * glow_pulse
        result[:, 1:2] += green_bloom_outer * 0.18 * glow_pulse
        result[:, 0:1] += green_bloom_outer * 0.04
        result[:, 2:3] += green_bloom_inner * 0.12 + green_bloom_outer * 0.08

        # ── BACKGROUND — shifts from deep blue to purple-magenta ──
        brightness = sampled.mean(dim=1, keepdim=True)
        dark_mask = (1.0 - brightness * 2.5).clamp(0, 1)
        purple_shift = min(t * speed / 25.0, 1.0) * 0.04
        result[:, 0:1] -= dark_mask * (0.02 - purple_shift * 0.8)
        result[:, 2:3] += dark_mask * (0.03 + purple_shift)

        # ── STARS ──
        self._ensure_stars(h, w, count=800)
        twinkle = 0.6 + 0.4 * math.sin(t * speed * 1.2)
        star_vis = self._star_field * dark_mask * twinkle * 0.9
        result[:, 0:1] += star_vis * 0.5
        result[:, 1:2] += star_vis * 0.7
        result[:, 2:3] += star_vis * 1.0

        # ── VIGNETTE ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.65) / 0.35).clamp(0, 1).pow(1.2) * 0.55)
        result = result * vignette

        # ── CLEARING HAZE — starts foggy, clears to reveal detail ──
        haze_t = min(t * speed / 25.0, 1.0)
        haze_amount = 0.35 * (1.0 - haze_t)  # 35% fog at start, 0% at end
        if haze_amount > 0.01:
            haze_color = torch.zeros_like(result)
            haze_color[:, 2:3] = 0.06  # Very faint blue tint
            result = result * (1.0 - haze_amount) + haze_color * haze_amount

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.16

        # ── Beat flash — green pulse at both scan positions ──
        if beat > 0.25:
            flash_mask = KF.gaussian_blur2d(green_dominant, (51, 51), (18.0, 18.0))
            flash_mask = flash_mask + scan_band * 0.3 + scan_band2 * 0.2
            result[:, 1:2] += flash_mask * beat * 0.35
            result[:, 2:3] += flash_mask * beat * 0.15
            result[:, 0:1] += flash_mask * beat * 0.06

        # ── TEMPORAL FEEDBACK — low for visible change ──
        feedback = float(params.get('feedback', 0.30))  # Was 0.40
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.022, rot_strength=0.010)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            sampled_brightness = sampled.mean(dim=1, keepdim=True)
            fb_mask = (sampled_brightness * 2.5).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0))
            fb_mask = fb_mask * 0.55 + 0.05
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.35)
            effective_fb = effective_fb.expand_as(result)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w

        return result.clamp(0, 1)

    def _bigbang_textured(self, h, w, t, audio, params):
        """Big Bang v2 — texture-based with dramatic radial expansion, energy pulses.

        Uses a photograph of galaxies/stars radiating from center with:
        strong radial zoom-out (visible expansion), beat-reactive light rays,
        chromatic aberration at edges, supernova beat flashes, deep parallax.
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture_path = params.get('texture', '')
        texture = self._load_texture(texture_path)

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)
        radius = (xx ** 2 + yy ** 2).sqrt()
        angle = torch.atan2(yy, xx)

        # ── ZOOM — dramatic radial expansion (zoom OUT = universe expands) ──
        zoom_duration = float(params.get('zoom_duration', 35.0))
        zoom_start = float(params.get('zoom_start', 1.7))
        zoom_end = float(params.get('zoom_end', 0.82))
        zoom_t = min(t * speed / zoom_duration, 1.0)
        zoom_t = zoom_t ** 0.6  # Ease-in (slow start, then accelerating expansion)
        zoom = zoom_start + (zoom_end - zoom_start) * zoom_t
        # Bass = expansion burst + breathing
        zoom -= bass * 0.05 + 0.02 * math.sin(t * speed * 0.18)

        # ── SLOW ROTATION ──
        rot = t * speed * 0.025
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── RADIAL WARP — particles fly outward from center ──
        warp_seed = t * speed * 0.10
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=5, octaves=4)
        radial_strength = (radius * 1.5).clamp(0, 1)
        warp_str = 0.10 * (0.5 + bass * 0.7) * radial_strength

        # Radial push: accumulates with time (galaxies fly apart)
        radial_push = 0.04 * t * speed * (0.4 + bass * 0.4)
        # Clamp so edges stay clean
        radial_push = min(radial_push, 0.30)
        safe_r = radius.clamp(min=0.01)
        push_x = (xx / safe_r) * radial_push * radial_strength
        push_y = (yy / safe_r) * radial_push * radial_strength

        # ── Build UV grid ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + push_x * 0.35
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + push_y * 0.35
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── CHROMATIC ABERRATION — energy distortion (mid-field only) ──
        # Fade to zero where UV approaches texture edge (prevents color fringing)
        uv_safe = ((1.0 - u_norm.abs()).clamp(0, 0.20) / 0.20) * \
                  ((1.0 - v_norm.abs()).clamp(0, 0.20) / 0.20)
        chroma_str = 0.004 * (1.0 + bass * 0.5) * radial_strength * uv_safe
        dir_x = (xx / safe_r).clamp(-1, 1)
        dir_y = (yy / safe_r).clamp(-1, 1)
        u_r = (u_norm + dir_x * chroma_str).clamp(-1, 1)
        v_r = (v_norm + dir_y * chroma_str).clamp(-1, 1)
        u_b = (u_norm - dir_x * chroma_str).clamp(-1, 1)
        v_b = (v_norm - dir_y * chroma_str).clamp(-1, 1)
        grid_r = torch.stack([u_r, v_r], dim=-1).unsqueeze(0)
        grid_b = torch.stack([u_b, v_b], dim=-1).unsqueeze(0)
        sampled_r = self._sample_texture(texture, grid_r)
        sampled_b = self._sample_texture(texture, grid_b)
        sampled[:, 0:1] = sampled_r[:, 0:1]
        sampled[:, 2:3] = sampled_b[:, 2:3]

        # Fade ALL channels to black near UV boundary (smooth radial fade, not rectangular)
        uv_max = torch.max(u_norm.abs(), v_norm.abs())  # Chebyshev distance (smooth corners)
        uv_fade = ((1.0 - uv_max).clamp(0, 0.15) / 0.15).pow(0.6)
        sampled = sampled * uv_fade.unsqueeze(0).unsqueeze(0)

        # ── AUDIO REACTIVITY ──
        brightness_mod = 0.75 + bass * 0.45 + beat * 0.25
        result = sampled * brightness_mod

        # ── CENTER BLOOM — white-hot core of the Big Bang ──
        radius_1d = radius.unsqueeze(0).unsqueeze(0)
        core_size = 0.07 + beat * 0.03 + bass * 0.02
        core = (1.0 - (radius_1d / core_size).clamp(0, 1)).pow(1.8)
        bloom_1 = (1.0 - (radius_1d / 0.14).clamp(0, 1)).pow(1.0) * 0.90
        bloom_2 = (1.0 - (radius_1d / 0.30).clamp(0, 1)).pow(1.2) * 0.45
        bloom_3 = (1.0 - (radius_1d / 0.55).clamp(0, 1)).pow(1.6) * 0.18
        all_bloom = bloom_1 + bloom_2 + bloom_3
        # White-hot center → golden → blue edges
        warmth = (1.0 - radius_1d / 0.35).clamp(0, 1)
        result[:, 0:1] += core * 1.0 + all_bloom * (0.95 + warmth * 0.05)
        result[:, 1:2] += core * 0.95 + all_bloom * (0.70 - warmth * 0.05)
        result[:, 2:3] += core * 0.75 + all_bloom * (0.35 - warmth * 0.20)

        # ── RADIAL STREAKS — light rays pulsing with beats ──
        num_rays = 16
        ray_angle = angle * num_rays + t * speed * 0.4
        rays = (torch.cos(ray_angle) * 0.5 + 0.5).pow(2.5)
        # Ray length extends but caps so galaxies stay the hero at late time
        ray_reach = 0.5 + 0.3 * min(t * speed / 25.0, 1.0)
        ray_falloff = (1.0 - (radius / ray_reach).clamp(0, 1)).pow(0.6)
        # Beat-reactive ray intensity — fades as universe expands
        expansion_fade = max(1.0 - t * speed / 50.0, 0.5)
        ray_intensity = 0.10 * (0.5 + bass * 0.5 + beat * 0.7) * expansion_fade
        ray_brightness = rays * ray_falloff * ray_intensity
        ray_brightness = ray_brightness.unsqueeze(0).unsqueeze(0)
        result[:, 0:1] += ray_brightness * 1.0
        result[:, 1:2] += ray_brightness * 0.80
        result[:, 2:3] += ray_brightness * 0.55

        # ── STARS — scattered particles ──
        self._ensure_stars(h, w, count=1500)
        brightness_mask = result.max(dim=1, keepdim=True)[0]
        dark_mask = (1.0 - brightness_mask * 1.2).clamp(0, 1)
        twinkle = 0.6 + 0.4 * math.sin(t * speed * 2.0)
        star_vis = self._star_field * dark_mask * twinkle * 1.2
        result[:, 0:1] += star_vis * 0.90
        result[:, 1:2] += star_vis * 0.85
        result[:, 2:3] += star_vis * 1.0

        # ── VIGNETTE — tightens as zoom decreases to hide edge artifacts ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        # As zoom shrinks (universe expands), vignette moves inward
        vig_start = 0.55 + 0.20 * min(zoom / zoom_start, 1.0)  # Tighter at low zoom
        vig_strength = 0.45 + 0.25 * (1.0 - min(zoom / zoom_start, 1.0))  # Stronger at low zoom
        vignette = (1.0 - ((screen_r / max_r - vig_start) / 0.35).clamp(0, 1).pow(1.2) * vig_strength)
        result = result * vignette

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.22

        # ── Beat flash — SUPERNOVA pulse ──
        if beat > 0.2:
            # Wide white flash from center
            supernova = bloom_3 * beat * 0.60
            result += supernova.expand_as(result)
            # Extra red/orange ring pulse
            ring_pulse = ((radius_1d - 0.15).abs() < 0.08).float() * beat * 0.15
            result[:, 0:1] += ring_pulse
            result[:, 1:2] += ring_pulse * 0.5

        # ── TEMPORAL FEEDBACK ──
        feedback = float(params.get('feedback', 0.50))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.035, rot_strength=0.005)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            sampled_brightness = sampled.mean(dim=1, keepdim=True)
            fb_mask = (sampled_brightness * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0))
            fb_mask = fb_mask * 0.60 + 0.10
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.4)
            effective_fb = effective_fb.expand_as(result)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w

        return result.clamp(0, 1)

    def _solar_textured(self, h, w, t, audio, params):
        """Solar surface — fiery texture with heat shimmer, solar flares, bass-reactive corona."""
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture = self._load_texture(params.get('texture', ''))

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)
        radius = (xx ** 2 + yy ** 2).sqrt()

        # ── ZOOM — slow push into the surface ──
        zoom_dur = float(params.get('zoom_duration', 50.0))
        zoom_s = float(params.get('zoom_start', 1.0))
        zoom_e = float(params.get('zoom_end', 1.45))
        zt = min(t * speed / zoom_dur, 1.0)
        zoom = zoom_s + (zoom_e - zoom_s) * (1.0 - (1.0 - zt) ** 2.0)
        zoom += bass * 0.02 + 0.015 * math.sin(t * speed * 0.22)

        # ── ROTATION — slow but visible ──
        rot = t * speed * 0.035
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── SURFACE DRIFT — lava flows across the surface ──
        drift_x = 0.06 * math.sin(t * speed * 0.07)
        drift_y = t * speed * 0.018  # Slow upward convection

        # ── HEAT SHIMMER — vertical bias (heat rises) ──
        warp_seed = t * speed * 0.12
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=5, octaves=4)
        warp_str = 0.10 * (0.5 + bass * 0.7)
        # Second layer: larger, slower convection cells
        curl_x2, curl_y2 = _curl_noise(h, w, self.device, seed=warp_seed * 0.3 + 50,
                                         base_scale=3, octaves=2)
        warp_str2 = 0.06 * (0.4 + bass * 0.4)
        # Vertical warp stronger than horizontal (heat shimmer)
        shimmer_y = curl_y.squeeze(0).squeeze(0) * warp_str * 1.5 + curl_y2.squeeze(0).squeeze(0) * warp_str2
        shimmer_x = curl_x.squeeze(0).squeeze(0) * warp_str * 0.7 + curl_x2.squeeze(0).squeeze(0) * warp_str2 * 0.5

        # ── Build UV ──
        u = rx + shimmer_x + drift_x
        v = ry + shimmer_y + drift_y
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS — fire pulses with bass ──
        brightness_mod = 0.80 + bass * 0.40 + beat * 0.20
        result = sampled * brightness_mod

        # ── WARM GLOW — enhance orange/red, suppress blue ──
        result[:, 0:1] *= 1.15  # Boost red
        result[:, 1:2] *= 0.95  # Slightly reduce green
        result[:, 2:3] *= 0.70  # Reduce blue for warmer look

        # ── CORONA BLOOM — bright center glow ──
        radius_1d = radius.unsqueeze(0).unsqueeze(0)
        core_size = 0.08 + beat * 0.03 + bass * 0.02
        core = (1.0 - (radius_1d / core_size).clamp(0, 1)).pow(1.5)
        bloom_1 = (1.0 - (radius_1d / 0.20).clamp(0, 1)).pow(1.0) * 0.60
        bloom_2 = (1.0 - (radius_1d / 0.45).clamp(0, 1)).pow(1.5) * 0.25
        result[:, 0:1] += core * 1.0 + bloom_1 * 1.0 + bloom_2 * 0.8
        result[:, 1:2] += core * 0.85 + bloom_1 * 0.7 + bloom_2 * 0.4
        result[:, 2:3] += core * 0.40 + bloom_1 * 0.2 + bloom_2 * 0.05

        # ── VIGNETTE ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.6) / 0.4).clamp(0, 1).pow(1.3) * 0.60)
        result = result * vignette

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.18

        # ── Beat flash — solar flare ──
        if beat > 0.25:
            flare = bloom_1 * beat * 0.50
            result[:, 0:1] += flare
            result[:, 1:2] += flare * 0.6

        # ── TEMPORAL FEEDBACK ──
        feedback = float(params.get('feedback', 0.60))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.030, rot_strength=0.005)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            fb_mask = (sampled.mean(dim=1, keepdim=True) * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0)) * 0.70 + 0.08
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.3)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w
        return result.clamp(0, 1)

    def _clouds_textured(self, h, w, t, audio, params):
        """Golden sunset clouds — slow morphing drift, warm tones, volumetric light."""
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture = self._load_texture(params.get('texture', ''))

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)

        # ── ZOOM — gentle push toward sun ──
        zoom_dur = float(params.get('zoom_duration', 55.0))
        zoom_s = float(params.get('zoom_start', 1.0))
        zoom_e = float(params.get('zoom_end', 1.30))
        zt = min(t * speed / zoom_dur, 1.0)
        zoom = zoom_s + (zoom_e - zoom_s) * (1.0 - (1.0 - zt) ** 2.0)
        zoom += 0.012 * math.sin(t * speed * 0.15)

        # ── HORIZONTAL DRIFT — clouds move right ──
        drift = t * speed * 0.015
        v_drift = 0.005 * math.sin(t * speed * 0.08)

        # ── CURL NOISE — cloud morphing ──
        warp_seed = t * speed * 0.04
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=4, octaves=3)
        warp_str = 0.07 * (0.4 + bass * 0.4)

        # ── Build UV ──
        u = xx + curl_x.squeeze(0).squeeze(0) * warp_str + drift
        v = yy + curl_y.squeeze(0).squeeze(0) * warp_str * 0.5 + v_drift
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS — bring down to cinematic range ──
        brightness_mod = 0.65 + bass * 0.15 + beat * 0.08
        result = sampled * brightness_mod

        # ── CONTRAST COMPRESSION — crush highlights, lift shadows ──
        # This brings 196→~130 avg brightness (cinematic, not blown out)
        result = result.pow(1.3)  # Gamma compress (darkens mids/highlights)

        # ── GOLDEN HOUR ENHANCEMENT — selective warm shift ──
        luminance = sampled.mean(dim=1, keepdim=True)
        bright_areas = (luminance * 1.5).clamp(0, 1)
        # Time-based color temperature: starts golden, shifts to amber
        temp_shift = min(t * speed / 30.0, 1.0)
        result[:, 0:1] += bright_areas * (0.04 + temp_shift * 0.03)  # More red over time
        result[:, 2:3] -= bright_areas * (0.02 + temp_shift * 0.02)  # Less blue over time

        # ── GOD RAYS — light streaming through clouds ──
        sun_x, sun_y = 0.0, -0.3
        ray_r = ((xx - sun_x) ** 2 + (yy - sun_y) ** 2).sqrt()
        ray_angle = torch.atan2(yy - sun_y, xx - sun_x)
        rays = (torch.cos(ray_angle * 8 + t * speed * 0.15) * 0.5 + 0.5).pow(4.0)
        ray_falloff = (1.0 - (ray_r / 1.2).clamp(0, 1)).pow(0.8)
        ray_intensity = rays * ray_falloff * 0.05 * (0.5 + bass * 0.3)
        ray_brightness = ray_intensity.unsqueeze(0).unsqueeze(0)
        result[:, 0:1] += ray_brightness * 0.80
        result[:, 1:2] += ray_brightness * 0.55
        result[:, 2:3] += ray_brightness * 0.20

        # ── VIGNETTE — stronger for cinematic framing ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.50) / 0.50).clamp(0, 1).pow(1.5) * 0.65)
        result = result * vignette

        # ── CINEMATIC BLOOM — reduced to avoid blowout ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (12.0, 12.0))
        result = result + bloom * 0.10

        # ── Beat flash — golden pulse ──
        if beat > 0.3:
            flash = KF.gaussian_blur2d(bright_areas, (41, 41), (15.0, 15.0))
            result[:, 0:1] += flash * beat * 0.12
            result[:, 1:2] += flash * beat * 0.06

        # ── TEMPORAL FEEDBACK ──
        feedback = float(params.get('feedback', 0.60))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.015, rot_strength=0.003)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            fb_mask = (luminance * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0)) * 0.65 + 0.10
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.2)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w
        return result.clamp(0, 1)

    def _energy_textured(self, h, w, t, audio, params):
        """Energy v5b — v5 structure with brightness boost.

        v5 achieved complexity 58.7% but t=2 brightness only 27.
        v5b fixes: brightness_mod 0.55→0.75, gamma 1.25→1.10.
        Keeps: dual curl, radial distortion, wider shockwaves, feedback 0.28.
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture = self._load_texture(params.get('texture', ''))

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)
        radius = (xx ** 2 + yy ** 2).sqrt()
        angle = torch.atan2(yy, xx)

        # ── ZOOM — pulsing inward ──
        zoom_dur = float(params.get('zoom_duration', 35.0))
        zoom_s = float(params.get('zoom_start', 1.05))
        zoom_e = float(params.get('zoom_end', 1.55))
        zt = min(t * speed / zoom_dur, 1.0)
        zoom = zoom_s + (zoom_e - zoom_s) * (1.0 - (1.0 - zt) ** 2.0)
        zoom += bass * 0.04 + 0.025 * math.sin(t * speed * 0.30)

        # ── ROTATION — faster for energy feel ──
        rot = t * speed * 0.06
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── CURL NOISE — dual layer, faster seed for frame variation ──
        warp_seed = t * speed * 0.25  # Much faster seed (was 0.18)
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=5, octaves=4)
        warp_str = 0.07 * (0.5 + bass * 0.6)
        # Second layer: large-scale plasma current
        curl_x2, curl_y2 = _curl_noise(h, w, self.device, seed=warp_seed * 0.35 + 42,
                                         base_scale=3, octaves=2)
        warp_str2 = 0.04 * (0.4 + bass * 0.3)

        # ── RADIAL DISTORTION — texture bends more at edges over time ──
        radial_t = min(t * speed / 30.0, 1.0)
        radial_push = radius * radial_t * 0.03  # Grows from 0 to 0.03 at edges

        # ── Build UV ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + curl_x2.squeeze(0).squeeze(0) * warp_str2
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + curl_y2.squeeze(0).squeeze(0) * warp_str2
        # Radial push: outward distortion at edges
        u = u + xx * radial_push
        v = v + yy * radial_push
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS — brightness arc, no gamma ──
        bright_t = min(t * speed / 30.0, 1.0)
        brightness_arc = 0.65 + bright_t * 0.45  # 0.65 → 1.10 over 30s
        brightness_mod = brightness_arc + bass * 0.35 + beat * 0.20
        result = sampled * brightness_mod

        # ── COLOR ARC — pink → violet over 30s ──
        color_t = min(t * speed / 30.0, 1.0)
        result[:, 0:1] *= 1.05 - color_t * 0.12
        result[:, 2:3] *= 1.03 + color_t * 0.15

        # ── CENTER BLOOM — white-hot core ──
        radius_1d = radius.unsqueeze(0).unsqueeze(0)
        core_size = 0.05 + beat * 0.025 + bass * 0.015
        core = (1.0 - (radius_1d / core_size).clamp(0, 1)).pow(2.0)
        bloom_1 = (1.0 - (radius_1d / 0.12).clamp(0, 1)).pow(1.2) * 0.55
        bloom_2 = (1.0 - (radius_1d / 0.30).clamp(0, 1)).pow(1.5) * 0.22
        result[:, 0:1] += core * 0.85 + bloom_1 * 0.80 + bloom_2 * 0.55
        result[:, 1:2] += core * 0.75 + bloom_1 * 0.40 + bloom_2 * 0.15
        result[:, 2:3] += core * 0.95 + bloom_1 * 0.85 + bloom_2 * 0.55

        # ── SHOCKWAVE RINGS — 4x wider, 2x brighter, more visible ──
        for ring_i in range(3):
            ring_phase = (t * speed * 0.8 + ring_i * 2.1) % 6.0
            ring_r = ring_phase * 0.18
            ring_width = 0.06 + ring_phase * 0.015  # 4x wider (was 0.015+0.005)
            ring_dist = (radius - ring_r).abs()
            ring = (1.0 - (ring_dist / ring_width).clamp(0, 1)).pow(1.2)  # Softer falloff
            ring_fade = max(0.0, min(1.0, 1.0 - ring_phase / 6.0))
            ring_bright = ring * ring_fade * 0.20 * (0.6 + bass * 0.5)  # 2x brighter
            ring_bright = ring_bright.unsqueeze(0).unsqueeze(0)
            result[:, 0:1] += ring_bright * 0.9
            result[:, 1:2] += ring_bright * 0.3
            result[:, 2:3] += ring_bright * 1.0

        # ── ENERGY RAYS ──
        num_rays = 20
        ray_angle_v = angle * num_rays + t * speed * 0.5
        rays = (torch.cos(ray_angle_v) * 0.5 + 0.5).pow(3.0)
        ray_falloff = (1.0 - (radius / 0.7).clamp(0, 1)).pow(0.5)
        ray_intensity = rays * ray_falloff * 0.10 * (0.5 + bass * 0.6 + beat * 0.5)
        ray_brightness = ray_intensity.unsqueeze(0).unsqueeze(0)
        result[:, 0:1] += ray_brightness * 1.0
        result[:, 1:2] += ray_brightness * 0.50
        result[:, 2:3] += ray_brightness * 0.85

        # ── STARS ──
        self._ensure_stars(h, w, count=800)
        dark_mask = (1.0 - result.max(dim=1, keepdim=True)[0] * 1.5).clamp(0, 1)
        twinkle = 0.6 + 0.4 * math.sin(t * speed * 1.8)
        star_vis = self._star_field * dark_mask * twinkle * 0.8
        result += star_vis.expand_as(result)

        # ── VIGNETTE ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.6) / 0.4).clamp(0, 1).pow(1.2) * 0.55)
        result = result * vignette

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.20

        # ── Beat flash — energy burst ──
        if beat > 0.2:
            flash = bloom_2 * beat * 0.50
            result += flash.expand_as(result)

        # ── TEMPORAL FEEDBACK — low for maximum warp visibility ──
        feedback = float(params.get('feedback', 0.28))  # Was 0.35
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.030, rot_strength=0.010)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            fb_mask = (sampled.mean(dim=1, keepdim=True) * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0)) * 0.50 + 0.05
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.5)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w
        return result.clamp(0, 1)

    def _water_textured(self, h, w, t, audio, params):
        """Water macro v6 — growing foam layer, color temperature shift, structural evolution.

        v6 approach: structural changes that create dramatically different PNG detail
        - Foam noise layer that builds 0% → 30% over 30s (adds high-freq detail)
        - Color temperature: warm (red tint) → cool (blue tint) over 30s
        - Evolving warp: 0.06 → 0.22 (texture distortion grows)
        - Moderate drift + breathing for visible flow
        - Reduced feedback (0.30) for responsive change
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture = self._load_texture(params.get('texture', ''))

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)

        # ── ZOOM with tidal breathing ──
        zoom_dur = float(params.get('zoom_duration', 45.0))
        zoom_s = float(params.get('zoom_start', 1.0))
        zoom_e = float(params.get('zoom_end', 1.45))
        zt = min(t * speed / zoom_dur, 1.0)
        zoom = zoom_s + (zoom_e - zoom_s) * (1.0 - (1.0 - zt) ** 2.0)
        zoom += 0.06 * math.sin(t * speed * 0.13) + bass * 0.03

        # ── ROTATION ──
        rot = t * speed * 0.020
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── MODERATE DRIFT ──
        drift_x = t * speed * 0.035 + 0.06 * math.sin(t * speed * 0.10)
        drift_y = 0.04 * math.sin(t * speed * 0.07 + 0.5) + t * speed * 0.012

        # ── EVOLVING WARP ──
        warp_seed = t * speed * 0.15
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=5, octaves=4)
        warp_t = min(t * speed / 30.0, 1.0)
        warp_base = 0.06 + warp_t * 0.16  # 0.06 → 0.22
        warp_str = warp_base * (0.5 + bass * 0.6)
        curl_x2, curl_y2 = _curl_noise(h, w, self.device, seed=warp_seed * 0.4 + 77,
                                         base_scale=3, octaves=2)
        warp_str2 = 0.06 * (0.5 + bass * 0.4)

        # ── Build UV ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + curl_x2.squeeze(0).squeeze(0) * warp_str2 + drift_x
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + curl_y2.squeeze(0).squeeze(0) * warp_str2 + drift_y
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS — steady, no arc (avoids brightness-driven PNG compression change) ──
        brightness_mod = 0.90 + bass * 0.20 + beat * 0.10
        result = sampled * brightness_mod

        # ── COLOR TEMPERATURE SHIFT — warm at start, cool at end ──
        temp_t = min(t * speed / 30.0, 1.0)
        # t=0: red*1.08, blue*0.92 (warm)  →  t=30: red*0.85, blue*1.15 (cool)
        r_mult = 1.08 - temp_t * 0.23
        b_mult = 0.92 + temp_t * 0.23
        result[:, 0:1] *= r_mult
        result[:, 2:3] *= b_mult

        # ── FOAM LAYER — high-freq noise that builds over time ──
        foam_strength = warp_t * 0.30  # 0% at start → 30% at t=30
        if foam_strength > 0.01:
            foam_seed = t * speed * 0.30
            foam_x, foam_y = _curl_noise(h, w, self.device, seed=foam_seed,
                                          base_scale=12, octaves=3)
            foam = (foam_x.squeeze(0).squeeze(0).abs() * foam_y.squeeze(0).squeeze(0).abs() * 8.0).clamp(0, 1)
            foam = foam.pow(2.0)  # Sharpen to white streaks
            # Foam appears more at wave crests (upper-y regions)
            foam_y_bias = (yy * 0.3 + 0.5).clamp(0, 1)
            foam = foam * foam_y_bias
            foam_4d = foam.unsqueeze(0).unsqueeze(0)
            foam_bloom = KF.gaussian_blur2d(foam_4d, (7, 7), (2.0, 2.0))
            result = result + foam_bloom * foam_strength * (0.7 + bass * 0.3)

        # ── CAUSTIC LIGHT — dancing highlights ──
        caustic_seed = t * speed * 0.20 + 100
        caustic_x, caustic_y = _curl_noise(h, w, self.device, seed=caustic_seed,
                                            base_scale=8, octaves=3)
        caustic = (caustic_x.squeeze(0).squeeze(0) * caustic_y.squeeze(0).squeeze(0) * 4.0 + 0.5).clamp(0, 1)
        caustic = caustic.pow(3.0)
        caustic_bloom = KF.gaussian_blur2d(caustic.unsqueeze(0).unsqueeze(0), (11, 11), (3.5, 3.5))
        caustic_intensity = 0.12 + bass * 0.10 + beat * 0.08
        result[:, 0:1] += caustic_bloom * caustic_intensity * 0.8
        result[:, 1:2] += caustic_bloom * caustic_intensity * 1.0
        result[:, 2:3] += caustic_bloom * caustic_intensity * 1.2

        # ── BOKEH GLOW ──
        bright = (sampled.max(dim=1, keepdim=True)[0] * 2.0 - 0.8).clamp(0, 1)
        bokeh_bloom = KF.gaussian_blur2d(bright, (21, 21), (7.0, 7.0))
        result[:, 0:1] += bokeh_bloom * 0.10
        result[:, 1:2] += bokeh_bloom * 0.12
        result[:, 2:3] += bokeh_bloom * 0.18

        # ── VIGNETTE ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.55) / 0.45).clamp(0, 1).pow(1.3) * 0.65)
        result = result * vignette

        # ── CINEMATIC BLOOM ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (12.0, 12.0))
        result = result + bloom * 0.15

        # ── Beat flash ──
        if beat > 0.3:
            result += bokeh_bloom.expand_as(result) * beat * 0.15
            result += caustic_bloom.expand_as(result) * beat * 0.10

        # ── TEMPORAL FEEDBACK ──
        feedback = float(params.get('feedback', 0.30))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.025, rot_strength=0.008)
            advected = KF.gaussian_blur2d(advected, (5, 5), (1.2, 1.2))
            fb_mask = (sampled.mean(dim=1, keepdim=True) * 2.0).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0)) * 0.55 + 0.08
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.3)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w
        return result.clamp(0, 1)

    def _synapses_textured(self, h, w, t, audio, params):
        """Neural synapses — purple/electric filaments, pulsing nodes, dark void."""
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']
        texture = self._load_texture(params.get('texture', ''))

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(h, 1).expand(h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, w).expand(h, w)
        radius = (xx ** 2 + yy ** 2).sqrt()

        # ── ZOOM — slow exploration ──
        zoom_dur = float(params.get('zoom_duration', 45.0))
        zoom_s = float(params.get('zoom_start', 1.0))
        zoom_e = float(params.get('zoom_end', 1.40))
        zt = min(t * speed / zoom_dur, 1.0)
        zoom = zoom_s + (zoom_e - zoom_s) * (1.0 - (1.0 - zt) ** 2.0)
        zoom += bass * 0.02 + 0.018 * math.sin(t * speed * 0.20)

        # ── ROTATION ──
        rot = t * speed * 0.04
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = xx * cos_r - yy * sin_r
        ry = xx * sin_r + yy * cos_r

        # ── PENDULUM DRIFT — floating through neural network ──
        h_off = 0.04 * math.sin(t * speed * 0.08)
        v_off = 0.03 * math.sin(t * speed * 0.06 + 1.0)

        # ── CURL NOISE — electric flicker ──
        warp_seed = t * speed * 0.10
        curl_x, curl_y = _curl_noise(h, w, self.device, seed=warp_seed,
                                      base_scale=7, octaves=3)
        warp_str = 0.06 * (0.4 + bass * 0.5)

        # ── Build UV ──
        u = rx + curl_x.squeeze(0).squeeze(0) * warp_str + h_off
        v = ry + curl_y.squeeze(0).squeeze(0) * warp_str + v_off
        u_norm = (u / (aspect * zoom)).clamp(-1, 1)
        v_norm = (v / zoom).clamp(-1, 1)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        sampled = self._sample_texture(texture, grid)

        # ── BRIGHTNESS — lift shadows, brighter base ──
        brightness_mod = 0.95 + bass * 0.35 + beat * 0.20
        result = sampled * brightness_mod

        # ── LIFT SHADOWS — prevent near-black crushing ──
        result = result + 0.03  # Floor lift (prevents pure black)

        # ── PURPLE/ELECTRIC ENHANCEMENT — stronger, wider glow ──
        blue_ch = sampled[:, 2:3]
        red_ch = sampled[:, 0:1]
        purple_mask = ((blue_ch + red_ch * 0.5 - sampled[:, 1:2] * 0.8) * 3.0).clamp(0, 1)
        # Dual-radius purple glow (like DNA green)
        purple_bloom_inner = KF.gaussian_blur2d(purple_mask, (15, 15), (5.0, 5.0))
        purple_bloom_outer = KF.gaussian_blur2d(purple_mask, (31, 31), (12.0, 12.0))
        glow_pulse = 0.8 + bass * 0.5
        result[:, 0:1] += (purple_bloom_inner * 0.18 + purple_bloom_outer * 0.08) * glow_pulse
        result[:, 2:3] += (purple_bloom_inner * 0.28 + purple_bloom_outer * 0.12) * glow_pulse

        # ── SYNAPSE FIRE — bright nodes pulse on beats ──
        bright_spots = (sampled.max(dim=1, keepdim=True)[0] * 2.0 - 0.6).clamp(0, 1)
        node_bloom = KF.gaussian_blur2d(bright_spots, (15, 15), (5.0, 5.0))
        node_pulse = 0.6 + beat * 1.2 + bass * 0.6
        result[:, 0:1] += node_bloom * 0.25 * node_pulse
        result[:, 1:2] += node_bloom * 0.08 * node_pulse
        result[:, 2:3] += node_bloom * 0.30 * node_pulse

        # ── BACKGROUND — deep blue-violet (not black) ──
        brightness = sampled.mean(dim=1, keepdim=True)
        dark_mask = (1.0 - brightness * 2.5).clamp(0, 1)
        result[:, 0:1] += dark_mask * 0.015  # Slight magenta in void
        result[:, 2:3] += dark_mask * 0.04   # Blue-violet void

        # ── STARS ──
        self._ensure_stars(h, w, count=600)
        twinkle = 0.5 + 0.5 * math.sin(t * speed * 1.5)
        star_vis = self._star_field * dark_mask * twinkle * 0.8
        result[:, 0:1] += star_vis * 0.6
        result[:, 1:2] += star_vis * 0.4
        result[:, 2:3] += star_vis * 1.0

        # ── VIGNETTE — gentler to preserve detail ──
        screen_r = (xx ** 2 + yy ** 2).sqrt().unsqueeze(0).unsqueeze(0)
        max_r = (aspect ** 2 + 1.0) ** 0.5
        vignette = (1.0 - ((screen_r / max_r - 0.70) / 0.30).clamp(0, 1).pow(1.0) * 0.40)
        result = result * vignette

        # ── CINEMATIC BLOOM — stronger for glow ──
        bloom = KF.gaussian_blur2d(result, (31, 31), (10.0, 10.0))
        result = result + bloom * 0.20

        # ── Beat flash — electric BURST ──
        if beat > 0.20:
            flash = node_bloom * beat * 0.55
            result[:, 0:1] += flash * 0.9
            result[:, 2:3] += flash * 1.0

        # ── TEMPORAL FEEDBACK ──
        feedback = float(params.get('feedback', 0.55))
        if feedback > 0 and self._prev_frame is not None and self._prev_h == h and self._prev_w == w:
            advected = _advect_frame(self._prev_frame, h, w, self.device, t,
                                     strength=0.020, rot_strength=0.008)
            advected = KF.gaussian_blur2d(advected, (3, 3), (0.8, 0.8))
            fb_mask = (brightness * 2.5).clamp(0, 1)
            fb_mask = KF.gaussian_blur2d(fb_mask, (15, 15), (5.0, 5.0)) * 0.60 + 0.05
            effective_fb = fb_mask * feedback * (1.0 - beat * 0.3)
            result = advected * effective_fb + result * (1.0 - effective_fb)

        self._prev_frame = result.clone().detach()
        self._prev_h = h
        self._prev_w = w
        return result.clamp(0, 1)

    def _cosmic_dust(self, h, w, t, audio, params):
        """Multi-layer particle cloud with parallax depth and glow."""
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        result = torch.zeros(1, 3, h, w, device=self.device)

        # Deep blue-teal base
        result[:, 0:1] += 0.02
        result[:, 1:2] += 0.03
        result[:, 2:3] += 0.08

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        # 5 depth layers with parallax
        for layer in range(5):
            layer_speed = (0.15 + layer * 0.12) * speed
            seed = t * layer_speed + layer * 50.0

            noise = _fractal_noise(h, w, 4, self.device, seed)
            # Smooth the noise for softer clouds
            noise = KF.gaussian_blur2d(noise, (5, 5), (1.5, 1.5))
            # Lower threshold → more visible particles
            thresh = 0.42 - bass * 0.06
            particles = ((noise - thresh) / (1.0 - thresh)).clamp(0, 1).pow(1.5)

            # Layer color: far=deep blue, mid=teal, near=warm
            depth = layer / 4.0
            r = 0.1 + depth * 0.6
            g = 0.25 + depth * 0.35
            b = 0.7 - depth * 0.3
            brightness = (0.5 + depth * 0.5 + beat * 0.3)

            result[:, 0:1] += particles * r * brightness
            result[:, 1:2] += particles * g * brightness
            result[:, 2:3] += particles * b * brightness

        # Central atmospheric glow (stronger)
        radius = (xx ** 2 + yy ** 2).sqrt()
        glow = (1.0 - (radius / 1.2).clamp(0, 1)).pow(1.5) * 0.25
        result[:, 0:1] += glow * 0.6
        result[:, 1:2] += glow * 0.7
        result[:, 2:3] += glow * 1.0

        # Sparse bright highlights (stars/embers)
        self._ensure_stars(h, w, count=100)
        star_vis = self._star_field * (0.3 + beat * 0.3)
        result[:, 0:1] += star_vis * 0.9
        result[:, 1:2] += star_vis * 0.7
        result[:, 2:3] += star_vis * 0.4

        return result.clamp(0, 1)

    def _dna_helix(self, h, w, t, audio, params):
        """Double helix with glowing nucleotide spheres.

        Two intertwined helical strands with SDF spheres along the path.
        Green/teal color scheme against dark background with energy crackles.
        """
        speed = float(params.get('speed', 1.0))
        bass = audio['bass']
        beat = audio['beat']

        result = torch.zeros(1, 3, h, w, device=self.device)
        result[:, 2:3] += 0.04  # Dark blue-black bg

        aspect = w / h
        yy = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-aspect, aspect, w, device=self.device).view(1, 1, 1, w).expand(1, 1, h, w)

        helix_freq = 3.0  # Turns visible
        helix_r = 0.25 + bass * 0.03
        sphere_r = 0.055 + beat * 0.01
        rotation = t * speed * 0.5

        # Strand colors: green + teal
        colors = [
            (0.3, 0.85, 0.3),   # Green strand
            (0.2, 0.6, 0.7),    # Teal strand
        ]

        for strand in range(2):
            phase_offset = strand * math.pi
            cr, cg, cb = colors[strand]
            num_spheres = 22

            for i in range(num_spheres):
                sy = -1.1 + (i / (num_spheres - 1)) * 2.2
                helix_phase = sy * helix_freq * math.pi + rotation + phase_offset
                sx = math.cos(helix_phase) * helix_r
                sz = math.sin(helix_phase)  # Depth
                depth = (sz + 1.0) * 0.5

                # SDF sphere
                dist = ((xx - sx) ** 2 + (yy - sy) ** 2).sqrt()
                r = sphere_r * (0.8 + depth * 0.4)
                sphere = (1.0 - (dist / r).clamp(0, 1)).pow(1.5)

                # Fresnel rim glow
                rim_zone = ((dist / r).clamp(0.6, 1.0) - 0.6) / 0.4
                rim = (1.0 - rim_zone).clamp(0, 1) * sphere * 0.4

                brightness = 0.4 + depth * 0.6
                result[:, 0:1] += (sphere * cr + rim * 0.2) * brightness
                result[:, 1:2] += (sphere * cg + rim * 0.8) * brightness
                result[:, 2:3] += (sphere * cb + rim * 0.3) * brightness

            # Backbone line connecting spheres
            # Render as thin gaussian tube along the helix path
            backbone = torch.zeros(1, 1, h, w, device=self.device)
            num_segments = 60
            for i in range(num_segments):
                frac = i / (num_segments - 1)
                sy = -1.1 + frac * 2.2
                helix_phase = sy * helix_freq * math.pi + rotation + phase_offset
                sx = math.cos(helix_phase) * helix_r
                dist = ((xx - sx) ** 2 + (yy - sy) ** 2).sqrt()
                segment = (1.0 - (dist / 0.012).clamp(0, 1)).clamp(0, 1)
                backbone = torch.max(backbone, segment)

            result[:, 0:1] += backbone * cr * 0.3
            result[:, 1:2] += backbone * cg * 0.3
            result[:, 2:3] += backbone * cb * 0.3

        # Cross-rungs between strands (base pairs)
        num_rungs = 11
        for i in range(num_rungs):
            frac = i / (num_rungs - 1)
            ry = -1.0 + frac * 2.0
            helix_phase = ry * helix_freq * math.pi + rotation
            x1 = math.cos(helix_phase) * helix_r
            x2 = math.cos(helix_phase + math.pi) * helix_r

            # Horizontal-ish line between the two strands
            in_y_range = ((yy - ry).abs() < 0.015).float()
            in_x_range = ((xx - min(x1, x2)) > -0.01).float() * \
                         ((xx - max(x1, x2)) < 0.01).float()
            rung = in_y_range * in_x_range * 0.25
            result[:, 0:1] += rung * 0.3
            result[:, 1:2] += rung * 0.6
            result[:, 2:3] += rung * 0.5

        # Background energy crackle on beats
        if beat > 0.4:
            energy = _fractal_noise(h, w, 3, self.device, t * 5.0)
            energy = (energy > 0.78).float() * beat * 0.12
            result[:, 1:2] += energy * 0.4
            result[:, 2:3] += energy * 0.7

        # Subtle ambient glow
        radius = (xx ** 2 + yy ** 2).sqrt()
        ambient = (1.0 - (radius / 1.8).clamp(0, 1)).pow(3) * 0.04
        result[:, 1:2] += ambient
        result[:, 2:3] += ambient * 0.5

        return result.clamp(0, 1)

    def process(self, tensor, params):
        self._frame_count += 1
        mode = params.get('mode', 'nebula')
        t = float(params.get('time', self._frame_count / 30.0))


        _, _, h, w = tensor.shape

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

        if mode == 'cosmic_dust':
            return self._cosmic_dust(h, w, t, audio, params)
        elif mode == 'dna_helix':
            return self._dna_helix(h, w, t, audio, params)
        else:
            return self._nebula(h, w, t, audio, params)
