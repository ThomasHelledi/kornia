"""Cinema Styles — named visual presets for DreamWave pipeline.

Each style configures DreamWave's full parameter set to produce a distinct
cinematic look. Styles are merged on top of preset defaults, with explicit
params taking highest priority.

Usage:
    from .cinema_styles import get_cinema_style, CINEMA_STYLES
    style = get_cinema_style('fantasy')  # returns full params dict
"""

# Dream layer presets map to InceptionV3 layer selections:
#   'soft'     — mixed3b: early-mid, soft flowing patterns
#   'abstract' — mixed3a: early features, edges/textures
#   'faces'    — mixed4c: mid features, face-like patterns
#   'fractal'  — mixed5a: deep features, complex fractal forms
#   'eyes'     — mixed5b: very deep, eye-like hallucinations

CINEMA_STYLES = {
    'fantasy': {
        'description': 'Warm golden tones, soft glow, ethereal depth',
        'edge_color': [255, 200, 100],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.3,
        'pre_boost': 0.12,
        'edge_blend': 0.6,
        'depth_strength': 3.5,
        'chain_mix': 0.80,
        'dream_lr': 0.007,
        'dream_iter': 10,
    },
    'noir': {
        'description': 'High contrast black & white, hard edges, dramatic shadows',
        'edge_color': [180, 180, 180],
        'dream_layer': 'fractal',
        'focus_mode': 'vertical',
        'saturation': 0.3,
        'pre_boost': 0.05,
        'edge_blend': 0.8,
        'depth_strength': 4.0,
        'chain_mix': 0.90,
        'dream_lr': 0.010,
        'dream_iter': 15,
    },
    'anime': {
        'description': 'Vivid flat colors, thick edges, cel-shaded look',
        'edge_color': [60, 200, 255],
        'dream_layer': 'faces',
        'focus_mode': 'none',
        'saturation': 1.5,
        'pre_boost': 0.10,
        'edge_blend': 0.7,
        'depth_strength': 2.0,
        'chain_mix': 0.85,
        'dream_lr': 0.006,
        'dream_iter': 8,
    },
    'nature': {
        'description': 'Earthy greens, gentle depth, organic warmth',
        'edge_color': [80, 180, 80],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.2,
        'pre_boost': 0.08,
        'edge_blend': 0.4,
        'depth_strength': 3.0,
        'chain_mix': 0.75,
        'dream_lr': 0.006,
        'dream_iter': 10,
    },
    'cyberpunk': {
        'description': 'Neon purple/teal, glitch hints, high contrast digital',
        'edge_color': [200, 50, 255],
        'dream_layer': 'eyes',
        'focus_mode': 'none',
        'saturation': 1.4,
        'pre_boost': 0.14,
        'edge_blend': 0.75,
        'depth_strength': 3.5,
        'chain_mix': 0.88,
        'dream_lr': 0.011,
        'dream_iter': 16,
    },
    'cinematic': {
        'description': 'Film-grade color, subtle dream, professional depth',
        'edge_color': [140, 160, 200],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.1,
        'pre_boost': 0.07,
        'edge_blend': 0.35,
        'depth_strength': 2.5,
        'chain_mix': 0.75,
        'dream_lr': 0.005,
        'dream_iter': 8,
    },
    'gorillaz': {
        'description': 'Flat comic art, blue edges, cartoon universe',
        'edge_color': [80, 160, 255],
        'dream_layer': 'faces',
        'focus_mode': 'radial',
        'saturation': 1.25,
        'pre_boost': 0.10,
        'edge_blend': 0.65,
        'depth_strength': 2.5,
        'chain_mix': 0.82,
        'dream_lr': 0.007,
        'dream_iter': 10,
    },
    'psychedelic': {
        'description': 'Full hallucination, maximum color, deep neural patterns',
        'edge_color': [255, 100, 200],
        'dream_layer': 'eyes',
        'focus_mode': 'none',
        'saturation': 1.6,
        'pre_boost': 0.18,
        'edge_blend': 0.85,
        'depth_strength': 5.0,
        'chain_mix': 0.92,
        'dream_lr': 0.014,
        'dream_iter': 22,
    },
    'electronic': {
        'description': 'Glitch+warp(wave), teal edges, abstract dream, responsive',
        'edge_color': [80, 220, 220],
        'dream_layer': 'abstract',
        'focus_mode': 'none',
        'saturation': 1.35,
        'pre_boost': 0.12,
        'edge_blend': 0.65,
        'depth_strength': 3.0,
        'chain_mix': 0.85,
        'dream_lr': 0.009,
        'dream_iter': 14,
        'glitch': True,
        'warp': True,
        'warp_mode': 'wave',
        'temporal_momentum': 0.2,
    },
    'industrial': {
        'description': 'Glitch+warp(vortex), amber edges, fractal dream, raw',
        'edge_color': [220, 160, 60],
        'dream_layer': 'fractal',
        'focus_mode': 'vertical',
        'saturation': 0.9,
        'pre_boost': 0.10,
        'edge_blend': 0.7,
        'depth_strength': 4.5,
        'chain_mix': 0.88,
        'dream_lr': 0.012,
        'dream_iter': 18,
        'glitch': True,
        'warp': True,
        'warp_mode': 'vortex',
        'temporal_momentum': 0.15,
    },
    'vaporwave': {
        'description': 'Glitch+warp(fisheye), pink edges, soft dream, dreamy',
        'edge_color': [255, 130, 200],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.5,
        'pre_boost': 0.14,
        'edge_blend': 0.55,
        'depth_strength': 3.5,
        'chain_mix': 0.82,
        'dream_lr': 0.007,
        'dream_iter': 12,
        'glitch': True,
        'warp': True,
        'warp_mode': 'fisheye',
        'temporal_momentum': 0.4,
        'chromatic_aberration': 1.0,   # VHS-style RGB fringing
    },
    'film': {
        'description': 'Cinema-grade: grain, vignette, halation, split-tone, no dream',
        'edge_color': [160, 140, 120],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.05,
        'pre_boost': 0.04,
        'edge_blend': 0.25,
        'depth_strength': 2.0,
        'chain_mix': 0.70,
        'dream_lr': 0,
        'dream_iter': 0,
        'film_grain': 0.025,
        'vignette': 0.35,
        'halation': 0.15,
        'split_tone': True,
        'temporal_momentum': 0.3,
    },
    'film-noir': {
        'description': 'Classic film noir: grain, hard vignette, desaturated, dramatic',
        'edge_color': [180, 170, 160],
        'dream_layer': 'soft',
        'focus_mode': 'vertical',
        'saturation': 0.35,
        'pre_boost': 0.03,
        'edge_blend': 0.6,
        'depth_strength': 3.5,
        'chain_mix': 0.80,
        'dream_lr': 0,
        'dream_iter': 0,
        'film_grain': 0.035,
        'vignette': 0.50,
        'halation': 0.10,
        'split_tone': True,
        'temporal_momentum': 0.25,
        'lift': [-0.02, -0.01, 0.03],
    },
    'music-video': {
        'description': 'Modern music video look: punchy, stylized, subtle grain',
        'edge_color': [120, 160, 220],
        'dream_layer': 'soft',
        'focus_mode': 'radial',
        'saturation': 1.2,
        'pre_boost': 0.08,
        'edge_blend': 0.35,
        'depth_strength': 2.5,
        'chain_mix': 0.75,
        'dream_lr': 0,
        'dream_iter': 0,
        'film_grain': 0.018,
        'vignette': 0.25,
        'halation': 0.20,
        'split_tone': True,
        'gain': [0.05, 0.0, -0.03],
        'temporal_momentum': 0.3,
    },
    'film-clean': {
        'description': 'Clean cinema finishing — NO neural effects, just film post on real footage',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,           # No edge glow
        'depth_strength': 0,       # No depth blur
        'focus_mode': 'none',      # No radial blur
        'pre_boost': 0,            # No brightness manipulation
        'saturation': 1.0,         # Natural saturation
        'chain_mix': 0.95,         # High mix since we're only doing film post
        'film_grain': 0.020,       # Subtle grain
        'vignette': 0.30,          # Cinema vignette
        'halation': 0.12,          # Gentle halation on highlights
        'split_tone': True,        # Cool shadows + warm highlights
        'skip_color_grade': True,  # Skip neural color pre-grade
        'skip_clahe': True,        # Skip CLAHE contrast
        'temporal_momentum': 0.15,
    },
    'film-anamorphic': {
        'description': 'Anamorphic cinema: 2.39:1 letterbox, horizontal streaks, S-curve, warm halation',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0,
        'saturation': 1.05,        # Slightly warm
        'chain_mix': 0.95,
        'film_grain': 0.022,       # Slightly more grain for cinema texture
        'vignette': 0.35,          # Stronger vignette for anamorphic feel
        'halation': 0.18,          # More halation for anamorphic glow
        'anamorphic': True,        # Horizontal streak halation
        'split_tone': True,
        's_curve': 0.5,            # Cinematic S-curve contrast
        'letterbox': True,         # 2.39:1 crop
        'letterbox_ratio': 2.39,
        'skip_color_grade': True,
        'skip_clahe': True,
        'temporal_momentum': 0.15,
        'gain': [0.03, 0.01, -0.02],  # Slight warm push
    },
    # ─── Video-Input Optimized Styles (real footage, no neural) ─────────
    'concert': {
        'description': 'Live performance: warm punch, dynamic contrast, stage glow',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0.06,            # Lift dark stage moments
        'saturation': 1.15,           # Punch up stage colors
        'chain_mix': 0.95,
        'film_grain': 0.015,          # Light grain
        'vignette': 0.20,             # Subtle focus to center stage
        'halation': 0.22,             # Stage lights bloom
        'split_tone': False,
        'skip_color_grade': True,
        'temporal_momentum': 0.2,
        'shadow_recovery': 0.6,       # Lift crushed stage blacks
        'gain': [0.04, 0.01, -0.02],  # Warm push
    },
    'bleach-bypass': {
        'description': 'Bleach bypass: desaturated, high contrast, gritty texture',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0,
        'saturation': 0.55,           # Heavy desaturation
        'chain_mix': 0.95,
        'film_grain': 0.040,          # Heavy grain
        'vignette': 0.40,
        'halation': 0.08,
        's_curve': 0.7,               # Strong contrast
        'split_tone': True,
        'skip_color_grade': True,
        'temporal_momentum': 0.2,
        'lift': [-0.03, -0.02, 0.01],  # Cold shadows
    },
    'teal-orange': {
        'description': 'Hollywood teal/orange split: warm skin, cool shadows',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0,
        'saturation': 1.2,
        'chain_mix': 0.95,
        'film_grain': 0.018,
        'vignette': 0.28,
        'halation': 0.15,
        'split_tone': True,
        's_curve': 0.4,
        'skip_color_grade': True,
        'temporal_momentum': 0.2,
        'lift': [-0.02, 0.0, 0.06],    # Teal shadows (blue lift)
        'gain': [0.06, 0.02, -0.04],   # Orange highlights
    },
    'vintage-8mm': {
        'description': 'Super 8mm: heavy grain, warm fade, soft blur, nostalgic',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 1.5,         # Slight soft focus
        'focus_mode': 'radial',
        'pre_boost': 0.03,
        'saturation': 0.85,            # Slightly faded
        'chain_mix': 0.95,
        'film_grain': 0.055,           # Heavy 8mm grain
        'vignette': 0.55,              # Heavy vignette (8mm lens)
        'halation': 0.25,              # Strong light bleed
        'split_tone': True,
        'skip_color_grade': True,
        'temporal_momentum': 0.35,     # More temporal smoothing (dreamy)
        'chromatic_aberration': 1.5,   # Lens fringing (8mm optics)
        'lift': [0.04, 0.02, -0.01],   # Warm faded shadows
        'gain': [0.03, 0.0, -0.03],    # Warm highlights
    },
    'neon-night': {
        'description': 'Neon night: high contrast, saturated highlights, dark shadows',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0,
        'saturation': 1.4,             # Punchy neon colors
        'chain_mix': 0.95,
        'film_grain': 0.020,
        'vignette': 0.45,              # Dark edges
        'halation': 0.30,              # Neon glow bloom
        'split_tone': False,
        's_curve': 0.6,                # High contrast
        'skip_color_grade': True,
        'temporal_momentum': 0.2,
        'highlight_recovery': 0.5,     # Protect neon from clipping
        'lift': [-0.04, -0.02, 0.02],  # Crushed blue shadows
        'gain': [0.02, 0.04, 0.05],    # Cool neon highlights
    },
    'dreamy': {
        'description': 'Soft music video: warm halation, gentle grain, breath zoom, ethereal',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 1.5,
        'focus_mode': 'radial',
        'pre_boost': 0.04,
        'saturation': 1.1,
        'chain_mix': 0.92,
        'film_grain': 0.015,
        'vignette': 0.30,
        'halation': 0.28,              # Strong dreamy halation
        'split_tone': True,
        'skip_color_grade': True,
        'temporal_momentum': 0.4,      # Heavy smoothing for dream feel
        'breath': 0.012,               # Subtle breathing zoom
        'shadow_recovery': 0.4,
        'gain': [0.02, 0.01, -0.01],   # Slight warmth
    },
    'documentary': {
        'description': 'Clean documentary: neutral grade, subtle grain, no dream',
        'dream_lr': 0,
        'dream_iter': 0,
        'edge_blend': 0,
        'depth_strength': 0,
        'focus_mode': 'none',
        'pre_boost': 0,
        'saturation': 1.0,             # Natural
        'chain_mix': 0.95,
        'film_grain': 0.012,           # Very subtle grain
        'vignette': 0.15,              # Minimal vignette
        'halation': 0.08,
        'split_tone': False,
        'skip_color_grade': True,
        'skip_clahe': True,
        'temporal_momentum': 0.2,
        'shadow_recovery': 0.3,        # Preserve shadow detail
        'highlight_recovery': 0.3,     # Preserve highlight detail
    },
    # ─── Architecture / Geometry Styles (video-input, structural neural) ─
    'architecture': {
        'description': 'Enhance architectural geometry: edges/textures amplified, person protected',
        'edge_color': [180, 200, 220],     # Cool architectural white
        'dream_layer': 'abstract',          # mixed3a: edges + textures ONLY (no faces)
        'focus_mode': 'none',
        'saturation': 1.1,
        'pre_boost': 0.05,
        'edge_blend': 0.45,                # Moderate edge glow on structure
        'depth_strength': 1.5,             # Subtle depth separation
        'chain_mix': 0.70,                 # Conservative mix — preserve source
        'dream_lr': 0.006,                 # Low LR for controlled amplification
        'dream_iter': 6,                   # Few iterations — enhance, don't hallucinate
        'film_grain': 0.015,
        'vignette': 0.20,
        'halation': 0.12,                  # Gentle architectural light bloom
        'split_tone': True,
        'skip_color_grade': True,
        'temporal_momentum': 0.25,
        'shadow_recovery': 0.4,
        'highlight_recovery': 0.3,
        'person_protect': 1.0,             # Keep performer clean, dream only background
        'edge_aware_dream': 0.6,           # Reduce dream on detailed objects
    },
    'geometric': {
        'description': 'Maximum geometric enhancement: strong edge neural + person protected',
        'edge_color': [200, 220, 255],     # Bright white-blue for clean lines
        'dream_layer': 'abstract',          # mixed3a: pure edge/texture amplification
        'focus_mode': 'none',
        'saturation': 1.15,
        'pre_boost': 0.08,
        'edge_blend': 0.65,                # Strong edge glow to highlight lines
        'depth_strength': 2.0,
        'chain_mix': 0.78,
        'dream_lr': 0.008,                 # Moderate LR
        'dream_iter': 10,                  # More iterations for visible structural dream
        'film_grain': 0.018,
        'vignette': 0.25,
        'halation': 0.18,
        'split_tone': True,
        'temporal_momentum': 0.3,
        'shadow_recovery': 0.3,
        'person_protect': 1.0,             # Full person protection
        'edge_aware_dream': 0.5,           # Reduce dream on detailed regions
    },
}


def get_cinema_style(name):
    """Get a cinema style by name.

    Args:
        name: Style name (fantasy, noir, anime, nature, cyberpunk,
              cinematic, gorillaz, psychedelic)

    Returns:
        dict of DreamWave params, or None if style not found
    """
    return CINEMA_STYLES.get(name)


def list_cinema_styles():
    """Return list of available cinema style names."""
    return list(CINEMA_STYLES.keys())
