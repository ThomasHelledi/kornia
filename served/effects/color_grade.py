"""Color Grade — Kornia-powered differentiable color correction.

V2: CLAHE adaptive contrast, auto white balance, named grade presets.
"""

import torch
import kornia.enhance as KE
import kornia.color as KC

from .audio_reactive import AudioReactiveEngine


# Named grade presets: {brightness, contrast, saturation, gamma, hue}
GRADE_PRESETS = {
    'warm_cinematic': {'brightness': 0.02, 'contrast': 1.15, 'saturation': 0.9, 'gamma': 0.95, 'hue': 0.05},
    'cold_blue':      {'brightness': -0.02, 'contrast': 1.1, 'saturation': 0.85, 'gamma': 1.05, 'hue': -0.15},
    'neon':           {'brightness': 0.05, 'contrast': 1.3, 'saturation': 1.6, 'gamma': 0.85, 'hue': 0.0},
    'vintage':        {'brightness': 0.03, 'contrast': 0.9, 'saturation': 0.7, 'gamma': 1.1, 'hue': 0.08},
}


class ColorGradeEffect:
    """Apply brightness, contrast, saturation, gamma, hue adjustments.

    Params:
        brightness: float — additive brightness (default: 0.0)
        contrast: float — multiplicative contrast (default: 1.0)
        saturation: float — multiplicative saturation (default: 1.0)
        gamma: float — gamma correction (default: 1.0)
        hue: float — hue shift in radians (default: 0.0)
        preset: str — named grade preset (overrides individual params)
        clahe: bool — apply CLAHE adaptive contrast (default: false)
        auto_wb: bool — apply auto white balance (default: false)
        clip_limit: float — CLAHE clip limit (default: 2.0)
    """

    def __init__(self, device):
        self.device = device
        self._audio = AudioReactiveEngine()

    def _auto_white_balance(self, tensor):
        """Gray world white balance: scale channels so mean = gray."""
        means = tensor.mean(dim=[2, 3], keepdim=True)  # (1, 3, 1, 1)
        gray = means.mean(dim=1, keepdim=True)  # Overall mean
        # Scale each channel so its mean matches the overall gray
        scale = gray / (means + 1e-6)
        return (tensor * scale).clamp(0, 1)

    def _apply_clahe(self, tensor, clip_limit):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        return KE.equalize_clahe(tensor, clip_limit=clip_limit)

    def process(self, tensor, params):
        # Audio-reactive
        beat_pulse = float(params.get('beat_pulse', 0))
        bass_energy = float(params.get('bass_energy', 0))
        audio_intensity = float(params.get('audio_intensity', 0))

        beat = self._audio.smooth('beat', beat_pulse, attack=0.4, release=0.06)
        bass = self._audio.smooth('bass', bass_energy, attack=0.3, release=0.05)
        intensity = self._audio.smooth('intensity', audio_intensity, attack=0.2, release=0.05)

        # Load preset or individual params
        preset = params.get('preset', '')
        if preset and preset in GRADE_PRESETS:
            p = GRADE_PRESETS[preset]
            brightness = p['brightness']
            contrast = p['contrast']
            saturation = p['saturation']
            gamma = p.get('gamma', 1.0)
            hue = p.get('hue', 0.0)
        else:
            brightness = float(params.get('brightness', 0))
            contrast = float(params.get('contrast', 1.0))
            saturation = float(params.get('saturation', 1.0))
            gamma = float(params.get('gamma', 1.0))
            hue = float(params.get('hue', 0.0))

        # Audio-reactive modulation
        if intensity > 0 or beat > 0:
            brightness += beat * 0.05
            contrast += bass * 0.15
            saturation += intensity * 0.2

        # Auto white balance (before grading)
        if params.get('auto_wb', False):
            tensor = self._auto_white_balance(tensor)

        # CLAHE adaptive contrast
        if params.get('clahe', False):
            clip_limit = float(params.get('clip_limit', 2.0))
            tensor = self._apply_clahe(tensor, clip_limit)

        # Standard adjustments
        if brightness != 0:
            tensor = KE.adjust_brightness(tensor, brightness)
        if contrast != 1.0:
            tensor = KE.adjust_contrast(tensor, contrast)
        if saturation != 1.0:
            tensor = KE.adjust_saturation(tensor, saturation)
        if gamma != 1.0:
            tensor = KE.adjust_gamma(tensor, gamma)
        if hue != 0.0:
            tensor = KE.adjust_hue(tensor, hue)

        return tensor.clamp(0, 1)
