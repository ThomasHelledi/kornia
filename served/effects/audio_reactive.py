"""AudioReactiveEngine — shared audio-reactive utilities for all effects.

Provides exponential smoothing, band mapping, beat detection, and onset
strength. All effects import this module for music-aware modulation.
"""

import math


class AudioReactiveEngine:
    """Stateful audio-reactive processor shared across effects.

    Maintains per-parameter smoothing state and beat detection state.
    Create one instance per effect for independent smoothing.
    """

    def __init__(self):
        self._smoothed = {}  # name → smoothed value
        self._prev_flux = 0.0
        self._prev_onset = 0.0
        self._beat_phase = 0.0

    def smooth(self, name, value, attack=0.3, release=0.05):
        """Exponential smoothing with asymmetric attack/release.

        Args:
            name: Parameter identifier for state tracking
            value: Raw input value
            attack: Rise coefficient (0-1, higher = faster attack)
            release: Fall coefficient (0-1, higher = faster decay)

        Returns:
            Smoothed value
        """
        prev = self._smoothed.get(name, value)
        if value >= prev:
            coeff = attack  # Fast rise
        else:
            coeff = release  # Slow fade
        smoothed = prev + coeff * (value - prev)
        self._smoothed[name] = smoothed
        return smoothed

    @staticmethod
    def map_band(value, curve='linear', exponent=2.0):
        """Map a band energy value through a response curve.

        Args:
            value: Input energy (0-1)
            curve: 'linear', 'exponential', or 'sigmoid'
            exponent: Power for exponential curve (default 2.0)

        Returns:
            Mapped value (0-1)
        """
        value = max(0.0, min(1.0, value))
        if curve == 'exponential':
            return value ** exponent
        elif curve == 'sigmoid':
            # Centered sigmoid: steep in middle, flat at extremes
            x = (value - 0.5) * 10.0
            return 1.0 / (1.0 + math.exp(-x))
        return value  # linear

    def detect_beat(self, flux, threshold=0.15):
        """Detect beat from spectral flux rising edge.

        Args:
            flux: Current spectral flux value
            threshold: Minimum flux increase to count as beat

        Returns:
            (is_beat, beat_strength) — bool and 0-1 strength
        """
        delta = flux - self._prev_flux
        self._prev_flux = flux
        is_beat = delta > threshold
        strength = max(0.0, min(1.0, delta / max(threshold * 3, 0.01)))
        return is_beat, strength

    def detect_downbeat(self, flux, bpm, time, threshold=0.2):
        """Detect downbeat using BPM phase alignment.

        Args:
            flux: Current spectral flux
            bpm: Beats per minute
            time: Current time in seconds
            threshold: Flux threshold for downbeat confirmation

        Returns:
            (is_downbeat, phase) — bool and current beat phase (0-1)
        """
        if bpm <= 0:
            return False, 0.0
        beat_duration = 60.0 / bpm
        phase = (time % beat_duration) / beat_duration
        # Downbeat = near phase 0 + strong flux
        near_downbeat = phase < 0.1 or phase > 0.9
        is_beat, strength = self.detect_beat(flux, threshold)
        return near_downbeat and is_beat, phase

    def onset_strength(self, flux):
        """Compute onset strength from spectral flux derivative.

        Returns positive flux change (onset = rising flux).
        """
        delta = max(0.0, flux - self._prev_onset)
        self._prev_onset = flux
        return delta


def get_6bands(params):
    """Extract 6-band breakdown from params dict.

    Returns:
        dict with sub_bass, bass, low_mid, mid, high_mid, treble (all 0-1)
    """
    return {
        'sub_bass': float(params.get('sub_bass', 0)),
        'bass': float(params.get('bass', 0)),
        'low_mid': float(params.get('low_mid', 0)),
        'mid': float(params.get('mid', 0)),
        'high_mid': float(params.get('high_mid', 0)),
        'treble': float(params.get('treble', 0)),
    }


def get_spectral(params):
    """Extract spectral features from params dict.

    Returns:
        dict with centroid, rolloff, flux, flatness (all 0-1)
    """
    return {
        'centroid': float(params.get('spectral_centroid', 0)),
        'rolloff': float(params.get('spectral_rolloff', 0)),
        'flux': float(params.get('spectral_flux', 0)),
        'flatness': float(params.get('spectral_flatness', 0)),
    }
