"""Scene change detection — histogram-based cut detector for smart frame processing.

Compares RGB histograms between consecutive frames using chi-squared distance.
Used by DreamWave to force-process on scene changes (don't interpolate across cuts).
"""

import torch


class SceneDetector:
    """Stateful scene change detector using RGB histogram comparison.

    Usage:
        detector = SceneDetector()
        is_change, score = detector.check(frame_tensor)
    """

    def __init__(self, bins=64, threshold=0.4):
        """
        Args:
            bins: Number of histogram bins per channel (default: 64)
            threshold: Chi-squared distance threshold for scene change (default: 0.4)
        """
        self.bins = bins
        self.threshold = threshold
        self._prev_hist = None

    def _compute_histogram(self, tensor):
        """Compute normalized RGB histogram from (1, 3, H, W) tensor.

        Returns:
            (3 * bins,) flat histogram tensor
        """
        # Flatten spatial dims, keep channels
        pixels = tensor.squeeze(0)  # (3, H, W)
        hists = []
        for c in range(3):
            channel = pixels[c].flatten()  # (H*W,)
            # Quantize to bins
            quantized = (channel * (self.bins - 1)).long().clamp(0, self.bins - 1)
            hist = torch.zeros(self.bins, device=tensor.device)
            hist.scatter_add_(0, quantized, torch.ones_like(quantized, dtype=torch.float32))
            # Normalize
            hist = hist / (hist.sum() + 1e-8)
            hists.append(hist)
        return torch.cat(hists)

    def _chi_squared(self, h1, h2):
        """Chi-squared distance between two histograms."""
        denom = h1 + h2 + 1e-8
        return (((h1 - h2) ** 2) / denom).sum().item() * 0.5

    def check(self, tensor):
        """Check if frame represents a scene change.

        Args:
            tensor: (1, 3, H, W) float tensor in [0, 1]

        Returns:
            (is_scene_change: bool, change_score: float)
        """
        hist = self._compute_histogram(tensor)

        if self._prev_hist is None:
            self._prev_hist = hist
            return False, 0.0

        score = self._chi_squared(self._prev_hist, hist)
        self._prev_hist = hist

        return score > self.threshold, score
