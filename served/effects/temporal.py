"""Temporal Coherence — exponential moving average frame blending.

Smooths neural effects across consecutive frames to eliminate flickering.
Keeps a running weighted average that blends each new processed frame with
the accumulated history, producing stable, coherent video output.
"""

import torch


class TemporalBlender:
    """EMA-based temporal smoothing for frame sequences.

    momentum=0.0: no blending (raw frame-by-frame)
    momentum=0.5: balanced smoothing
    momentum=0.8: heavy smoothing (very stable, less responsive)
    """

    def __init__(self, device, momentum=0.4):
        self.device = device
        self.momentum = momentum
        self._prev = None
        self._shape = None

    def blend(self, current):
        """Blend current frame with temporal history.

        Args:
            current: (1, 3, H, W) float tensor in [0, 1]
        Returns:
            Blended (1, 3, H, W) float tensor in [0, 1]
        """
        if self.momentum <= 0 or self._prev is None or current.shape != self._shape:
            self._prev = current.detach().clone()
            self._shape = current.shape
            return current

        # EMA: result = momentum * prev + (1 - momentum) * current
        with torch.no_grad():
            blended = self.momentum * self._prev + (1.0 - self.momentum) * current
            blended = blended.clamp(0, 1)
            self._prev = blended.clone()

        return blended

    def reset(self):
        """Reset temporal buffer (e.g., on section change)."""
        self._prev = None
        self._shape = None
