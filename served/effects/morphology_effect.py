"""Morphology Effect — organic texture evolution via Kornia morphological ops.

Applies morphological gradient, opening, closing, top-hat, or black-hat to
reveal structural texture patterns. Creates organic, alien-like visuals.
"""

import torch
import kornia.morphology as KM


class MorphologyEffect:
    """Morphological operations for organic texture evolution.

    Params:
        operation: str — 'gradient' | 'opening' | 'closing' | 'top_hat' | 'black_hat' (default: 'gradient')
        kernel_size: int — structuring element size (default: 5)
        iterations: int — repeat count for stronger effect (default: 1)
        blend: float — blend with original (0=original, 1=full effect) (default: 0.7)
    """

    def __init__(self, device):
        self.device = device

    def process(self, tensor, params):
        op = params.get('operation', 'gradient')
        ks = int(params.get('kernel_size', 5))
        iterations = int(params.get('iterations', 1))
        blend = float(params.get('blend', 0.7))

        kernel = torch.ones(ks, ks, device=self.device)

        result = tensor
        for _ in range(iterations):
            if op == 'gradient':
                result = KM.gradient(result, kernel)
            elif op == 'opening':
                result = KM.opening(result, kernel)
            elif op == 'closing':
                result = KM.closing(result, kernel)
            elif op == 'top_hat':
                result = KM.top_hat(result, kernel)
            elif op == 'black_hat':
                result = KM.black_hat(result, kernel)
            elif op == 'dilate':
                result = KM.dilation(result, kernel)
            elif op == 'erode':
                result = KM.erosion(result, kernel)
            else:
                return tensor

        # Blend with original
        if blend < 1.0:
            result = tensor * (1 - blend) + result * blend

        return result.clamp(0, 1)
