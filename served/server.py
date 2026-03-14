#!/usr/bin/env python3
"""Kornia Vision Server — persistent frame processor for atlas-synth.

Binary protocol over stdin/stdout:

REQUEST (Go → Python):
  [uint32 width] [uint32 height] [uint32 effect_id] [uint32 params_len]
  [params_len bytes JSON] [width*height*4 bytes RGBA]

RESPONSE (Python → Go):
  [width*height*4 bytes RGBA]

Logs and progress go to stderr (visible in terminal).
"""

import sys
import struct
import torch

from effects.registry import EffectRegistry


def read_exact(stream, n):
    """Read exactly n bytes from stream, or return None at EOF."""
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def main():
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sys.stderr.write(f"kornia-server: device={device}\n")
    sys.stderr.flush()

    registry = EffectRegistry(device)

    # Signal ready — Go client waits for this
    sys.stderr.write("kornia-server: ready\n")
    sys.stderr.flush()

    frame_count = 0

    while True:
        # Read header: 16 bytes
        header = read_exact(sys.stdin.buffer, 16)
        if header is None:
            break

        w, h, effect_id, params_len = struct.unpack('<IIII', header)

        # Read params JSON
        params_json = b'{}'
        if params_len > 0:
            params_json = read_exact(sys.stdin.buffer, params_len)
            if params_json is None:
                break

        # Read RGBA frame
        frame_size = w * h * 4
        rgba = read_exact(sys.stdin.buffer, frame_size)
        if rgba is None:
            break

        # Process
        result = registry.process(effect_id, rgba, w, h, params_json)

        # Write result
        sys.stdout.buffer.write(result)
        sys.stdout.buffer.flush()

        frame_count += 1
        if frame_count % 10 == 0:
            sys.stderr.write(f"kornia-server: processed {frame_count} frames\n")
            sys.stderr.flush()

    sys.stderr.write(f"kornia-server: done ({frame_count} frames total)\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
