#!/usr/bin/env python3
"""Test the Kornia Vision Server with a real image frame."""

import struct
import subprocess
import sys
import os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PY = os.path.join(SCRIPT_DIR, "server.py")

EFFECT_PASSTHROUGH = 0x00
EFFECT_DEEPDREAM = 0x01
EFFECT_EDGE_GLOW = 0x03
EFFECT_COLOR_GRADE = 0x07


def send_frame(proc, rgba_bytes, w, h, effect_id, params_json=b'{}'):
    """Send one frame to the server, return result RGBA bytes."""
    header = struct.pack('<IIII', w, h, effect_id, len(params_json))
    proc.stdin.write(header)
    proc.stdin.write(params_json)
    proc.stdin.write(rgba_bytes)
    proc.stdin.flush()

    # Read response
    result = bytearray()
    expected = w * h * 4
    while len(result) < expected:
        chunk = proc.stdout.read(expected - len(result))
        if not chunk:
            raise RuntimeError("Server closed stdout")
        result.extend(chunk)
    return bytes(result)


def test_passthrough():
    """Test passthrough: output should equal input."""
    w, h = 64, 64
    rgba = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8).tobytes()

    proc = subprocess.Popen(
        [sys.executable, SERVER_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=SCRIPT_DIR,
    )

    # Wait for ready
    while True:
        line = proc.stderr.readline().decode()
        if "ready" in line:
            break

    result = send_frame(proc, rgba, w, h, EFFECT_PASSTHROUGH)
    assert result == rgba, "Passthrough failed: output != input"
    print("PASS: passthrough")

    proc.stdin.close()
    proc.wait()


def test_deepdream():
    """Test DeepDream on a small frame."""
    w, h = 256, 256
    # Create a colorful test pattern
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = [
                int(128 + 127 * np.sin(x * 0.05)),
                int(128 + 127 * np.sin(y * 0.05)),
                int(128 + 127 * np.sin((x + y) * 0.03)),
                255
            ]
    rgba = arr.tobytes()

    proc = subprocess.Popen(
        [sys.executable, SERVER_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=SCRIPT_DIR,
    )

    # Wait for ready
    while True:
        line = proc.stderr.readline().decode()
        sys.stderr.write(f"  server: {line}")
        if "ready" in line:
            break

    params = b'{"layer":"mixed3a","iterations":5,"octaves":2,"lr":0.01}'
    t0 = time.time()
    result = send_frame(proc, rgba, w, h, EFFECT_DEEPDREAM, params)
    dt = time.time() - t0

    assert len(result) == w * h * 4, f"Wrong output size: {len(result)}"
    assert result != rgba, "DeepDream output should differ from input"

    # Save result as raw file for inspection
    out_arr = np.frombuffer(result, dtype=np.uint8).reshape(h, w, 4)
    print(f"PASS: deepdream ({dt:.1f}s, 256x256, 5 iter, 2 octaves)")
    print(f"  Output range: R[{out_arr[:,:,0].min()}-{out_arr[:,:,0].max()}] "
          f"G[{out_arr[:,:,1].min()}-{out_arr[:,:,1].max()}] "
          f"B[{out_arr[:,:,2].min()}-{out_arr[:,:,2].max()}]")

    proc.stdin.close()
    proc.wait()


def test_edge_glow():
    """Test edge glow on a test pattern."""
    w, h = 256, 256
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    # Sharp edges: white rectangle on black
    arr[64:192, 64:192, :3] = 255
    arr[:, :, 3] = 255
    rgba = arr.tobytes()

    proc = subprocess.Popen(
        [sys.executable, SERVER_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=SCRIPT_DIR,
    )

    while True:
        line = proc.stderr.readline().decode()
        if "ready" in line:
            break

    params = b'{"glow_color":[0,200,255],"glow_radius":8}'
    t0 = time.time()
    result = send_frame(proc, rgba, w, h, EFFECT_EDGE_GLOW, params)
    dt = time.time() - t0

    assert len(result) == w * h * 4
    out = np.frombuffer(result, dtype=np.uint8).reshape(h, w, 4)
    # Edge glow should add blue/cyan around the rectangle edges
    assert out[:, :, 2].max() > 200, "Should have blue glow"
    print(f"PASS: edge_glow ({dt:.2f}s)")

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    print("=== Kornia Vision Server Tests ===")
    test_passthrough()
    test_edge_glow()
    print("\nRunning DeepDream test (may take 10-30s for model download)...")
    test_deepdream()
    print("\n=== All tests passed ===")
