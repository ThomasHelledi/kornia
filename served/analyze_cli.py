#!/usr/bin/env python3
"""CLI bridge for video_analyzer — called from Go's analyze-video command.

Usage:
    python3 analyze_cli.py --frames img1.jpg img2.jpg [--edges] [--device cpu]

Outputs JSON analysis to stdout. Edge maps saved alongside input frames.
"""

import argparse
import json
import os
import sys

# Add parent directory so 'effects' is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from effects.video_analyzer import (
    analyze_frames, extract_edge_guide,
    map_to_cinema_style, map_to_environment, map_to_preset, map_to_mood,
)


def main():
    parser = argparse.ArgumentParser(description='Analyze video keyframes')
    parser.add_argument('--frames', nargs='+', required=True, help='Keyframe image paths')
    parser.add_argument('--edges', action='store_true', help='Extract edge maps')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu/cuda/mps)')
    parser.add_argument('--max-dim', type=int, default=512, help='Max analysis dimension')
    args = parser.parse_args()

    analyses = analyze_frames(args.frames, device=args.device, max_dim=args.max_dim)

    # Add style mappings to each result
    for a in analyses:
        a['cinema_style'] = map_to_cinema_style(a)
        a['environment'] = map_to_environment(a)
        a['preset'] = map_to_preset(a)

    # Extract edge maps if requested
    if args.edges:
        for a in analyses:
            path = a['path']
            base, ext = os.path.splitext(path)
            edge_path = base + '_edges.png'
            extract_edge_guide(path, edge_path, device=args.device, max_dim=args.max_dim)
            a['edge_ref'] = edge_path

    # Auto-detect mood from aggregate
    mood = map_to_mood(analyses)

    output = {
        'mood': mood,
        'frame_count': len(analyses),
        'frames': analyses,
    }

    json.dump(output, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
