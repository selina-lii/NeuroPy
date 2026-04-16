#!/usr/bin/env python3
"""
Generate unicorn avatar PNG images using go-unicornify.

Usage:
    python scripts/generate_unicorn_avatars.py --n 50

Outputs PNGs to data/images/random/unused/ with random UUID filenames.
Requires the go-unicornify binary to be built in ~/Documents/go-unicornify/.
Falls back to simple colored circle avatars if the binary is not found.
"""
import argparse
import os
import subprocess
import random
import uuid
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.path.join(_ROOT, 'data', 'images', 'random', 'unused')
_GO_UNICORNIFY_DIR = os.path.expanduser('~/Documents/go-unicornify')


def _find_go_binary():
    """Find the unicornify go binary."""
    # Common names for the compiled binary
    for name in ('unicornify', 'unicornify-png', 'main'):
        candidate = os.path.join(_GO_UNICORNIFY_DIR, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _generate_with_go(binary: str, out_path: str, size: int = 128) -> bool:
    """Generate one avatar using the go binary. Returns True on success."""
    seed = random.randint(0, 2**31 - 1)
    try:
        result = subprocess.run(
            [binary, '-seed', str(seed), '-size', str(size), '-out', out_path],
            capture_output=True, timeout=10
        )
        return result.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def _generate_fallback(out_path: str, size: int = 128) -> bool:
    """Generate a simple colored circle avatar as fallback."""
    if not _HAS_PIL:
        print("[fallback] PIL not available -- cannot generate images without go binary.")
        return False
    colors = [
        '#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#9b5de5',
        '#f15bb5', '#00bbf9', '#00f5d4', '#fee440', '#fb5607',
    ]
    color = random.choice(colors)
    img = Image.new('RGB', (size, size), '#1e1e2e')
    draw = ImageDraw.Draw(img)
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    # Add a small inner highlight
    inner = size // 4
    draw.ellipse([inner, inner, inner + size // 4, inner + size // 4],
                 fill='white', outline=None)
    img.save(out_path, 'PNG')
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate unicorn avatar PNGs')
    parser.add_argument('--n', type=int, default=20, help='Number of avatars to generate')
    parser.add_argument('--out', default=_DEFAULT_OUT, help='Output directory')
    parser.add_argument('--size', type=int, default=128, help='Image size in pixels')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    binary = _find_go_binary()
    if binary:
        print(f"[unicornify] Using go binary: {binary}")
    else:
        print("[unicornify] Go binary not found -- using fallback colored circles.")

    success = 0
    for i in range(args.n):
        img_id = str(uuid.uuid4())
        out_path = os.path.join(args.out, f'{img_id}.png')
        if binary:
            ok = _generate_with_go(binary, out_path, args.size)
            if not ok:
                ok = _generate_fallback(out_path, args.size)
        else:
            ok = _generate_fallback(out_path, args.size)
        if ok:
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{args.n}...")
        else:
            print(f"  Failed to generate image {i + 1}")

    print(f"Done: {success}/{args.n} images in {args.out}")


if __name__ == '__main__':
    main()
