#!/usr/bin/env python3
"""Generate the placeholder app icon used by the macOS bundle.

Draws the same clipboard glyph as the menu-bar icon, renders it at the sizes
``iconutil`` wants and (on macOS) compiles ``packaging/ClipMind.icns``. The
result is a *placeholder*: replace ``packaging/ClipMind.icns`` with real artwork
when you have it.

Usage::

    python scripts/make_icon.py            # writes packaging/ClipMind.icns (macOS)
    python scripts/make_icon.py --png-only  # writes packaging/ClipMind-icon.png only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
ICONSET = PACKAGING / "ClipMind.iconset"

#: Background colour of the rounded tile (matches the "link" accent used in-app).
BACKGROUND = (37, 99, 235, 255)
FOREGROUND = (255, 255, 255, 255)

#: Sizes required by ``iconutil`` (base name, pixel size).
ICNS_ENTRIES = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]


def draw(size: int):
    """Render the placeholder icon at ``size`` x ``size`` pixels.

    Args:
        size: Edge length in pixels.

    Returns:
        A :class:`PIL.Image.Image` in RGBA mode.

    Raises:
        SystemExit: If Pillow is not installed.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - dev-only helper
        print(
            "error: Pillow is required (pip install pillow)", file=sys.stderr
        )
        raise SystemExit(1)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_ctx = ImageDraw.Draw(img)

    pad = size * 0.07
    radius = size * 0.24
    draw_ctx.rounded_rectangle(
        [pad, pad, size - pad, size - pad], radius=radius, fill=BACKGROUND
    )

    # Clipboard body.
    body_w, body_h = size * 0.46, size * 0.58
    body_x = (size - body_w) / 2
    body_y = size * 0.24
    draw_ctx.rounded_rectangle(
        [body_x, body_y, body_x + body_w, body_y + body_h],
        radius=size * 0.07,
        outline=FOREGROUND,
        width=max(1, int(size * 0.045)),
    )
    # Clip at the top of the body.
    clip_w, clip_h = size * 0.24, size * 0.11
    clip_x = (size - clip_w) / 2
    clip_y = body_y - clip_h * 0.45
    draw_ctx.rounded_rectangle(
        [clip_x, clip_y, clip_x + clip_w, clip_y + clip_h],
        radius=size * 0.04,
        fill=BACKGROUND,
        outline=FOREGROUND,
        width=max(1, int(size * 0.045)),
    )
    # Two text lines.
    line_x0 = body_x + body_w * 0.2
    line_x1 = body_x + body_w * 0.8
    for index in range(2):
        y = body_y + body_h * (0.42 + index * 0.22)
        draw_ctx.line(
            [line_x0, y, line_x1, y], fill=FOREGROUND, width=max(1, int(size * 0.045))
        )
    return img


def main(argv: list[str] | None = None) -> int:
    """Render the PNG(s) and, on macOS, compile the ``.icns``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png-only", action="store_true", help="Skip the iconutil step."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing .icns."
    )
    args = parser.parse_args(argv)

    PACKAGING.mkdir(parents=True, exist_ok=True)
    png_path = PACKAGING / "ClipMind-icon.png"
    icns_path = PACKAGING / "ClipMind.icns"

    if icns_path.exists() and not args.force and not args.png_only:
        print(f"{icns_path} already exists (use --force to regenerate)")
        return 0

    draw(1024).save(png_path)
    print(f"wrote {png_path}")

    if args.png_only:
        return 0

    if sys.platform != "darwin":
        print("note: .icns generation needs macOS (iconutil); PNG written only")
        return 0

    if ICONSET.exists():
        shutil.rmtree(ICONSET)
    ICONSET.mkdir(parents=True)
    for name, size in ICNS_ENTRIES:
        draw(size).save(ICONSET / name)

    result = subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET), "-o", str(icns_path)],
        capture_output=True,
        text=True,
    )
    shutil.rmtree(ICONSET, ignore_errors=True)
    if result.returncode != 0:
        print(f"error: iconutil failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"wrote {icns_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
