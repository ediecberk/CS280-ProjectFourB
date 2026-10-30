"""Click four corners on the fridge and microwave. No upload needed.

Run from the project root:

    source .venv/bin/activate
    python code/pick_corners.py

Click TL, TR, BR, BL for each photo (window opens twice).
Press Enter after the four clicks. Coordinates are saved and used by main.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code" / "rect_quads.json"
MAX_SIDE = 1200


def load(path):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    scale = MAX_SIDE / max(width, height)
    if scale < 1:
        image = image.resize(
            (int(round(width * scale)), int(round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(image)


def pick(path, title):
    image = load(path)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8 * height / width))
    ax.imshow(image)
    ax.set_title(f"{title}\nClick TL, TR, BR, BL then press Enter")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    print(f"\n{title}: click top-left, top-right, bottom-right, bottom-left, then Enter")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    if len(pts) != 4:
        raise SystemExit(f"Need 4 clicks for {title}, got {len(pts)}")
    pts = np.asarray(pts, dtype=np.float64)
    frac = pts.copy()
    frac[:, 0] /= width - 1
    frac[:, 1] /= height - 1
    print("pixels:", np.round(pts, 1))
    print("fractions:", np.round(frac, 4))
    return frac.tolist()


def find_image(stem):
    folder = ROOT / "images" / "rectify"
    for ext in (".jpg", ".jpeg", ".png", ".heic", ".HEIC"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def main():
    data = {
        "fridge": pick(find_image("fridge"), "Fridge front"),
        "microwave": pick(find_image("microwave"), "Microwave glass"),
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nSaved {OUT}")
    print("Next: python code/main.py --only rectify")


if __name__ == "__main__":
    main()
