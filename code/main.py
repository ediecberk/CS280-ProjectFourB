import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import match_images  # noqa: E402
from mosaic import blend_layers, output_frame, ransac_homography, render_warp  # noqa: E402

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    register_heif_opener = None


SCENES = ("courtyard", "livingroom", "laundry")
SIDES = ("left", "center", "right")


def resolve_image(folder, stem):
    for ext in (".jpg", ".jpeg", ".png", ".heic", ".HEIC"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(folder / stem)


def load_image(path, max_side):
    if path.suffix.lower() == ".heic" and register_heif_opener is None:
        raise RuntimeError("Install pillow-heif to read the HEIC photos")
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    scale = max_side / max(width, height)
    if scale < 1:
        image = image.resize(
            (int(round(width * scale)), int(round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(image).astype(np.float32) / 255.0


def to_uint8(image):
    return np.clip(np.round(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def save_image(path, image, max_side=1800, quality=86):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pic = Image.fromarray(to_uint8(image))
    width, height = pic.size
    scale = max_side / max(width, height)
    if scale < 1:
        pic = pic.resize(
            (int(round(width * scale)), int(round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    pic.save(path, quality=quality, optimize=True)


def dedupe_matches(src, dst, tol=1.5):
    keep = []
    for i, point in enumerate(src):
        if any(np.linalg.norm(point - src[j]) < tol for j in keep):
            continue
        keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return src[keep], dst[keep]


def fit_pair(src_image, dst_image):
    # H maps src onto dst. 0.6 is the Figure 6b cutoff; 0.8 is the fallback.
    src = dst = None
    for ratio in (0.6, 0.8):
        src, dst = match_images(src_image, dst_image, ratio=ratio)
        if len(src) >= 16:
            break
    if src is None or len(src) < 4:
        raise RuntimeError("not enough feature matches")
    src, dst = dedupe_matches(src, dst)
    H, inliers, err = ransac_homography(src, dst, threshold=4.0, iterations=2500, seed=0)
    if int(inliers.sum()) < 15:
        H, inliers, err = ransac_homography(src, dst, threshold=8.0, iterations=2500, seed=1)
    return H, src, dst, inliers, err


def mosaic_scene(images, homographies):
    frames = []
    for image, H in zip(images, homographies):
        shape, offset = output_frame(image, H)
        frames.append((shape, offset))
    min_x = min(offset[0] for _, offset in frames)
    min_y = min(offset[1] for _, offset in frames)
    max_x = max(offset[0] + shape[1] - 1 for shape, offset in frames)
    max_y = max(offset[1] + shape[0] - 1 for shape, offset in frames)
    out_shape = (int(max_y - min_y + 1), int(max_x - min_x + 1))
    shared = (float(min_x), float(min_y))
    layers = []
    for image, H in zip(images, homographies):
        warped, weight, _ = render_warp(
            image, H, out_shape=out_shape, offset=shared, method="bilinear"
        )
        layers.append((warped, weight))
    return blend_layers(layers)


def stitch_scene(name, max_side, out_dir):
    folder = ROOT / "images" / name
    images = [load_image(resolve_image(folder, side), max_side) for side in SIDES]
    print(f"[{name}] matching")
    left_H, left_src, left_dst, left_in, left_err = fit_pair(images[0], images[1])
    right_H, right_src, right_dst, right_in, right_err = fit_pair(images[2], images[1])
    print(
        f"[{name}] inliers left={int(left_in.sum())} ({left_err[left_in].mean():.2f} px) "
        f"right={int(right_in.sum())} ({right_err[right_in].mean():.2f} px)"
    )
    print(f"[{name}] warping")
    t0 = time.perf_counter()
    mosaic = mosaic_scene(images, [left_H, np.eye(3), right_H])
    save_image(out_dir / f"{name}_mosaic.jpg", mosaic)
    print(f"[{name}] wrote {name}_mosaic.jpg in {time.perf_counter() - t0:.1f} s")
    return {
        "images": images,
        "mosaic": mosaic,
        "pairs": [
            ("Left and center", images[0], images[1], left_src[left_in], left_dst[left_in]),
            ("Right and center", images[2], images[1], right_src[right_in], right_dst[right_in]),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Stitch photos with automatic feature matching.")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = {}
    for name in SCENES:
        scenes[name] = stitch_scene(name, args.max_side, args.out)
    from deliverables import write_figures

    write_figures(scenes, args.out)


if __name__ == "__main__":
    main()
