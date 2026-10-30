import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mosaic import (  # noqa: E402
    blend_layers,
    computeH,
    design_matrix,
    normalized_cylinder,
    output_frame,
    overwrite_layers,
    render_cylinder_mosaic,
    render_warp,
    ransac_homography,
)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    register_heif_opener = None


# Fridge front / microwave glass corners, as fractions of the photo. TL, TR, BR, BL.
# Prefer code/rect_quads.json from: python code/pick_corners.py
_RECT_DEFAULT = {
    "fridge": [
        [0.271, 0.138],
        [0.757, 0.220],
        [0.634, 0.870],
        [0.310, 0.887],
    ],
    "microwave": [
        [0.300, 0.556],
        [0.706, 0.417],
        [0.691, 0.644],
        [0.316, 0.785],
    ],
}
_rect_path = Path(__file__).resolve().parent / "rect_quads.json"
if _rect_path.exists():
    RECT_QUADS = {
        name: np.asarray(pts, dtype=np.float64)
        for name, pts in json.loads(_rect_path.read_text()).items()
    }
else:
    RECT_QUADS = {name: np.asarray(pts, dtype=np.float64) for name, pts in _RECT_DEFAULT.items()}

SCENES = ("courtyard", "livingroom", "laundry")
PORTFOLIO = Path("/Users/ethandiec/Desktop/CS280/ui/public/p4")
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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
    if path.suffix.lower() == ".png":
        pic.save(path)
    else:
        pic.save(path, quality=quality, optimize=True)


def match_features(im1, im2, ratio):
    sift = cv2.SIFT_create(nfeatures=5000)
    gray1 = cv2.cvtColor(to_uint8(im1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(to_uint8(im2), cv2.COLOR_RGB2GRAY)
    keys1, desc1 = sift.detectAndCompute(gray1, None)
    keys2, desc2 = sift.detectAndCompute(gray2, None)
    if desc1 is None or desc2 is None or len(keys1) < 4 or len(keys2) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc1, desc2, k=2)
    src, dst = [], []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            src.append(keys1[best.queryIdx].pt)
            dst.append(keys2[best.trainIdx].pt)
    if not src:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)


def fit_pair(src_image, dst_image):
    # H maps src onto dst.
    src = dst = None
    for ratio in (0.75, 0.85):
        src, dst = match_features(src_image, dst_image, ratio=ratio)
        if len(src) >= 16:
            break
    if src is None or len(src) < 4:
        raise RuntimeError("not enough feature matches")
    src, dst = dedupe_matches(src, dst)
    H, inliers, err = ransac_homography(src, dst, threshold=4.0, iterations=2500, seed=0)
    if int(inliers.sum()) < 15:
        H, inliers, err = ransac_homography(src, dst, threshold=8.0, iterations=2500, seed=1)
    return {
        "H": H,
        "src": src,
        "dst": dst,
        "inliers": inliers,
        "error": err,
    }


def dedupe_matches(src, dst, tol=1.5):
    keep = []
    for i, point in enumerate(src):
        if any(np.linalg.norm(point - src[j]) < tol for j in keep):
            continue
        keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return src[keep], dst[keep]


def spread_indices(points, count):
    if len(points) <= count:
        return np.arange(len(points))
    center = np.median(points, axis=0)
    chosen = [int(np.argmin(np.linalg.norm(points - center, axis=1)))]
    while len(chosen) < count:
        dist = np.full(len(points), np.inf)
        for index in chosen:
            dist = np.minimum(dist, np.linalg.norm(points - points[index], axis=1))
        dist[chosen] = -1
        nxt = int(np.argmax(dist))
        if dist[nxt] < 1:
            break
        chosen.append(nxt)
    return np.asarray(chosen)


def draw_pair(im1, im2, pts1, pts2, title, max_height=560):
    scale = min(1.0, max_height / max(im1.shape[0], im2.shape[0]))

    def shrink(image):
        pic = Image.fromarray(to_uint8(image))
        if scale < 1:
            pic = pic.resize(
                (int(round(pic.width * scale)), int(round(pic.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return pic

    left = shrink(im1)
    right = shrink(im2)
    gap = 18
    header = 36
    canvas = Image.new(
        "RGB",
        (left.width + gap + right.width, header + max(left.height, right.height)),
        (8, 12, 20),
    )
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small = load_font(16)
    draw.text((12, 8), title, fill=(220, 232, 242), font=font)
    colors = (
        (255, 99, 99),
        (88, 214, 141),
        (96, 176, 255),
        (255, 206, 84),
        (198, 128, 255),
        (255, 154, 92),
    )
    idx = spread_indices(pts1, 12)
    for n, i in enumerate(idx):
        color = colors[n % len(colors)]
        x1, y1 = pts1[i] * scale
        x2, y2 = pts2[i] * scale
        y1 += header
        y2 += header
        x2 += left.width + gap
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        r = 5
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=color, width=2)
        draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), outline=color, width=2)
        draw.text((x1 + 7, y1 - 14), str(n + 1), fill=color, font=small)
    return canvas


def stack_figures(figures):
    width = max(fig.width for fig in figures)
    gap = 10
    height = sum(fig.height for fig in figures) + gap * (len(figures) - 1)
    canvas = Image.new("RGB", (width, height), (8, 12, 20))
    y = 0
    for fig in figures:
        canvas.paste(fig, (0, y))
        y += fig.height + gap
    return canvas


def source_strip(images, labels):
    height = 520
    tiles = []
    font = load_font(22)
    for image, label in zip(images, labels):
        scale = height / image.shape[0]
        pic = Image.fromarray(to_uint8(image)).resize(
            (int(round(image.shape[1] * scale)), height),
            Image.Resampling.LANCZOS,
        )
        tile = Image.new("RGB", (pic.width, pic.height + 34), (8, 12, 20))
        tile.paste(pic, (0, 34))
        ImageDraw.Draw(tile).text((10, 6), label, fill=(220, 232, 242), font=font)
        tiles.append(tile)
    gap = 8
    canvas = Image.new(
        "RGB",
        (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height + 34),
        (8, 12, 20),
    )
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + gap
    return canvas


def order_quad(points):
    pts = np.asarray(points, dtype=np.float64)
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.vstack([top[0], top[1], bottom[1], bottom[0]])


def quad_pixels(image, fractions):
    height, width = image.shape[:2]
    pts = np.asarray(fractions, dtype=np.float64).copy()
    pts[:, 0] *= width - 1
    pts[:, 1] *= height - 1
    return order_quad(pts)


def draw_quad(image, quad):
    pic = Image.fromarray(to_uint8(image))
    draw = ImageDraw.Draw(pic)
    polygon = [tuple(p) for p in quad]
    draw.line(polygon + [polygon[0]], fill=(255, 196, 64), width=4)
    font = load_font(max(16, pic.width // 50))
    for n, (x, y) in enumerate(polygon):
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 80, 80))
        draw.text((x + 10, y + 6), str(n + 1), fill=(255, 230, 160), font=font)
    return pic


def rectify(image, quad, method):
    # Make the outlined shape a flat rectangle.
    width = 0.5 * (
        np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
    )
    height = 0.5 * (
        np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
    )
    margin_x = int(round(0.08 * width))
    margin_y = int(round(0.08 * height))
    dst = np.array(
        [
            [margin_x, margin_y],
            [margin_x + width, margin_y],
            [margin_x + width, margin_y + height],
            [margin_x, margin_y + height],
        ],
        dtype=np.float64,
    )
    H = computeH(quad, dst)
    out_w = int(np.ceil(width + 2 * margin_x))
    out_h = int(np.ceil(height + 2 * margin_y))
    warped, weight, _ = render_warp(
        image, H, out_shape=(out_h, out_w), offset=(0.0, 0.0), method=method
    )
    box = (margin_x, margin_y, int(round(width)), int(round(height)))
    return warped, weight, H, box


def detail_crop(nearest, bilinear, box, size=260):
    x0, y0, bw, bh = box
    size = int(min(size, bw - 2, bh - 2))
    gray = cv2.cvtColor(to_uint8(bilinear), cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = (x0, y0)
    best_energy = -1.0
    for y in range(y0, max(y0 + 1, y0 + bh - size), 12):
        for x in range(x0, max(x0 + 1, x0 + bw - size), 12):
            patch = gray[y : y + size, x : x + size]
            if patch.shape != (size, size):
                continue
            energy = np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()
            if energy > best_energy:
                best_energy = energy
                best = (x, y)
    x, y = best
    left = nearest[y : y + size, x : x + size]
    right = bilinear[y : y + size, x : x + size]
    gap = 12
    header = 32
    canvas = Image.new("RGB", (size * 2 + gap, size + header), (8, 12, 20))
    canvas.paste(Image.fromarray(to_uint8(left)), (0, header))
    canvas.paste(Image.fromarray(to_uint8(right)), (size + gap, header))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    draw.text((8, 6), "Nearest neighbor", fill=(220, 232, 242), font=font)
    draw.text((size + gap + 8, 6), "Bilinear", fill=(220, 232, 242), font=font)
    return canvas


def mosaic_scene(images, homographies, method):
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
            image, H, out_shape=out_shape, offset=shared, method=method
        )
        layers.append((warped, weight))
    return blend_layers(layers), overwrite_layers(layers)


def pair_record(name, fit, src_name, dst_name):
    err = fit["error"][fit["inliers"]]
    src_in = fit["src"][fit["inliers"]]
    dst_in = fit["dst"][fit["inliers"]]
    shown = spread_indices(src_in, 2)
    A, b = design_matrix(src_in[shown], dst_in[shown])
    full_A, _ = design_matrix(src_in, dst_in)
    return {
        "name": name,
        "from": src_name,
        "to": dst_name,
        "inliers": int(fit["inliers"].sum()),
        "matches": int(len(fit["src"])),
        "mean_error_px": round(float(err.mean()), 3),
        "median_error_px": round(float(np.median(err)), 3),
        "H": [[round(float(v), 6) for v in row] for row in fit["H"]],
        "equation_rows": int(full_A.shape[0]),
        "equation_cols": int(full_A.shape[1]),
        "unnormalized_condition": round(float(np.linalg.cond(full_A)), 1),
        "example_A": [[round(float(v), 2) for v in row] for row in A[:4]],
        "example_b": [round(float(v), 2) for v in b[:4]],
        "example_src": [[round(float(v), 1) for v in row] for row in src_in[shown]],
        "example_dst": [[round(float(v), 1) for v in row] for row in dst_in[shown]],
    }


def choose_focal_length(pairs, shapes):
    # Pick cylinder radius where matches share one shift.
    width = float(np.mean([shape[1] for shape in shapes]))
    guess = width * 26.0 / 36.0

    def residual(f):
        scores = []
        for src, dst, src_shape, dst_shape in pairs:
            a = normalized_cylinder(src, f, src_shape)
            b = normalized_cylinder(dst, f, dst_shape)
            shift = np.median(b - a, axis=0)
            scores.append(np.median(np.linalg.norm(a + shift - b, axis=1)))
        return float(np.mean(scores))

    coarse = np.linspace(0.35 * width, 2.4 * width, 42)
    values = [(residual(f), float(f)) for f in coarse]
    _, best = min(values, key=lambda item: item[0])
    fine = np.linspace(best * 0.85, best * 1.15, 21)
    values.extend((residual(f), float(f)) for f in fine)
    score, f = min(values, key=lambda item: item[0])
    return f, guess, score


def build_scene(name, max_side, out_dir):
    folder = ROOT / "images" / name
    images = {side: load_image(resolve_image(folder, side), max_side) for side in ("left", "center", "right")}
    print(f"[{name}] matching")
    left = fit_pair(images["left"], images["center"])
    right = fit_pair(images["right"], images["center"])
    homographies = [left["H"], np.eye(3), right["H"]]
    ordered = [images["left"], images["center"], images["right"]]
    print(f"[{name}] warping  left inliers={int(left['inliers'].sum())} right inliers={int(right['inliers'].sum())}")
    t0 = time.perf_counter()
    blended, hard = mosaic_scene(ordered, homographies, method="bilinear")
    elapsed = time.perf_counter() - t0

    source_strip(
        [images["left"], images["center"], images["right"]],
        ["Left", "Center", "Right"],
    ).save(out_dir / f"{name}_sources.jpg", quality=86, optimize=True)
    stack_figures(
        [
            draw_pair(
                images["left"],
                images["center"],
                left["src"][left["inliers"]],
                left["dst"][left["inliers"]],
                "Left and center",
            ),
            draw_pair(
                images["right"],
                images["center"],
                right["src"][right["inliers"]],
                right["dst"][right["inliers"]],
                "Right and center",
            ),
        ]
    ).save(out_dir / f"{name}_matches.jpg", quality=86, optimize=True)
    save_image(out_dir / f"{name}_mosaic.jpg", blended)
    if name == "courtyard":
        save_image(out_dir / f"{name}_hard.jpg", hard)

    summary = {
        "size": list(images["center"].shape),
        "warp_seconds": round(elapsed, 2),
        "left": pair_record("left", left, "left", "center"),
        "right": pair_record("right", right, "right", "center"),
    }
    return images, left, right, summary


def cylinder_shifts(left, right, images, f):
    def shift_of(fit, src_shape):
        src = fit["src"][fit["inliers"]]
        dst = fit["dst"][fit["inliers"]]
        a = normalized_cylinder(src, f, src_shape)
        b = normalized_cylinder(dst, f, images["center"].shape)
        return np.median(b - a, axis=0) * f

    return [
        shift_of(left, images["left"].shape),
        np.zeros(2),
        shift_of(right, images["right"].shape),
    ]


def rectify_one(name, max_side, out_dir):
    path = resolve_image(ROOT / "images" / "rectify", name)
    image = load_image(path, max_side)
    quad = quad_pixels(image, RECT_QUADS[name])
    print(f"[rectify {name}]")
    t0 = time.perf_counter()
    nearest, _, H, box = rectify(image, quad, method="nearest")
    nearest_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    bilinear, _, _, _ = rectify(image, quad, method="bilinear")
    bilinear_s = time.perf_counter() - t1
    draw_quad(image, quad).save(out_dir / f"{name}_marked.jpg", quality=86, optimize=True)
    save_image(out_dir / f"{name}_nearest.jpg", nearest)
    save_image(out_dir / f"{name}_bilinear.jpg", bilinear)
    detail_crop(nearest, bilinear, box).save(out_dir / f"{name}_compare.jpg", quality=90)
    return {
        "quad": [[round(float(v), 1) for v in row] for row in quad],
        "H": [[round(float(v), 6) for v in row] for row in H],
        "nearest_seconds": round(nearest_s, 2),
        "bilinear_seconds": round(bilinear_s, 2),
        "output_size": list(bilinear.shape),
    }


def write_summary(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def copy_to_portfolio(out_dir, web_dir):
    if web_dir is None:
        return
    web_dir = Path(web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    for image in sorted(out_dir.glob("*.jpg")):
        shutil.copy2(image, web_dir / image.name)
    print(f"copied figures to {web_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Project 4A figures.")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--web-dir", type=Path, default=PORTFOLIO)
    parser.add_argument("--only", choices=("all", "mosaic", "rectify", "cylinder"), default="all")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir = None if str(args.web_dir) == "" else args.web_dir

    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists() and args.only != "all":
        summary = json.loads(summary_path.read_text())
    summary["max_side"] = args.max_side

    scenes = {}
    if args.only in ("all", "mosaic", "cylinder"):
        for name in SCENES:
            images, left, right, record = build_scene(name, args.max_side, out_dir)
            scenes[name] = (images, left, right)
            summary[name] = record
            print(
                f"[{name}] error left {record['left']['mean_error_px']} px, "
                f"right {record['right']['mean_error_px']} px, warp {record['warp_seconds']} s"
            )

    if args.only in ("all", "cylinder"):
        if "courtyard" not in scenes:
            images = {
                side: load_image(resolve_image(ROOT / "images" / "courtyard", side), args.max_side)
                for side in ("left", "center", "right")
            }
            left = fit_pair(images["left"], images["center"])
            right = fit_pair(images["right"], images["center"])
        else:
            images, left, right = scenes["courtyard"]
        pairs = [
            (left["src"][left["inliers"]], left["dst"][left["inliers"]], images["left"].shape, images["center"].shape),
            (right["src"][right["inliers"]], right["dst"][right["inliers"]], images["right"].shape, images["center"].shape),
        ]
        f, guess, score = choose_focal_length(pairs, [im.shape for im in images.values()])
        shifts = cylinder_shifts(left, right, images, f)
        print(f"[cylinder] f={f:.1f} px  26mm-equivalent guess={guess:.1f}  residual={score:.5f}")
        t0 = time.perf_counter()
        cyl = render_cylinder_mosaic(
            [images["left"], images["center"], images["right"]],
            shifts,
            f,
            method="bilinear",
        )
        save_image(out_dir / "courtyard_cylinder.jpg", cyl, max_side=2000)
        summary["cylinder"] = {
            "f_px": round(float(f), 1),
            "f_35mm_guess_px": round(float(guess), 1),
            "normalized_residual": round(float(score), 5),
            "seconds": round(time.perf_counter() - t0, 2),
            "shifts": [[round(float(v), 2) for v in shift] for shift in shifts],
        }

    if args.only in ("all", "rectify"):
        summary["rectify"] = {
            name: rectify_one(name, args.max_side, out_dir) for name in ("fridge", "microwave")
        }

    write_summary(summary_path, summary)
    copy_to_portfolio(out_dir, web_dir)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
