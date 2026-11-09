import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    WINDOW,
    adaptive_nms,
    feature_descriptors,
    harris_corners,
    match_images,
)
from main import SCENES, to_uint8  # noqa: E402
from mosaic import apply_homography, render_warp  # noqa: E402

MANUAL = ROOT / "manual"
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)
COLORS = (
    (255, 99, 99),
    (88, 214, 141),
    (96, 176, 255),
    (255, 206, 84),
    (198, 128, 255),
    (255, 154, 92),
    (120, 230, 220),
    (255, 140, 190),
)


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_pic(path, pic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pic.save(path, quality=86, optimize=True)
    print(f"wrote {path.name}")


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


def corner_sets(image):
    strength, ys, xs = harris_corners(image)
    response = strength[ys, xs]
    strong = response > 0.01 * float(response.max())
    ys, xs, response = ys[strong], xs[strong], response[strong]
    kept_y, kept_x = adaptive_nms(ys, xs, response)
    return np.column_stack([xs, ys]), np.column_stack([kept_x, kept_y])


def paint_corners(image, points, radius, color):
    canvas = to_uint8(image).copy()
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), radius, color, 1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)


def titled(pic, title):
    header = 36
    canvas = Image.new("RGB", (pic.width, pic.height + header), (8, 12, 20))
    canvas.paste(pic, (0, header))
    ImageDraw.Draw(canvas).text((12, 8), title, fill=(220, 232, 242), font=load_font(22))
    return canvas


def descriptor_sheet(image, points):
    ys = points[:, 1].astype(int)
    xs = points[:, 0].astype(int)
    desc, keep = feature_descriptors(image, ys, xs)
    ys, xs, desc = ys[keep], xs[keep], desc[keep]
    chosen = spread_indices(np.column_stack([xs, ys]), 6)
    marked = paint_corners(image, np.column_stack([xs[chosen], ys[chosen]]), 8, (255, 196, 64))
    draw = ImageDraw.Draw(marked)
    font = load_font(max(18, marked.width // 50))
    for n, i in enumerate(chosen):
        x, y = int(xs[i]), int(ys[i])
        draw.text((x + 10, y - 8), str(n + 1), fill=(255, 230, 160), font=font)

    cell = 86
    gap = 14
    tiles = []
    for n, i in enumerate(chosen):
        y, x = int(ys[i]), int(xs[i])
        half = WINDOW // 2
        crop = to_uint8(image)[y - half : y + half, x - half : x + half]
        patch = desc[i].reshape(8, 8)
        patch = patch - patch.min()
        patch = patch / max(float(patch.max()), 1e-6)
        small = Image.fromarray((patch * 255).astype(np.uint8), mode="L").resize(
            (cell, cell), Image.Resampling.NEAREST
        )
        photo = Image.fromarray(crop).resize((cell, cell), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (cell, cell * 2 + 28), (8, 12, 20))
        tile.paste(photo, (0, 24))
        tile.paste(small.convert("RGB"), (0, 24 + cell))
        ImageDraw.Draw(tile).text((8, 2), str(n + 1), fill=COLORS[n], font=load_font(18))
        tiles.append(tile)

    width = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    row = Image.new("RGB", (width, tiles[0].height), (8, 12, 20))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile.width + gap
    scale = min(1.0, marked.width / row.width)
    if scale < 1:
        row = row.resize((int(row.width * scale), int(row.height * scale)), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (marked.width, marked.height + row.height + 8), (8, 12, 20))
    canvas.paste(marked, (0, 0))
    canvas.paste(row, ((marked.width - row.width) // 2, marked.height + 8))
    return canvas


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
    draw.text((12, 8), title, fill=(220, 232, 242), font=load_font(22))
    small = load_font(16)
    idx = spread_indices(pts1, 12)
    for n, i in enumerate(idx):
        color = COLORS[n % len(COLORS)]
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


def stack(figures):
    width = max(fig.width for fig in figures)
    gap = 10
    height = sum(fig.height for fig in figures) + gap * (len(figures) - 1)
    canvas = Image.new("RGB", (width, height), (8, 12, 20))
    y = 0
    for fig in figures:
        canvas.paste(fig, (0, y))
        y += fig.height + gap
    return canvas


def load_manual(name):
    path = MANUAL / f"{name}_mosaic.jpg"
    if not path.exists():
        raise FileNotFoundError(f"missing Part A mosaic {path}")
    pic = Image.open(path).convert("RGB")
    return np.asarray(pic).astype(np.float32) / 255.0


def compare(manual, auto):
    def panel(image, label):
        pic = Image.fromarray(to_uint8(image))
        header = 36
        tile = Image.new("RGB", (pic.width, pic.height + header), (8, 12, 20))
        tile.paste(pic, (0, header))
        ImageDraw.Draw(tile).text((12, 8), label, fill=(220, 232, 242), font=load_font(22))
        return tile

    left = panel(manual, "Points I clicked")
    right = panel(auto, "Points the matcher found")
    height = 640
    tiles = []
    for tile in (left, right):
        scale = height / tile.height
        tiles.append(tile.resize((int(round(tile.width * scale)), height), Image.Resampling.LANCZOS))
    gap = 12
    canvas = Image.new("RGB", (tiles[0].width + gap + tiles[1].width, height), (8, 12, 20))
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (tiles[0].width + gap, 0))
    return canvas


def spin(image, degrees):
    # Rotate around the photo's center. H maps a pixel here onto the spun photo.
    height, width = image.shape[:2]
    ang = np.deg2rad(degrees)
    c, s = np.cos(ang), np.sin(ang)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    H = np.array(
        [
            [c, -s, cx - c * cx + s * cy],
            [s, c, cy - s * cx - c * cy],
            [0.0, 0.0, 1.0],
        ]
    )
    warped, _, offset = render_warp(image, H, method="bilinear")
    return np.clip(warped, 0, 1).astype(np.float32), H, offset


def write_figures(scenes, out_dir):
    """Draw the Part B figures from a stitch run. Called by main.py."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    center = scenes["courtyard"]["images"][1]
    raw, kept = corner_sets(center)
    print(f"harris {len(raw)}  anms {len(kept)}")
    save_pic(out_dir / "b_harris.jpg", titled(paint_corners(center, raw, 2, (255, 90, 90)), "Harris corners without ANMS"))
    save_pic(
        out_dir / "b_anms.jpg",
        titled(paint_corners(center, kept, 4, (255, 196, 64)), "Harris corners with ANMS"),
    )
    save_pic(out_dir / "b_descriptors.jpg", descriptor_sheet(center, kept))

    for name in SCENES:
        scene = scenes[name]
        pairs = [draw_pair(left, right, src, dst, title) for title, left, right, src, dst in scene["pairs"]]
        save_pic(out_dir / f"b_{name}_matches.jpg", stack(pairs))
        save_pic(out_dir / f"b_{name}_compare.jpg", compare(load_manual(name), scene["mosaic"]))

    original = center
    turned, true_H, offset = spin(original, 40)
    for oriented, slug, title in (
        (False, "flat", "Without rotation invariance"),
        (True, "turned", "With rotation invariance"),
    ):
        src, dst = match_images(original, turned, ratio=0.8, oriented=oriented)
        if len(src) == 0:
            good = np.zeros(0, dtype=bool)
        else:
            pred = apply_homography(src, true_H)
            pred = pred - np.array(offset, dtype=np.float64)
            err = np.linalg.norm(pred - dst, axis=1)
            good = np.isfinite(err) & (err < 5)
        print(f"rotate {slug}: matches={len(src)} correct={int(good.sum())}")
        drawn_src = src[good] if good.any() else np.zeros((0, 2))
        drawn_dst = dst[good] if good.any() else np.zeros((0, 2))
        save_pic(
            out_dir / f"b_rotate_{slug}.jpg",
            draw_pair(original, turned, drawn_src, drawn_dst, title, max_height=480),
        )
