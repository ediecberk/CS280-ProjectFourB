import numpy as np
from scipy.ndimage import distance_transform_edt


def design_matrix(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rows = []
    b = []
    for (x, y), (xp, yp) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y])
        b.append(xp)
        b.append(yp)
    return np.asarray(rows, dtype=np.float64), np.asarray(b, dtype=np.float64)


def _normalizer(pts):
    # Center points and scale them so math stays stable.
    mean = pts.mean(axis=0)
    dist = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
    if dist < 1e-8:
        raise np.linalg.LinAlgError("point set is degenerate")
    scale = np.sqrt(2.0) / dist
    return np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_homography(points, H):
    pts = np.asarray(points, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    mapped = hom @ np.asarray(H, dtype=np.float64).T
    w = mapped[:, 2:3]
    bad = np.abs(w) < 1e-12
    w = np.where(bad, np.nan, w)
    return mapped[:, :2] / w


def computeH(im1_pts, im2_pts):
    # Find H that maps im1_pts onto im2_pts.
    src = np.asarray(im1_pts, dtype=np.float64)
    dst = np.asarray(im2_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("im1_pts and im2_pts must both have shape (n, 2)")
    if len(src) < 4:
        raise ValueError("computeH needs at least 4 point pairs")

    t_src = _normalizer(src)
    t_dst = _normalizer(dst)
    A, b = design_matrix(apply_homography(src, t_src), apply_homography(dst, t_dst))
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    Hn = np.array(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ],
        dtype=np.float64,
    )
    H = np.linalg.inv(t_dst) @ Hn @ t_src
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        raise np.linalg.LinAlgError("homography is degenerate")
    return H / H[2, 2]


def reprojection_error(src, dst, H):
    pred = apply_homography(src, H)
    return np.linalg.norm(pred - np.asarray(dst, dtype=np.float64), axis=1)


def ransac_homography(src, dst, threshold=4.0, iterations=2500, seed=0):
    # Try many 4-point Hs, keep the one with the most good matches.
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        raise ValueError("need at least 4 correspondences")
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_H = None
    best_count = -1
    for _ in range(iterations):
        idx = rng.choice(n, size=4, replace=False)
        try:
            H = computeH(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = reprojection_error(src, dst, H)
        inliers = np.isfinite(err) & (err < threshold)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H
    if best_inliers is None or best_count < 4:
        raise RuntimeError("RANSAC did not find a homography")
    try:
        H = computeH(src[best_inliers], dst[best_inliers])
    except np.linalg.LinAlgError:
        H = best_H
    err = reprojection_error(src, dst, H)
    inliers = np.isfinite(err) & (err < threshold)
    if int(inliers.sum()) >= 4:
        try:
            H = computeH(src[inliers], dst[inliers])
            err = reprojection_error(src, dst, H)
        except np.linalg.LinAlgError:
            pass
    return H, inliers, err


def _as_float_image(image):
    im = np.asarray(image)
    if im.ndim == 2:
        im = im[..., None]
    im = im.astype(np.float32, copy=False)
    if im.dtype == np.float32 and im.max(initial=0) > 1.5:
        im = im / 255.0
    return np.clip(im, 0.0, 1.0)


def border_weight(height, width, feather=0.22):
    # 1 in the middle, fades to 0 at the edges.
    mask = np.ones((height, width), dtype=np.uint8)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    dist = distance_transform_edt(mask).astype(np.float32)
    reach = max(feather * min(height, width), 1.0)
    return np.clip(dist / reach, 0.0, 1.0)


def _sample(image, x, y, method):
    squeeze = image.ndim == 2
    if squeeze:
        image = image[..., None]
    height, width = image.shape[:2]
    valid = np.isfinite(x) & np.isfinite(y)
    if method == "nearest":
        xi = np.floor(x + 0.5).astype(np.int32)
        yi = np.floor(y + 0.5).astype(np.int32)
        valid &= (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        xi = np.clip(xi, 0, width - 1)
        yi = np.clip(yi, 0, height - 1)
        sampled = image[yi, xi]
    elif method == "bilinear":
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        valid &= (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)
        x0c = np.clip(x0, 0, width - 1)
        x1c = np.clip(x1, 0, width - 1)
        y0c = np.clip(y0, 0, height - 1)
        y1c = np.clip(y1, 0, height - 1)
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        sampled = (
            wa[..., None] * image[y0c, x0c]
            + wb[..., None] * image[y0c, x1c]
            + wc[..., None] * image[y1c, x0c]
            + wd[..., None] * image[y1c, x1c]
        )
    else:
        raise ValueError(f"unknown interpolation {method}")
    sampled = np.where(valid[..., None], sampled, 0).astype(np.float32)
    if squeeze:
        sampled = sampled[..., 0]
    return sampled, valid


def output_frame(image, H, pad=2, max_pixels=40_000_000):
    # How big the stretched photo needs to be.
    height, width = image.shape[:2]
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]
    )
    warped = apply_homography(corners, H)
    if not np.isfinite(warped).all():
        raise ValueError("warped corners are not finite")
    min_xy = np.floor(warped.min(axis=0)).astype(int) - pad
    max_xy = np.ceil(warped.max(axis=0)).astype(int) + pad
    out_w = int(max_xy[0] - min_xy[0] + 1)
    out_h = int(max_xy[1] - min_xy[1] + 1)
    if out_w <= 0 or out_h <= 0 or out_w * out_h > max_pixels:
        raise ValueError(f"warp output is unreasonable: {out_w} x {out_h}")
    return (out_h, out_w), (int(min_xy[0]), int(min_xy[1]))


def render_warp(image, H, out_shape=None, offset=None, method="bilinear", chunk=96):
    # Stretch image with H by looking up each output pixel in the source.
    image = _as_float_image(image)
    if out_shape is None or offset is None:
        out_shape, offset = output_frame(image, H)
    out_h, out_w = out_shape
    Hinv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    src_h, src_w = image.shape[:2]
    weight = border_weight(src_h, src_w)
    warped = np.zeros((out_h, out_w, image.shape[2]), dtype=np.float32)
    warped_w = np.zeros((out_h, out_w), dtype=np.float32)
    xs = np.arange(out_w, dtype=np.float64) + offset[0]

    for y0 in range(0, out_h, chunk):
        y1 = min(out_h, y0 + chunk)
        ys = np.arange(y0, y1, dtype=np.float64) + offset[1]
        grid_x, grid_y = np.meshgrid(xs, ys)
        dest = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
        src = dest @ Hinv.T
        src = src[..., :2] / src[..., 2:3]
        sampled, valid = _sample(image, src[..., 0], src[..., 1], method)
        w_sampled, _ = _sample(weight, src[..., 0], src[..., 1], method)
        warped[y0:y1] = sampled
        warped_w[y0:y1] = np.where(valid, w_sampled, 0.0)
    return warped, warped_w, offset


def warpImageNearestNeighbor(im, H):
    warped, _, _ = render_warp(im, H, method="nearest")
    return warped


def warpImageBilinear(im, H):
    warped, _, _ = render_warp(im, H, method="bilinear")
    return warped


def blend_layers(layers):
    # Soft mix: average by weight.
    acc = None
    wacc = None
    for image, weight in layers:
        if acc is None:
            acc = np.zeros(image.shape, dtype=np.float64)
            wacc = np.zeros(image.shape[:2], dtype=np.float64)
        acc += image * weight[..., None]
        wacc += weight
    out = acc / np.maximum(wacc, 1e-8)[..., None]
    out[wacc < 1e-6] = 0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def overwrite_layers(layers):
    # Hard paste: later layers win.
    out = np.zeros_like(layers[0][0])
    for image, weight in layers:
        mask = weight > 1e-4
        out[mask] = image[mask]
    return out


def cylinder_points(points, f, center):
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0] - center[0]
    y = pts[:, 1] - center[1]
    theta = np.arctan(x / f)
    yc = f * y / np.sqrt(x * x + f * f)
    return np.stack([f * theta, yc], axis=1)


def _cylinder_limits(shape, f):
    height, width = shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    theta_min = np.arctan((0.0 - cx) / f)
    theta_max = np.arctan((width - 1.0 - cx) / f)
    y_min = -cy
    y_max = (height - 1.0) - cy
    out_w = int(np.floor(f * (theta_max - theta_min))) + 1
    out_h = int(np.floor(y_max - y_min)) + 1
    return (cx, cy), theta_min, y_min, out_w, out_h


def cylinder_pixels(points, f, shape):
    center, theta_min, y_min, _, _ = _cylinder_limits(shape, f)
    cyl = cylinder_points(points, f, center)
    px = cyl[:, 0] - f * theta_min
    py = cyl[:, 1] - y_min
    return np.stack([px, py], axis=1)


def normalized_cylinder(points, f, shape):
    height, width = shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return cylinder_points(points, f, center) / f


def render_cylinder_mosaic(images, shifts, f, method="bilinear", chunk=96):
    # Wrap photos onto a cylinder, then blend.
    images = [_as_float_image(im) for im in images]
    limits = [_cylinder_limits(im.shape, f) for im in images]
    widths = []
    heights = []
    for shift, (_, _, _, out_w, out_h) in zip(shifts, limits):
        widths.extend([shift[0], shift[0] + out_w - 1])
        heights.extend([shift[1], shift[1] + out_h - 1])
    min_x = int(np.floor(min(widths))) - 2
    min_y = int(np.floor(min(heights))) - 2
    max_x = int(np.ceil(max(widths))) + 2
    max_y = int(np.ceil(max(heights))) + 2
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    if out_w * out_h > 40_000_000:
        raise ValueError(f"cylindrical canvas is unreasonable: {out_w} x {out_h}")

    acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    wacc = np.zeros((out_h, out_w), dtype=np.float64)
    xs = np.arange(out_w, dtype=np.float64) + min_x

    prepared = []
    for image, shift, (center, theta_min, y_min, out_w_i, out_h_i) in zip(images, shifts, limits):
        prepared.append(
            {
                "image": image,
                "weight": border_weight(image.shape[0], image.shape[1]),
                "shift": np.asarray(shift, dtype=np.float64),
                "center": center,
                "theta_min": theta_min,
                "y_min": y_min,
                "out_w": out_w_i,
                "out_h": out_h_i,
            }
        )

    for y0 in range(0, out_h, chunk):
        y1 = min(out_h, y0 + chunk)
        ys = np.arange(y0, y1, dtype=np.float64) + min_y
        grid_x, grid_y = np.meshgrid(xs, ys)
        for item in prepared:
            local_x = grid_x - item["shift"][0]
            local_y = grid_y - item["shift"][1]
            inside = (
                (local_x >= 0)
                & (local_x < item["out_w"])
                & (local_y >= 0)
                & (local_y < item["out_h"])
            )
            if not np.any(inside):
                continue
            theta = item["theta_min"] + local_x / f
            y_cyl = item["y_min"] + local_y
            cos_t = np.cos(theta)
            safe = inside & (np.abs(cos_t) > 1e-3)
            x_src = f * np.tan(np.clip(theta, -1.4, 1.4)) + item["center"][0]
            y_src = y_cyl / np.where(np.abs(cos_t) < 1e-3, 1e-3, cos_t) + item["center"][1]
            sampled, valid = _sample(item["image"], x_src, y_src, method)
            w_sampled, _ = _sample(item["weight"], x_src, y_src, method)
            use = safe & valid
            w = np.where(use, w_sampled, 0.0)
            acc[y0:y1] += sampled * w[..., None]
            wacc[y0:y1] += w

    out = acc / np.maximum(wacc, 1e-8)[..., None]
    out[wacc < 1e-6] = 0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _self_test():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 200, size=(12, 2))
    H_true = np.array(
        [[1.15, 0.08, 12.0], [-0.05, 0.92, -6.0], [0.0004, -0.0002, 1.0]],
        dtype=np.float64,
    )
    dst = apply_homography(src, H_true)
    H_est = computeH(src, dst)
    if not np.allclose(H_est, H_true, atol=1e-6):
        raise AssertionError(f"computeH missed a known homography\n{H_est}\n{H_true}")

    image = np.zeros((36, 48, 3), dtype=np.float32)
    image[8, 11] = (1.0, 0.4, 0.2)
    H_shift = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
    warped, _, offset = render_warp(image, H_shift, method="nearest")
    x = 18 - offset[0]
    y = 13 - offset[1]
    if not np.allclose(warped[y, x], image[8, 11]):
        raise AssertionError("nearest-neighbor translation landed on the wrong pixel")

    flat = np.full((20, 24, 3), 0.35, dtype=np.float32)
    H_id = np.eye(3)
    bilinear, weight, _ = render_warp(flat, H_id, method="bilinear")
    interior = weight > 0.5
    if bilinear[interior].size == 0 or not np.allclose(bilinear[interior], 0.35, atol=1e-5):
        raise AssertionError("bilinear warp did not keep a flat field")
    print("mosaic.py self-test ok")


if __name__ == "__main__":
    _self_test()
