import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import cdist

from harris import dist2, get_harris_corners

N_POINTS = 500
PATCH = 8
WINDOW = 40
STRIDE = WINDOW // PATCH
C_ROBUST = 0.9
LOWE_RATIO = 0.6


def grayscale(image):
    im = np.asarray(image, dtype=np.float64)
    if im.ndim == 2:
        gray = im
    else:
        gray = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
    if gray.max(initial=0) > 1.5:
        gray = gray / 255.0
    return gray


def harris_corners(image, edge_discard=20):
    # Course detector: coords come back as (y, x).
    gray = grayscale(image)
    strength, coords = get_harris_corners(gray, edge_discard=edge_discard)
    if coords.size == 0:
        return strength, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return strength, coords[0].astype(int), coords[1].astype(int)


def adaptive_nms(ys, xs, strength, n_points=N_POINTS, c_robust=C_ROBUST):
    # r_i is the distance to the nearest corner that is c_robust times stronger
    # Keep the corners with the largest radii so they spread across the image
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    strength = np.asarray(strength, dtype=np.float64)
    n = len(strength)
    if n == 0:
        return ys, xs
    if n <= n_points:
        return ys, xs

    order = np.argsort(-strength)
    ys = ys[order]
    xs = xs[order]
    strength = strength[order]
    points = np.column_stack([xs, ys]).astype(np.float64)

    limit = strength / c_robust
    # Points strictly stronger than this limit can suppress the current point
    stronger = np.searchsorted(-strength, -limit, side="left")
    stronger = np.minimum(stronger, np.arange(n))

    radii = np.full(n, np.inf)
    chunk = 256
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        kmax = int(stronger[end - 1])
        if kmax == 0:
            continue
        dist = cdist(points[start:end], points[:kmax])
        for row, k in enumerate(stronger[start:end]):
            if k > 0:
                radii[start + row] = dist[row, :k].min()

    keep = np.argsort(-radii)[:n_points]
    return ys[keep], xs[keep]


def feature_descriptors(image, ys, xs):
    # Average each 40x40 window down to 8x8, then zero-mean / unit-variance.
    gray = gaussian_filter(grayscale(image), sigma=1.0)
    height, width = gray.shape
    half = WINDOW // 2
    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)
    desc = np.zeros((len(ys), PATCH * PATCH), dtype=np.float64)
    keep = np.zeros(len(ys), dtype=bool)
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0 = y - half
        x0 = x - half
        if y0 < 0 or x0 < 0 or y0 + WINDOW > height or x0 + WINDOW > width:
            continue
        window = gray[y0 : y0 + WINDOW, x0 : x0 + WINDOW]
        small = window.reshape(PATCH, STRIDE, PATCH, STRIDE).mean(axis=(1, 3))
        flat = small.reshape(-1)
        std = float(flat.std())
        if std < 1e-6:
            continue
        desc[i] = (flat - flat.mean()) / std
        keep[i] = True
    return desc, keep


def gradient_angles(image, ys, xs, sigma=4.0):
    # Angle of the smoothed gradient. A rotated photo rotates this with it.
    gray = grayscale(image)
    gy = gaussian_filter(gray, sigma, order=(1, 0))
    gx = gaussian_filter(gray, sigma, order=(0, 1))
    return np.arctan2(gy[ys, xs], gx[ys, xs])


def oriented_descriptors(image, ys, xs, angles):
    # Same 8x8 samples over a 40x40 reach, spun so the gradient points the same way.
    gray = gaussian_filter(grayscale(image), sigma=2.0)
    height, width = gray.shape
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    offs = (np.arange(PATCH) - (PATCH - 1) / 2.0) * STRIDE
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    sample_x = xs[:, None, None] + cos * ox - sin * oy
    sample_y = ys[:, None, None] + sin * ox + cos * oy
    inside = (
        (sample_x >= 1)
        & (sample_x < width - 2)
        & (sample_y >= 1)
        & (sample_y < height - 2)
    )
    coords = np.vstack([sample_y.ravel(), sample_x.ravel()])
    sampled = map_coordinates(gray, coords, order=1, mode="nearest").reshape(len(ys), -1)
    mean = sampled.mean(axis=1, keepdims=True)
    std = sampled.std(axis=1, keepdims=True)
    keep = inside.all(axis=(1, 2)) & (std[:, 0] > 1e-6)
    desc = (sampled - mean) / np.maximum(std, 1e-8)
    return desc, keep


def match_descriptors(desc1, desc2, ratio=LOWE_RATIO):
    # Lowe's test: keep a match when the nearest descriptor is much closer than the second nearest
    desc1 = np.asarray(desc1, dtype=np.float64)
    desc2 = np.asarray(desc2, dtype=np.float64)
    if len(desc1) == 0 or len(desc2) < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    squared = np.maximum(dist2(desc1, desc2), 0.0)
    nearest = np.argpartition(squared, kth=1, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    pair = squared[rows[:, None], nearest]
    order = np.argsort(pair, axis=1)
    first = nearest[rows, order[:, 0]]
    best = pair[rows, order[:, 0]]
    second = pair[rows, order[:, 1]]
    keep = second > 1e-12
    keep &= (best / np.maximum(second, 1e-12)) < ratio
    return rows[keep], first[keep]


def match_images(im1, im2, ratio=LOWE_RATIO, n_points=N_POINTS, oriented=False):
    src_y, src_x, src_desc = _describe(im1, n_points, oriented=oriented)
    dst_y, dst_x, dst_desc = _describe(im2, n_points, oriented=oriented)
    if len(src_desc) == 0 or len(dst_desc) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    i1, i2 = match_descriptors(src_desc, dst_desc, ratio=ratio)
    if len(i1) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    src = np.column_stack([src_x[i1], src_y[i1]]).astype(np.float64)
    dst = np.column_stack([dst_x[i2], dst_y[i2]]).astype(np.float64)
    return src, dst


def _describe(image, n_points, oriented=False):
    strength, ys, xs = harris_corners(image)
    if len(ys) == 0:
        return ys, xs, np.zeros((0, PATCH * PATCH))
    response = strength[ys, xs]
    floor = 0.01 * float(response.max())
    strong = response > floor
    ys, xs, response = ys[strong], xs[strong], response[strong]
    ys, xs = adaptive_nms(ys, xs, response, n_points=n_points)
    if oriented:
        angles = gradient_angles(image, ys, xs)
        desc, keep = oriented_descriptors(image, ys, xs, angles)
    else:
        desc, keep = feature_descriptors(image, ys, xs)
    return ys[keep], xs[keep], desc[keep]


def _self_test():
    from mosaic import apply_homography, ransac_homography

    rng = np.random.default_rng(0)
    gray = gaussian_filter(rng.normal(size=(280, 360)), 1.2)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    shift = 28
    moved = np.zeros_like(gray)
    moved[:, shift:] = gray[:, :-shift]

    src, dst = match_images(gray, moved, ratio=0.8, n_points=300)
    if len(src) < 8:
        raise AssertionError(f"translation pair produced {len(src)} matches")
    desc, valid = feature_descriptors(gray, np.array([140]), np.array([180]))
    if not valid[0] or abs(desc[0].mean()) > 1e-8 or abs(desc[0].std() - 1) > 1e-6:
        raise AssertionError("descriptor is not bias/gain normalized")

    H, inliers, _ = ransac_homography(src, dst, threshold=3.0, iterations=800, seed=0)
    if int(inliers.sum()) < 8:
        raise AssertionError(f"RANSAC kept only {int(inliers.sum())} inliers")
    pred = apply_homography(np.array([[100.0, 140.0]]), H)[0]
    if abs(pred[0] - (100.0 + shift)) > 2.5 or abs(pred[1] - 140.0) > 2.5:
        raise AssertionError(f"recovered shift landed at {pred}, expected {(128.0, 140.0)}")
    print(f"features.py self-test ok ({int(inliers.sum())} inliers)")


if __name__ == "__main__":
    _self_test()
