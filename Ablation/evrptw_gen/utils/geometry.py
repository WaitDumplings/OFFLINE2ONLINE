# evrptw_gen/utils/geometry.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

Rect = Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin, xmax), (ymin, ymax))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two (2,) points."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))

def dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances between A(N,2) and B(M,2) -> (N,M).
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def running_bbox_init(p: np.ndarray) -> Rect:
    x, y = float(p[0]), float(p[1])
    return ((x, x), (y, y))

def running_bbox_update(rect: Rect, p: np.ndarray) -> Rect:
    (xmin, xmax), (ymin, ymax) = rect
    x, y = float(p[0]), float(p[1])
    return ((min(xmin, x), max(xmax, x)), (min(ymin, y), max(ymax, y)))

def inflate_and_clip(rect: Rect, radius: float, bounds: Rect) -> Rect:
    """
    Inflate rect by ±radius and clip to global bounds.
    """
    (xmin, xmax), (ymin, ymax) = rect
    (bxmin, bxmax), (bymin, bymax) = bounds
    x0 = max(bxmin, xmin - radius)
    x1 = min(bxmax, xmax + radius)
    y0 = max(bymin, ymin - radius)
    y1 = min(bymax, ymax + radius)
    return ((x0, x1), (y0, y1))

def sample_uniform_rect(rng: np.random.Generator, rect: Rect, n: int = 1, round_ndigits: int = 3) -> np.ndarray:
    """Uniformly sample n points within rect; returns (n,2)."""
    (xmin, xmax), (ymin, ymax) = rect
    xs = rng.uniform(xmin, xmax, size=n)
    ys = rng.uniform(ymin, ymax, size=n)
    pts = np.stack([xs, ys], axis=1)
    if round_ndigits is not None:
        pts = np.round(pts, round_ndigits)
    return pts

def clip_point_to_rect(p: np.ndarray, rect: Rect) -> np.ndarray:
    (xmin, xmax), (ymin, ymax) = rect
    x = float(np.clip(p[0], xmin, xmax))
    y = float(np.clip(p[1], ymin, ymax))
    return np.array([x, y], dtype=float)

def clamp(v: float, vmin: float, vmax: float) -> float:
    """Clamp a scalar value v to the range [vmin, vmax]."""
    return max(vmin, min(v, vmax))

def clamp_rect(center: tuple[float, float],
               half_extent: float,
               area_bounds: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Build a rectangle centered at (cx, cy) with half-extent 'half_extent',
    clamped to the given area bounds.

    Parameters
    ----------
    center : (float, float)
        The rectangle center (x, y).
    half_extent : float
        Half of the rectangle's width/height (in the same unit as area_bounds).
    area_bounds : ((xmin, xmax), (ymin, ymax))
        The total bounding area to which the rectangle should be clipped.

    Returns
    -------
    ((x_min, x_max), (y_min, y_max))
        The final clamped rectangle boundaries.
    """
    (cx, cy) = center
    (xmin, xmax), (ymin, ymax) = area_bounds

    rxmin = clamp(cx - half_extent, xmin, xmax)
    rxmax = clamp(cx + half_extent, xmin, xmax)
    rymin = clamp(cy - half_extent, ymin, ymax)
    rymax = clamp(cy + half_extent, ymin, ymax)

    return (rxmin, rxmax), (rymin, rymax)
