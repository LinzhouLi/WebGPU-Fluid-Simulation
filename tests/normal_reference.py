from __future__ import annotations

import math

import numpy as np


def normalized(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(value)
    if length == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return (value / length).astype(np.float32)


def perspective_scale(
    width: int,
    height: int,
    vertical_fov_degrees: float = 60.0,
) -> tuple[float, float]:
    aspect = width / height
    scale_y = -2.0 * math.tan(math.radians(vertical_fov_degrees) * 0.5)
    scale_x = -scale_y * aspect
    return scale_x, scale_y


def plane_depth_map(
    width: int,
    height: int,
    normal,
    center_depth: float,
    projection_scale: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect pixel-center view rays with a view-space plane."""
    normal_value = normalized(normal)
    if normal_value[2] <= 0.0:
        raise ValueError("The expected plane normal must face the camera (+Z)")

    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    uv_x, uv_y = np.meshgrid(u, v)
    rays = np.stack(
        [
            (uv_x - 0.5) * projection_scale[0],
            (uv_y - 0.5) * projection_scale[1],
            -np.ones_like(uv_x),
        ],
        axis=-1,
    )

    # dot(normal, position) + offset = 0, with position = ray * depth.
    plane_offset = float(normal_value[2] * center_depth)
    denominator = np.sum(rays * normal_value, axis=-1)
    depth = -plane_offset / denominator
    if not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ValueError("The plane does not produce a valid positive depth map")

    return depth.astype(np.float32), normal_value
