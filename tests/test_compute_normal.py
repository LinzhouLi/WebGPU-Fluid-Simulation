from __future__ import annotations

import numpy as np
import pytest

from normal_reference import normalized, perspective_scale, plane_depth_map


COMPONENT_ATOL = 2e-3
UNIT_LENGTH_ATOL = 2e-3
BACKGROUND_ATOL = 1e-6
MAX_ANGLE_DEGREES = 1.0
FRONT_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def assert_finite(output: np.ndarray) -> None:
    assert np.isfinite(output).all(), "normal map contains NaN or Inf"


def assert_background_is_zero(output: np.ndarray, background_mask: np.ndarray) -> None:
    if np.any(background_mask):
        np.testing.assert_allclose(
            output[background_mask],
            0.0,
            atol=BACKGROUND_ATOL,
            rtol=0.0,
        )


def assert_normals_match(
    output: np.ndarray,
    expected_normal,
    valid_mask: np.ndarray | None = None,
    max_angle_degrees: float = MAX_ANGLE_DEGREES,
) -> None:
    assert_finite(output)
    if valid_mask is None:
        valid_mask = np.ones(output.shape[:2], dtype=bool)
    assert np.any(valid_mask), "test must contain at least one valid pixel"

    values = output[valid_mask]
    np.testing.assert_allclose(
        values[:, 3],
        1.0,
        atol=COMPONENT_ATOL,
        rtol=0.0,
    )

    lengths = np.linalg.norm(values[:, :3], axis=1)
    np.testing.assert_allclose(
        lengths,
        1.0,
        atol=UNIT_LENGTH_ATOL,
        rtol=0.0,
    )

    expected = normalized(expected_normal)
    actual_unit = values[:, :3] / lengths[:, None]
    dots = np.clip(actual_unit @ expected, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    assert float(np.max(angles)) <= max_angle_degrees, (
        f"maximum normal error {float(np.max(angles)):.4f} degrees exceeds "
        f"{max_angle_degrees:.4f} degrees"
    )


def test_shader_compiles_and_dispatches_odd_extent(normal_runner):
    width, height = 13, 11
    depth = np.full((height, width), 2.0, dtype=np.float32)
    output = normal_runner.run(depth, perspective_scale(width, height))

    assert output.shape == (height, width, 4)
    assert_finite(output)
    np.testing.assert_allclose(output[..., 3], 1.0, atol=COMPONENT_ATOL, rtol=0.0)


def test_front_facing_plane(normal_runner):
    width, height = 12, 10
    depth = np.full((height, width), 2.0, dtype=np.float32)
    output = normal_runner.run(depth, perspective_scale(width, height, 50.0))

    assert_normals_match(output, FRONT_NORMAL)


def test_perspective_slanted_plane(normal_runner):
    width, height = 17, 11
    scale = perspective_scale(width, height, 60.0)
    depth, expected = plane_depth_map(
        width,
        height,
        normal=(0.25, -0.35, 0.9),
        center_depth=2.5,
        projection_scale=scale,
    )

    output = normal_runner.run(depth, scale)
    assert_normals_match(output, expected)


def test_slanted_plane_image_boundaries(normal_runner):
    width, height = 5, 4
    scale = perspective_scale(width, height, 55.0)
    depth, expected = plane_depth_map(
        width,
        height,
        normal=(-0.3, 0.2, 0.93),
        center_depth=3.0,
        projection_scale=scale,
    )
    output = normal_runner.run(depth, scale)

    boundary = np.zeros((height, width), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    assert_normals_match(output, expected, boundary)


def test_zero_and_negative_depth_are_background(normal_runner):
    width = height = 11
    scale = perspective_scale(width, height)
    plane, expected = plane_depth_map(
        width,
        height,
        normal=(0.15, -0.2, 0.96),
        center_depth=2.0,
        projection_scale=scale,
    )

    valid = np.zeros((height, width), dtype=bool)
    valid[3:8, 3:8] = True
    yy, xx = np.indices((height, width))
    background = np.where((xx + yy) % 2 == 0, 0.0, -1.0).astype(np.float32)
    depth = np.where(valid, plane, background).astype(np.float32)

    output = normal_runner.run(depth, scale)
    assert_normals_match(output, expected, valid)
    assert_background_is_zero(output, ~valid)


def test_surface_outline_and_background_hole(normal_runner):
    width, height = 17, 15
    scale = perspective_scale(width, height, 65.0)
    plane, expected = plane_depth_map(
        width,
        height,
        normal=(-0.2, -0.25, 0.95),
        center_depth=2.2,
        projection_scale=scale,
    )

    valid = np.zeros((height, width), dtype=bool)
    valid[2:-2, 2:-2] = True
    valid[6:9, 7:10] = False
    depth = np.where(valid, plane, 0.0).astype(np.float32)

    output = normal_runner.run(depth, scale)
    assert_normals_match(output, expected, valid)
    assert_background_is_zero(output, ~valid)


def test_depth_discontinuity_prefers_closer_side(normal_runner):
    width, height = 12, 9
    scale = perspective_scale(width, height)
    depth = np.full((height, width), 1.5, dtype=np.float32)
    depth[:, width // 2 :] = 4.0

    output = normal_runner.run(depth, scale)
    assert_normals_match(output, FRONT_NORMAL)

    discontinuity_columns = np.zeros((height, width), dtype=bool)
    discontinuity_columns[:, width // 2 - 1 : width // 2 + 1] = True
    assert_normals_match(output, FRONT_NORMAL, discontinuity_columns)


@pytest.mark.parametrize("shape", ["isolated", "horizontal", "vertical"])
def test_degenerate_neighborhood_falls_back_safely(normal_runner, shape):
    width = height = 7
    scale = perspective_scale(width, height)
    valid = np.zeros((height, width), dtype=bool)
    if shape == "isolated":
        valid[height // 2, width // 2] = True
    elif shape == "horizontal":
        valid[height // 2, 1:-1] = True
    else:
        valid[1:-1, width // 2] = True

    depth = np.where(valid, 2.0, 0.0).astype(np.float32)
    output = normal_runner.run(depth, scale)

    assert_normals_match(output, FRONT_NORMAL, valid)
    assert_background_is_zero(output, ~valid)


def test_sparse_depth_map_never_produces_non_finite_output(normal_runner):
    width, height = 23, 17
    scale = perspective_scale(width, height, 70.0)
    rng = np.random.default_rng(20260814)
    valid = rng.random((height, width)) > 0.55
    positive_depth = rng.uniform(0.25, 12.0, size=(height, width))
    yy, xx = np.indices((height, width))
    background = np.where((xx + yy) % 2 == 0, 0.0, -0.5)
    depth = np.where(valid, positive_depth, background).astype(np.float32)

    output = normal_runner.run(depth, scale)
    assert_finite(output)
    assert_background_is_zero(output, ~valid)
    np.testing.assert_allclose(
        output[..., 3][valid],
        1.0,
        atol=COMPONENT_ATOL,
        rtol=0.0,
    )
    lengths = np.linalg.norm(output[..., :3][valid], axis=1)
    np.testing.assert_allclose(
        lengths,
        1.0,
        atol=UNIT_LENGTH_ATOL,
        rtol=0.0,
    )
