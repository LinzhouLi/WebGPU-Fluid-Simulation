from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu


CAMERA_FLOAT_COUNT = 56
CAMERA_BUFFER_SIZE = CAMERA_FLOAT_COUNT * np.dtype(np.float32).itemsize
WORKGROUP_SIZE = 8


def make_camera_uniform(projection_scale: tuple[float, float]) -> np.ndarray:
    """Build the 224-byte Camera uniform used by the production shader."""
    camera = np.zeros(CAMERA_FLOAT_COUNT, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32).reshape(-1)

    # position occupies floats 0..2; float 3 is its WGSL alignment padding.
    camera[4:20] = identity
    camera[20:36] = identity
    camera[36:52] = identity
    camera[52:54] = projection_scale
    return camera


class NormalMapRunner:
    """Execute the repository's computeNormal.wgsl without a browser."""

    def __init__(self, device, shader_path: Path):
        self.device = device
        self.shader_path = shader_path
        self.shader_code = shader_path.read_text(encoding="utf-8")
        self.bind_group_layout = self._create_bind_group_layout()
        self.pipeline = self._create_pipeline()

    def _create_bind_group_layout(self):
        return self.device.create_bind_group_layout(
            label="Compute normal test bind group layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": "write-only",
                        "format": "rgba16float",
                        "view_dimension": "2d",
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": False,
                        "min_binding_size": CAMERA_BUFFER_SIZE,
                    },
                },
            ],
        )

    def _create_pipeline(self):
        shader_module = self.device.create_shader_module(
            label=str(self.shader_path),
            code=self.shader_code,
        )
        pipeline_layout = self.device.create_pipeline_layout(
            label="Compute normal test pipeline layout",
            bind_group_layouts=[self.bind_group_layout],
        )
        return self.device.create_compute_pipeline(
            label="Compute normal test pipeline",
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )

    def run(
        self,
        depth_map: np.ndarray,
        projection_scale: tuple[float, float],
    ) -> np.ndarray:
        depth = np.ascontiguousarray(depth_map, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth_map must be 2D, got shape {depth.shape}")

        height, width = depth.shape
        if width == 0 or height == 0:
            raise ValueError("depth_map dimensions must be non-zero")

        input_texture = self.device.create_texture(
            label="Compute normal test depth map",
            size=(width, height, 1),
            dimension="2d",
            format="r32float",
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        output_texture = self.device.create_texture(
            label="Compute normal test normal map",
            size=(width, height, 1),
            dimension="2d",
            format="rgba16float",
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )

        self.device.queue.write_texture(
            {"texture": input_texture},
            depth,
            {
                "offset": 0,
                "bytes_per_row": width * np.dtype(np.float32).itemsize,
                "rows_per_image": height,
            },
            (width, height, 1),
        )

        camera = make_camera_uniform(projection_scale)
        camera_buffer = self.device.create_buffer_with_data(
            label="Compute normal test camera",
            data=camera,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        bind_group = self.device.create_bind_group(
            label="Compute normal test bind group",
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": input_texture.create_view()},
                {"binding": 1, "resource": output_texture.create_view()},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": camera_buffer,
                        "offset": 0,
                        "size": CAMERA_BUFFER_SIZE,
                    },
                },
            ],
        )

        encoder = self.device.create_command_encoder(
            label="Compute normal test command encoder"
        )
        compute_pass = encoder.begin_compute_pass(label="Compute normal test pass")
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(
            (width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
            (height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
            1,
        )
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        raw = self.device.queue.read_texture(
            {"texture": output_texture},
            {
                "offset": 0,
                "bytes_per_row": width * 4 * np.dtype(np.float16).itemsize,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        return (
            np.frombuffer(raw, dtype=np.float16)
            .reshape(height, width, 4)
            .astype(np.float32)
        )
