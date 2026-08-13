from __future__ import annotations

import os
from pathlib import Path
import sys

# The target headless Linux environment uses Mesa llvmpipe/Lavapipe. On other
# platforms, leave backend selection to wgpu-native (for example, D3D12 on
# Windows) so the same tests remain convenient to run locally.
if sys.platform.startswith("linux"):
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import pytest
import wgpu

from wgpu_runner import NormalMapRunner


SHADER_RELATIVE_PATH = Path(
    "src/renderer/filteredParticleFluid/shader/computeNormal.wgsl"
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    configured_root = os.environ.get("PROJECT_ROOT")
    candidates = [
        Path(configured_root) if configured_root else None,
        Path("/workspace"),
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / SHADER_RELATIVE_PATH).is_file():
            return candidate

    pytest.fail(
        "Project shader not found. Place the repository in /workspace or set "
        "PROJECT_ROOT to the repository root."
    )


@pytest.fixture(scope="session")
def gpu_device():
    adapters = wgpu.gpu.enumerate_adapters_sync()
    if not adapters:
        pytest.fail(
            "No WebGPU adapter found. Install a Vulkan driver or Mesa LavaPipe "
            "and run with WGPU_BACKEND_TYPE=Vulkan."
        )

    adapter = adapters[0]
    print(f"\nWebGPU adapter: {adapter.info}")
    return adapter.request_device_sync()


@pytest.fixture(scope="session")
def normal_runner(gpu_device, project_root: Path) -> NormalMapRunner:
    shader_path = project_root / SHADER_RELATIVE_PATH
    assert shader_path.is_file(), f"Shader not found: {shader_path}"
    return NormalMapRunner(gpu_device, shader_path)
