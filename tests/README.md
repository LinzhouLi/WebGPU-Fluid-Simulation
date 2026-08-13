# Depth Map to Normal Map GPU tests

These tests compile and execute the repository's production shader:

```text
src/renderer/filteredParticleFluid/shader/computeNormal.wgsl
```

They do not copy the WGSL implementation into the test suite. `wgpu-py`
creates an `r32float` depth texture, dispatches the compute shader, reads the
`rgba16float` normal texture back, and compares it with CPU reference data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r tests/requirements.txt
python -m pytest tests -q -s
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Linux defaults to the Vulkan backend, so the suite works without a browser or
display server when Mesa llvmpipe/Lavapipe is available.

See [VULKAN_HEADLESS_SETUP.md](VULKAN_HEADLESS_SETUP.md) for the complete
Debian container setup, verification commands, and troubleshooting notes.

The cases cover pipeline compilation and non-workgroup-aligned dimensions,
front-facing and perspective-slanted planes, image boundaries, zero/negative
background depth, surface outlines and holes, sharp depth discontinuities,
degenerate neighborhoods, and randomized sparse input.
