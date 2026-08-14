import { ShaderStruct, ShaderCode, ShaderFunction } from '../../common/shader';

/**
 * Screen-space fluid rendering based on:
 * "A Narrow-Range Filter for Screen-Space Fluid Rendering" (i3D 2018)
 *
 * Frame graph:
 *   1. prepass   : billboard particles -> particle depth map (spherical front depth)
 *   2. convert   : reverse-Z depth -> linear eye depth (0 = empty sentinel)
 *   3. filter    : separable narrow-range filter H/V + 5x5 clean-up (2+1 iterations)
 *   4. composite : reconstruct surface from smoothed depth, reflection/refraction via envMap
 */

/* ------------------------------------------------------------------ */
/* Pass 1: billboard particle depth prepass                            */
/* ------------------------------------------------------------------ */

const PrepassShared = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}

struct FluidParams {
  particleRadius: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32
};

@group(1) @binding(0) var<storage, read> particlePositions: array<vec4<f32>>;
@group(1) @binding(1) var<uniform> fluidParams: FluidParams;
`;

const PrepassVertexShader = /* wgsl */`
${PrepassShared}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) zCenter: f32
};

@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex % 6u];

  let center = particlePositions[instanceIndex].xyz;
  let clip = camera.projectionMatrix * camera.viewMatrix * vec4<f32>(center, 1.0);

  var out: VertexOutput;

  // particle behind / too close to the camera: collapse offscreen
  if (clip.w < 0.1) {
    out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.corner = vec2<f32>(2.0, 0.0);
    out.zCenter = 0.0;
    return out;
  }

  // Approximate the perspective projection of the sphere by a screen-space
  // circle: clip-space radius offset (projection error accepted).
  let r = fluidParams.particleRadius;
  let offX = corner.x * r * 2.0 / camera.params.x;
  let offY = corner.y * r * -2.0 / camera.params.y; // params.y < 0

  out.position = vec4<f32>(clip.x + offX * clip.w, clip.y + offY * clip.w, clip.z, clip.w);
  out.corner = corner;
  out.zCenter = clip.w;
  return out;
}
`;

const PrepassFragmentShader = /* wgsl */`
${PrepassShared}

struct FragmentInput {
  @location(0) corner: vec2<f32>,
  @location(1) zCenter: f32
};

@fragment
fn main(input: FragmentInput) -> @builtin(frag_depth) f32 {
  let r2 = dot(input.corner, input.corner);
  if (r2 > 1.0) { discard; }

  // front depth of the spherical particle in view space
  let r = fluidParams.particleRadius;
  let x = input.corner.x * r;
  let y = input.corner.y * r;
  let zFront = input.zCenter - sqrt(max(r * r - x * x - y * y, 0.0));

  // reverse-Z device depth from eye depth: 1/z = d * params.z + params.w
  return (1.0 / zFront - camera.params.w) / camera.params.z;
}
`;

/* ------------------------------------------------------------------ */
/* Pass 2/3: narrow-range filter (separable) + clean-up                */
/* ------------------------------------------------------------------ */

const FilterShared = /* wgsl */`
${ShaderStruct.Camera}

struct FilterParams {
  sigma: f32, // world-space kernel std dev
  delta: f32, // narrow range
  mu: f32,    // clamping distance
  pad0: f32
};

@group(0) @binding(0) var<uniform> filterParams: FilterParams;
@group(0) @binding(1) var<uniform> camera: Camera;
`;

// reverse-Z depth -> linear eye depth
const ConvertShader = /* wgsl */`
${FilterShared}

@group(0) @binding(2) var depthIn: texture_depth_2d;
@group(0) @binding(3) var texOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = vec2<i32>(textureDimensions(texOut));
  let coord = vec2<i32>(gid.xy);
  if (coord.x >= size.x || coord.y >= size.y) { return; }

  let d = textureLoad(depthIn, coord, 0);
  var eye = 0.0;
  if (d > 0.0) {
    eye = 1.0 / (d * camera.params.z + camera.params.w);
  }
  textureStore(texOut, coord, vec4<f32>(eye, 0.0, 0.0, 0.0));
}
`;

// separable 1D narrow-range filter with bias correction (Eq. 6) and
// dynamic range adjustment (Eq. 7-9). Kernel size per pixel from Eq. 5.
function makeFilterShader(horizontal: boolean) {
  return /* wgsl */`
${FilterShared}

@group(0) @binding(2) var texIn: texture_2d<f32>;
@group(0) @binding(3) var texOut: texture_storage_2d<r32float, write>;

fn loadIn(coord: vec2<i32>) -> f32 {
  return textureLoad(texIn, coord, 0).r;
}

fn filterCore(zc: f32, coord: vec2<i32>, dir: vec2<i32>) -> f32 {
  let size = vec2<f32>(textureDimensions(texOut));

  // Eq. 5: world-space kernel size projected to screen space
  let tanHalfFov = -camera.params.y * 0.5;
  let sigma_i = max(1.0, ceil(size.y * filterParams.sigma / (2.0 * zc * tanHalfFov)));
  let radius = min(i32(3.0 * sigma_i), 64);
  let inv2s2 = 1.0 / (2.0 * sigma_i * sigma_i);

  // dynamic range (Eq. 7-9): expand thresholds along the scanline
  var dLow = filterParams.delta;
  var dHigh = filterParams.delta;
  for (var d = 1; d <= radius; d++) {
    let zl = loadIn(coord - d * dir);
    if (zl > 0.0) {
      dLow = max(dLow, zc - zl + filterParams.delta);
    }
    let zr = loadIn(coord + d * dir);
    if (zr > 0.0) {
      dHigh = max(dHigh, zr - zc + filterParams.delta);
    }
  }

  var weightSum = 1.0; // center pixel
  var valueSum = zc;
  for (var d = -radius; d <= radius; d++) {
    if (d == 0) { continue; }
    let zj = loadIn(coord + d * dir);
    if (zj <= 0.0) { continue; }
    // narrow range (Eq. 3): ignore pixels far behind the center
    if (zj > zc + dHigh) { continue; }
    // bias correction (Eq. 6): ignore when the mirrored pixel is occluded
    let zk = loadIn(coord - d * dir);
    if (zk > zc + dHigh) { continue; }
    // clamp near-side depth so curvature at discontinuities is preserved
    let zEff = max(zj, zc - filterParams.mu);
    let w = exp(-f32(d * d) * inv2s2);
    weightSum += w;
    valueSum += w * zEff;
  }
  return valueSum / weightSum;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = vec2<i32>(textureDimensions(texOut));
  let coord = vec2<i32>(gid.xy);
  if (coord.x >= size.x || coord.y >= size.y) { return; }

  let zc = loadIn(coord);
  var out = 0.0;
  if (zc > 0.0) {
    ${horizontal ? 'out = filterCore(zc, coord, vec2<i32>(1, 0));' : 'out = filterCore(zc, coord, vec2<i32>(0, 1));'}
  }
  textureStore(texOut, coord, vec4<f32>(out, 0.0, 0.0, 0.0));
}
`;
}

const FilterHShader = makeFilterShader(true);
const FilterVShader = makeFilterShader(false);

// small fixed-size 2D narrow-range filter to hide separable-filter streaks
const CleanUpShader = /* wgsl */`
${FilterShared}

@group(0) @binding(2) var texIn: texture_2d<f32>;
@group(0) @binding(3) var texOut: texture_storage_2d<r32float, write>;

fn loadIn(coord: vec2<i32>) -> f32 {
  return textureLoad(texIn, coord, 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = vec2<i32>(textureDimensions(texOut));
  let coord = vec2<i32>(gid.xy);
  if (coord.x >= size.x || coord.y >= size.y) { return; }

  let zc = loadIn(coord);
  if (zc <= 0.0) {
    textureStore(texOut, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  let radius = 2; // 5x5 clean-up kernel
  let sigma = 1.0;
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);

  var weightSum = 1.0;
  var valueSum = zc;
  for (var dy = -radius; dy <= radius; dy++) {
    for (var dx = -radius; dx <= radius; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let zj = loadIn(coord + vec2<i32>(dx, dy));
      if (zj <= 0.0) { continue; }
      if (zj > zc + filterParams.delta) { continue; }
      let zEff = max(zj, zc - filterParams.mu);
      let w = exp(-f32(dx * dx + dy * dy) * inv2s2);
      weightSum += w;
      valueSum += w * zEff;
    }
  }
  textureStore(texOut, coord, vec4<f32>(valueSum / weightSum, 0.0, 0.0, 0.0));
}
`;

/* ------------------------------------------------------------------ */
/* Pass 4: screen-space composite (reflection / refraction)            */
/* ------------------------------------------------------------------ */

const CompositeVertexShader = /* wgsl */`
@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0), vec2<f32>(3.0, 1.0), vec2<f32>(-1.0, 1.0)
  );
  // just in front of the far plane (skybox depth = 0) so the fluid is
  // occluded by scene geometry but drawn over the skybox
  return vec4<f32>(pos[vertexIndex], 1e-5, 1.0);
}
`;

const CompositeFragmentShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}
${ShaderFunction.sRGBGammaEncode}

@group(1) @binding(0) var smoothDepth: texture_2d<f32>;

fn reconstructWorld(coord: vec2<i32>, ze: f32) -> vec3<f32> {
  let size = vec2<f32>(textureDimensions(smoothDepth));
  let uv = (vec2<f32>(coord) + 0.5) / size;
  let ndc = uv * 2.0 - 1.0;
  let pView = vec3<f32>(
    ndc.x * ze * camera.params.x * 0.5,
    ndc.y * ze * -camera.params.y * 0.5,
    -ze
  );
  return (camera.viewMatrixInverse * vec4<f32>(pView, 1.0)).xyz;
}

fn loadEye(coord: vec2<i32>, fallback: f32) -> f32 {
  let size = vec2<i32>(textureDimensions(smoothDepth));
  let c = clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
  let z = textureLoad(smoothDepth, c, 0).r;
  return select(fallback, z, z > 0.0);
}

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(fragCoord.xy);
  let ze = textureLoad(smoothDepth, coord, 0).r;
  if (ze <= 0.0) { discard; }

  let p = reconstructWorld(coord, ze);

  // surface normal from smoothed depth neighbors
  let pL = reconstructWorld(coord - vec2<i32>(1, 0), loadEye(coord - vec2<i32>(1, 0), ze));
  let pR = reconstructWorld(coord + vec2<i32>(1, 0), loadEye(coord + vec2<i32>(1, 0), ze));
  let pD = reconstructWorld(coord - vec2<i32>(0, 1), loadEye(coord - vec2<i32>(0, 1), ze));
  let pU = reconstructWorld(coord + vec2<i32>(0, 1), loadEye(coord + vec2<i32>(0, 1), ze));
  var n = normalize(cross(pR - pL, pU - pD));

  let viewDir = normalize(camera.position - p);
  n = select(-n, n, dot(n, viewDir) > 0.0);

  // fresnel (schlick)
  let cosTheta = saturate(dot(n, viewDir));
  let F0 = 0.02;
  let F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);

  // reflection
  let reflDir = reflect(-viewDir, n);
  let refl = textureSampleLevel(envMap, linearSampler, reflDir, 0).rgb;

  // refraction with beer-lambert absorption tint
  var refrDir = refract(-viewDir, n, 1.0 / 1.33);
  if (dot(refrDir, refrDir) < 1e-6) { // total internal reflection
    refrDir = reflDir;
  }
  let refr = textureSampleLevel(envMap, linearSampler, refrDir, 0).rgb;
  let absorption = vec3<f32>(4.0, 0.6, 0.45);
  let opticalPath = 0.15;
  let tint = exp(-absorption * opticalPath);

  // specular highlight from the directional light
  let h = normalize(viewDir + light.direction);
  let spec = pow(saturate(dot(n, h)), 256.0) * light.color;

  let color = refl * F + refr * (1.0 - F) * tint + spec;
  return vec4<f32>(sRGBGammaEncode(color), 1.0);
}
`;

export {
  PrepassVertexShader,
  PrepassFragmentShader,
  ConvertShader,
  FilterHShader,
  FilterVShader,
  CleanUpShader,
  CompositeVertexShader,
  CompositeFragmentShader
};
