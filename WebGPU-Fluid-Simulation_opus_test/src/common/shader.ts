
const Camera = /* wgsl */`
struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};
`;

const DirectionalLight = /* wgsl */`
struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>
};
`;

const Transform = /* wgsl */`
struct Transform {
  modelMatrix: mat4x4<f32>,
  normalMatrix: mat3x3<f32>
};
`;

const MeshMaterial = /* wgsl */`
struct MeshMaterial {
  shininess: f32,
  color: vec3<f32>
};
`;

const SimulationOptions = /* wgsl */`
struct SimulationOptions {
  particleCount: u32,
  XSPHCoef: f32,
  vorticityCoef: f32,
  tensionCoef: f32,
  gravity: vec3<f32>
}
`;

// Screen space fluid rendering options.
// Keep the memory layout in sync with FluidRenderer.setConfig().
const FluidOptions = /* wgsl */`
struct FluidOptions {
  particleRadius: f32,    // world space radius of the sphere imposters
  filterSigma: f32,       // world space filter size, sigma of Eq.5
  filterDelta: f32,       // eye space depth threshold, delta of Eq.2/3
  filterMu: f32,          // eye space depth clamping offset, mu of Eq.2
  maxFilterSigma: f32,    // upper bound of the screen space kernel size (pixel)
  screenHeight: f32,      // vertical resolution, H of Eq.5
  tanHalfFov: f32,        // tan(alpha / 2) of Eq.5
  ior: f32,               // index of refraction
  fluidColor: vec3<f32>,
  absorption: f32,        // Beer-Lambert absorption coefficient
  opacity: f32            // thickness to alpha coefficient
}
`;

const ShaderStruct = { Camera, DirectionalLight, Transform, MeshMaterial, SimulationOptions, FluidOptions };


const sRGBGammaEncode = /* wgsl */`
fn sRGBGammaEncode(color: vec3<f32>) -> vec3<f32> {
  return mix(
    color.rgb * 12.92,                                    // x <= 0.0031308
    pow(color.rgb, vec3<f32>(0.41666)) * 1.055 - 0.055,   // x >  0.0031308
    saturate(sign(color.rgb - 0.0031308))
  );
}
`;

const ShaderFunction = { sRGBGammaEncode };


const GlobalGroup = /* wgsl */`
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: DirectionalLight;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var envMap: texture_cube<f32>;
`;

// Sphere imposter (billboard) helpers, shared by the raw particle renderer and
// the screen space fluid renderer. Drawn as a 4 vertex triangle strip per
// instance, so the quad corner is derived from the vertex index directly.
const SphereImposter = /* wgsl */`
fn imposterCorner(vertexIndex: u32) -> vec2<f32> {
  return vec2<f32>(
    select(-1.0, 1.0, (vertexIndex & 1u) == 1u),
    select(-1.0, 1.0, (vertexIndex & 2u) == 2u)
  );
}

// View space aligned quad that covers the perspective projection of the sphere.
// The half extent is the tangent cone radius rather than the sphere radius,
// otherwise spheres close to the camera get clipped by their own quad.
fn imposterPosition(centerView: vec3<f32>, radius: f32, corner: vec2<f32>) -> vec3<f32> {
  let distance = max(-centerView.z, radius * 1.01);
  let halfExtent = radius * distance * inverseSqrt(max(distance * distance - radius * radius, 1e-12));
  return centerView + vec3<f32>(corner * halfExtent, 0.0);
}

// Sphere surface point of the imposter, using the orthographic approximation of
// the sphere projection (the accepted projection error): the view ray is assumed
// to be parallel to the view axis, so the normal follows directly from the
// position inside the unit circle.
// max() guards the square root: discard demotes the invocation to a helper but
// does not stop it, so radiusSqr can still be greater than one here.
fn imposterSurfaceNormal(corner: vec2<f32>, radiusSqr: f32) -> vec3<f32> {
  return vec3<f32>(corner, sqrt(max(1.0 - radiusSqr, 0.0)));
}
`;

const ShaderCode = { GlobalGroup, SphereImposter };


export { ShaderStruct, ShaderFunction, ShaderCode };
