
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

const SphereMaterial = /* wgsl */`
struct SphereMaterial {
  sphereRadius: f32,
  color: vec3<f32>
};
`;

const RenderingOptions = /* wgsl */`
struct RenderingOptions {
  mode: u32,
  filterSize: u32,
  radius: f32,
  tickness: f32,
  tintColor: vec3<f32>
}
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

const ShaderStruct = { Camera, DirectionalLight, Transform, MeshMaterial, SphereMaterial, RenderingOptions, SimulationOptions };


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

const ShaderCode = { GlobalGroup };


export { ShaderStruct, ShaderFunction, ShaderCode };