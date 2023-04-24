
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

const SphereMaterial = /* wgsl */`
struct SphereMaterial {
  sphereRadius: f32,
  color: vec3<f32>
};
`;

const ShaderStruct = { Camera, DirectionalLight, SphereMaterial };

export { ShaderStruct };