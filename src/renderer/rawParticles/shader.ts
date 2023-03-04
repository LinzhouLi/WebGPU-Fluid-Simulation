import { wgsl } from '../../3rd-party/wgsl-preprocessor';

const vertexShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};

struct Material {
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct VertexInput {
  @builtin(instance_index) instanceIndex: u32,
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective, center) vPosition: vec3<f32>,
  @location(1) @interpolate(perspective, center) vNormal: vec3<f32>,
  @location(2) @interpolate(perspective, center) vUv: vec2<f32>
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<storage> instancePositions: array<vec3<f32>>;

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let positionWorld = input.position +  instancePositions[input.instanceIndex];
  // let positionWorld = input.position;
  let positionScreen = camera.projectionMatrix * camera.viewMatrix * vec4<f32>(positionWorld, 1.0);
  return VertexOutput(
    positionScreen, positionWorld,
    input.normal, input.uv
  );
}

`;


const fragmentShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};

struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>
};

struct Material {
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPosition: vec3<f32>,
  @location(1) @interpolate(perspective, center) vNormal: vec3<f32>,
  @location(2) @interpolate(perspective, center) vUv: vec2<f32>
};

@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
  let NoL = saturate(dot(normalize(input.vNormal), light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 * material.color; // RECIPROCAL_PI
  return vec4<f32>(pow(diffuse, vec3<f32>(0.454545)), 1.0);
}

`;

export { vertexShader, fragmentShader }