import { wgsl } from '../../3rd-party/wgsl-preprocessor';

const vertexShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>
};

struct Material {
  sphereRadius: f32,
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct VertexInput {
  @builtin(instance_index) instanceIndex: u32,
  @location(0) position: vec3<f32>,
  @location(1) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<storage> instancePositions: array<vec3<f32>>;

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let centerPositonCam = camera.viewMatrix * vec4<f32>(instancePositions[input.instanceIndex], 1.0);
  let positionCam = centerPositonCam + vec4<f32>(input.position * material.sphereRadius, 0.0);
  let positionScreen = camera.projectionMatrix * positionCam;
  return VertexOutput(
    positionScreen, positionCam, input.uv
  );
}

`;


const fragmentShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>
};

struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>
};

struct Material {
  sphereRadius: f32,
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) depth: f32,
  @location(0) color: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

fn sRGBGammaEncode(color: vec3<f32>) -> vec3<f32> {
  return mix(
    color.rgb * 12.92,                                    // x <= 0.0031308
    pow(color.rgb, vec3<f32>(0.41666)) * 1.055 - 0.055,   // x >  0.0031308
    saturate(sign(color.rgb - 0.0031308))
  );
}

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  // caculate normal from uv
  var normalCam = vec4<f32>(input.vUv * 2.0 - 1.0, 0.0, 0.0);
  let radius2 = dot(normalCam.xy, normalCam.xy);
  if (radius2 > 1.0) { discard; }
  normalCam.z = sqrt(1.0 - radius2);
  let normalWorld = camera.viewMatrixInverse * normalCam; // camera space normal -> world space normal

  // caculate depth
  let fragPosCam = input.vPositionCam + normalCam * material.sphereRadius;
  let positionClip = camera.projectionMatrix * fragPosCam;
  let depth = positionClip.z / positionClip.w;

  // simple diffuse shading
  let NoL = saturate(dot(normalize(normalWorld.xyz), light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 * material.color; // RECIPROCAL_PI

  return FragmentOutput(
    depth, vec4<f32>(sRGBGammaEncode(diffuse), 1.0)
  );
}

`;

export { vertexShader, fragmentShader }