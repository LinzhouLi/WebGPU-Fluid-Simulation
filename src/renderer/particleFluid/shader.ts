import { wgsl } from '../../3rd-party/wgsl-preprocessor';
import { ShaderStruct, ShaderFunction } from "../../common/shader";

const vertexShader = wgsl/* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.SphereMaterial}

struct VertexInput {
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: SphereMaterial;
@group(0) @binding(2) var<storage> instancePositions: array<vec3<f32>>;

const positions = array<vec2<f32>, 4>(
  vec2<f32>(-0.5, -0.5), // Bottom Left
  vec2<f32>( 0.5, -0.5), // Bottom Right
  vec2<f32>(-0.5,  0.5), // Top Left
  vec2<f32>( 0.5,  0.5)  // Top Right
);

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let position = positions[input.vertexIndex];
  let uv = position + 0.5;
  let centerPositonCam = camera.viewMatrix * vec4<f32>(instancePositions[input.instanceIndex], 1.0);
  let positionCam = centerPositonCam + vec4<f32>(position * material.sphereRadius, 0.0, 0.0);
  let positionScreen = camera.projectionMatrix * positionCam;
  return VertexOutput(
    positionScreen, positionCam, uv
  );
}
`;


const fragmentShader = wgsl/* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.SphereMaterial}

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) depth: f32,
  @location(0) color: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: SphereMaterial;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

${ShaderFunction.sRGBGammaEncode}

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