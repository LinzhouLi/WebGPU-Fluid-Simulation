import { ShaderStruct, ShaderFunction, ShaderCode } from "../../common/shader";

const VertexShader = /* wgsl */`
struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(perspective, center) vPosition: vec3<f32>,
  @location(1) @interpolate(perspective, center) vNormal: vec3<f32>,
  @location(2) @interpolate(perspective, center) vUv: vec2<f32>,
};

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.Transform}
${ShaderCode.GlobalGroup}
@group(1) @binding(0) var<uniform> transform: Transform;

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let positionObject = vec4<f32>(input.position, 1.0);
  let normalObject = input.normal;
  let normalWorld = transform.normalMatrix * normalObject;
  let positionWord = transform.modelMatrix * positionObject;
  let positionScreen = camera.projectionMatrix * camera.viewMatrix * positionWord;
  return VertexOutput(
    positionScreen,
    positionWord.xyz,
    normalWorld,
    input.uv
  );
}
`;

const FragmentShader = /* wgsl */`
struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPosition: vec3<f32>,
  @location(1) @interpolate(perspective, center) vNormal: vec3<f32>,
  @location(2) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @location(0) color: vec4<f32>
};

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.MeshMaterial}
${ShaderCode.GlobalGroup}
@group(1) @binding(1) var<uniform> material: MeshMaterial;
// @group(1) @binding(2) var baseMap: texture_2d<f32>; // todo

${ShaderFunction.sRGBGammaEncode}

@fragment
fn main(input: FragmentInput) -> FragmentOutput {
  // simple diffuse shading
  let NoL = saturate(dot(normalize(input.vNormal), light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 * material.color; // RECIPROCAL_PI

  return FragmentOutput(
    vec4<f32>(sRGBGammaEncode(diffuse), 1.0)
  );
}
`;

export { VertexShader, FragmentShader };