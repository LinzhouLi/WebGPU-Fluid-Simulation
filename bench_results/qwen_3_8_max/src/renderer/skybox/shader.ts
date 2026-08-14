import { ShaderStruct, ShaderFunction, ShaderCode } from "../../common/shader";


const VertexShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}

struct VertOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) vPosition: vec3<f32>,
};

@vertex
fn main(@location(0) position: vec3<f32>) -> VertOutput {
  let positionCamera = camera.viewMatrix * vec4<f32>(position, 0.0);
  let positionNDC = camera.projectionMatrix * vec4<f32>(positionCamera.xyz, 1.0);
  let positionReverseZ = vec4<f32>(positionNDC.xy, 0.0, positionNDC.w);
  return VertOutput(positionReverseZ, position);
}
`;


const FragmentShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}
${ShaderFunction.sRGBGammaEncode}

@fragment
fn main(
  @builtin(position) position : vec4<f32>,
  @location(0) fragPosition : vec3<f32>,
) -> @location(0) vec4<f32> {
  let color_linear = textureSampleLevel(envMap, linearSampler, fragPosition, 0);
  return vec4<f32>(sRGBGammaEncode(color_linear.xyz), 1.0);
}
`;

export { VertexShader, FragmentShader };