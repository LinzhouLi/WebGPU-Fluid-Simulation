import { ShaderStruct } from "../../../common/shaderStruct";

const depthPassfragmentShader = /* wgsl */`

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.SphereMaterial}

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) frag_depth: f32,
  @location(0) depthCam: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: SphereMaterial;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  // caculate normal from uv
  var normalCam = vec4<f32>(input.vUv * 2.0 - 1.0, 0.0, 0.0);
  let radius2 = dot(normalCam.xy, normalCam.xy);
  if (radius2 > 1.0) { discard; }
  normalCam.z = sqrt(1.0 - radius2);

  // caculate depth
  let positionCam = input.vPositionCam + normalCam * material.sphereRadius;
  let positionClip = camera.projectionMatrix * positionCam;
  let depth = positionClip.z / positionClip.w;
  let depthCam = -positionCam.z / positionCam.w;

  return FragmentOutput( depth, vec4<f32>(depthCam) );

}

`;

export { depthPassfragmentShader }