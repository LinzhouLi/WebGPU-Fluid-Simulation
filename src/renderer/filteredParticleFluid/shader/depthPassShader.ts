import { ShaderStruct } from "../../../common/shader";

const depthPassVertexShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.RenderingOptions}

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
@group(0) @binding(1) var<uniform> options: RenderingOptions;
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
  let positionCam = centerPositonCam + vec4<f32>(position * options.radius, 0.0, 0.0);
  let positionScreen = camera.projectionMatrix * positionCam;
  return VertexOutput(
    positionScreen, positionCam, uv
  );
}
`;

const depthPassfragmentShader = /* wgsl */`

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.RenderingOptions}

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) frag_depth: f32,
  @location(0) depthCam: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> options: RenderingOptions;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  // caculate normal from uv
  var normalCam = vec4<f32>(input.vUv * 2.0 - 1.0, 0.0, 0.0);
  let radius2 = dot(normalCam.xy, normalCam.xy);
  if (radius2 > 1.0) { discard; }
  normalCam.z = sqrt(1.0 - radius2);

  // caculate depth
  let positionCam = input.vPositionCam + normalCam * options.radius;
  let positionClip = camera.projectionMatrix * positionCam;
  let depth = positionClip.z / positionClip.w;
  let depthCam = -positionCam.z / positionCam.w;

  return FragmentOutput( depth, vec4<f32>(depthCam) );

}

`;

export { depthPassVertexShader, depthPassfragmentShader }