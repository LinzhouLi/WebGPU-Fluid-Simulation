import { ShaderStruct, ShaderCode } from '../../common/shader';

const vertexShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderCode.GlobalGroup}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) centerView: vec3<f32>,
  @location(1) corner: vec2<f32>,
};

@group(1) @binding(0) var<storage> particlePositions: array<vec3<f32>>;
@group(1) @binding(1) var<uniform> particleRadius: f32;

@vertex
fn main(
  @builtin(vertex_index) vertexIdx: u32,
  @builtin(instance_index) instanceIdx: u32
) -> VertexOutput {
  let particlePos = particlePositions[instanceIdx];
  let centerView = (camera.viewMatrix * vec4<f32>(particlePos, 1.0)).xyz;

  // Generate corner offsets from vertex_index (0-5 for two triangles)
  let cornerX = select(-1.0, 1.0, vertexIdx == 1u || vertexIdx == 2u || vertexIdx == 4u);
  let cornerY = select(-1.0, 1.0, vertexIdx == 2u || vertexIdx == 4u || vertexIdx == 5u);

  let r = particleRadius;
  var viewPos = centerView;
  viewPos.x += cornerX * r;
  viewPos.y += cornerY * r;

  let clipPos = camera.projectionMatrix * vec4<f32>(viewPos, 1.0);

  return VertexOutput(clipPos, centerView, vec2<f32>(cornerX, cornerY));
}
`;

const fragmentShader = /* wgsl */`
${ShaderStruct.Camera}
@group(0) @binding(0) var<uniform> camera: Camera;

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) centerView: vec3<f32>,
  @location(1) corner: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) depth: f32,
  @location(0) storedDepth: f32,
};

@group(1) @binding(1) var<uniform> particleRadius: f32;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {
  // Discard fragments outside the projected sphere
  let dist2 = input.corner.x * input.corner.x + input.corner.y * input.corner.y;
  if (dist2 > 1.01) {
    discard;
  }

  let r = particleRadius;

  // Compute sphere surface depth at this fragment
  let dz = r * sqrt(max(0.0, 1.0 - dist2));
  let surfaceViewZ = input.centerView.z + dz;

  // Convert view-space Z to reverse-Z NDC depth
  let eyeDepth = -surfaceViewZ;
  let d = (1.0 / eyeDepth - camera.params.w) / camera.params.z;

  return FragmentOutput(d, d);
}
`;

export { vertexShader, fragmentShader };
