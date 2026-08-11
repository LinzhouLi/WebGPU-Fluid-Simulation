import { ShaderStruct, ShaderCode } from '../../common/shader';

// Sphere imposter passes of the screen space fluid renderer.
//
// Both passes rasterize the particles as screen space circles and share the same
// vertex stage. The depth pass writes the (negative) eye space depth of the
// sphere surface, which is the input of the narrow-range filter. The thickness
// pass accumulates the length of the ray inside each sphere with additive
// blending, which drives absorption and opacity during compositing.

const VertexCommon = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.FluidOptions}
${ShaderCode.SphereImposter}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> particlePosition: array<vec3<f32>>;
@group(1) @binding(1) var<uniform> options: FluidOptions;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) centerView: vec3<f32>
};

@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let corner = imposterCorner(vertexIndex);
  let centerView = (camera.viewMatrix * vec4<f32>(particlePosition[instanceIndex], 1.0)).xyz;
  let positionView = imposterPosition(centerView, options.particleRadius, corner);
  return VertexOutput(
    camera.projectionMatrix * vec4<f32>(positionView, 1.0),
    corner, centerView
  );
}
`;


const DepthShader = /* wgsl */`
${VertexCommon}

struct FragmentOutput {
  @location(0) depth: f32,              // negative eye space depth, 0 means no fluid
  @builtin(frag_depth) fragDepth: f32   // reverse Z device depth
};

@fragment
fn depth(input: VertexOutput) -> FragmentOutput {
  let radiusSqr = dot(input.corner, input.corner);
  if (radiusSqr > 1.0) { discard; }

  let normalView = imposterSurfaceNormal(input.corner, radiusSqr);
  let surfaceView = input.centerView + normalView * options.particleRadius;

  // Reverse Z: the projection matrix is patched in controller.ts, w = -z_eye > 0.
  let positionClip = camera.projectionMatrix * vec4<f32>(surfaceView, 1.0);

  return FragmentOutput(surfaceView.z, positionClip.z / positionClip.w);
}
`;


const ThicknessShader = /* wgsl */`
${VertexCommon}

@fragment
fn thickness(input: VertexOutput) -> @location(0) f32 {
  let radiusSqr = dot(input.corner, input.corner);
  if (radiusSqr > 1.0) { discard; }

  // Chord length of the view ray through the sphere.
  return 2.0 * options.particleRadius * sqrt(max(1.0 - radiusSqr, 0.0));
}
`;

export { DepthShader, ThicknessShader };
