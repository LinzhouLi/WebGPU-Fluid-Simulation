import { ShaderStruct, ShaderCode } from '../../common/shader';

// Raw particle view: the particles are drawn as sphere imposters (screen space
// circles) instead of sphere meshes, which keeps the debug view cheap. The
// perspective projection of a sphere is approximated by a circle, and the sphere
// surface is reconstructed in the fragment stage.

const vertexShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}
${ShaderCode.SphereImposter}

struct Material {
  color: vec4<f32>,
  radius: f32
};

@group(1) @binding(0) var<storage, read> particlePosition: array<vec3<f32>>;
@group(1) @binding(1) var<uniform> material: Material;

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
  let positionView = imposterPosition(centerView, material.radius, corner);
  return VertexOutput(
    camera.projectionMatrix * vec4<f32>(positionView, 1.0),
    corner, centerView
  );
}

`;


const fragmentShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderCode.GlobalGroup}
${ShaderCode.SphereImposter}

struct Material {
  color: vec4<f32>,
  radius: f32
};

@group(1) @binding(1) var<uniform> material: Material;

struct FragmentInput {
  @location(0) corner: vec2<f32>,
  @location(1) centerView: vec3<f32>
};

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32
};

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  let radiusSqr = dot(input.corner, input.corner);
  if (radiusSqr > 1.0) { discard; }

  let normalView = imposterSurfaceNormal(input.corner, radiusSqr);
  let surfaceView = input.centerView + normalView * material.radius;

  // Reverse Z, see the patched projection matrix in controller.ts
  let positionClip = camera.projectionMatrix * vec4<f32>(surfaceView, 1.0);

  let normal = normalize((camera.viewMatrixInverse * vec4<f32>(normalView, 0.0)).xyz);
  let NoL = saturate(dot(normal, light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 * material.color.rgb; // RECIPROCAL_PI

  return FragmentOutput(
    vec4<f32>(pow(diffuse, vec3<f32>(0.454545)), 1.0),
    positionClip.z / positionClip.w
  );

}

`;

export { vertexShader, fragmentShader }
