import { ShaderStruct, ShaderCode } from '../../common/shader';

const vertexShader = /* wgsl */`
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) vertexIdx: u32) -> VertexOutput {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0,  3.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
  );
  var uv  = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0),
  );

  let p = pos[vertexIdx];
  return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv[vertexIdx]);
}
`;

const fragmentShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderCode.GlobalGroup}

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(1) @binding(0) var fluidDepthTexture: texture_2d<f32>;
@group(1) @binding(1) var pointSampler: sampler;
@group(1) @binding(2) var sceneColorTexture: texture_2d<f32>;
@group(1) @binding(3) var sceneDepthTexture: texture_depth_2d;
@group(1) @binding(4) var<uniform> surfaceParams: vec4<f32>; // xy: texelSize, z: eta (n1/n2), w: unused
@group(1) @binding(5) var<uniform> invProj: mat4x4<f32>;

fn viewPosFromDepth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  // Reconstruct view-space position from UV and reverse-Z depth
  let clipX = uv.x * 2.0 - 1.0;
  let clipY = (1.0 - uv.y) * 2.0 - 1.0;
  let clipPos = vec4<f32>(clipX, clipY, depth, 1.0);
  let viewPos4 = invProj * clipPos;
  return viewPos4.xyz / viewPos4.w;
}

fn computeNormal(viewPos: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let texelSize = surfaceParams.xy;

  // Sample depths at neighboring pixels
  let dU = textureSampleLevel(fluidDepthTexture, pointSampler, uv + vec2<f32>( texelSize.x, 0.0), 0.0).r;
  let dD = textureSampleLevel(fluidDepthTexture, pointSampler, uv + vec2<f32>(-texelSize.x, 0.0), 0.0).r;
  let dL = textureSampleLevel(fluidDepthTexture, pointSampler, uv + vec2<f32>(0.0,  texelSize.y), 0.0).r;
  let dR = textureSampleLevel(fluidDepthTexture, pointSampler, uv + vec2<f32>(0.0, -texelSize.y), 0.0).r;

  // For boundary pixels, fall back to center position
  let eps = 0.0001;
  let posR = select(viewPosFromDepth(uv + vec2<f32>( texelSize.x, 0.0), dU), viewPos, dU < eps);
  let posL = select(viewPosFromDepth(uv + vec2<f32>(-texelSize.x, 0.0), dD), viewPos, dD < eps);
  let posU = select(viewPosFromDepth(uv + vec2<f32>(0.0,  texelSize.y), dL), viewPos, dL < eps);
  let posD = select(viewPosFromDepth(uv + vec2<f32>(0.0, -texelSize.y), dR), viewPos, dR < eps);

  let dp_dx = posR - posL;
  let dp_dy = posU - posD;

  let n = normalize(cross(dp_dx, dp_dy));

  // Ensure normal faces the camera
  let viewDir = normalize(viewPos);
  return select(n, -n, dot(n, viewDir) < 0.0);
}

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
  let sceneColor = textureSampleLevel(sceneColorTexture, pointSampler, input.uv, 0.0);
  let fluidDepth = textureSampleLevel(fluidDepthTexture, pointSampler, input.uv, 0.0).r;

  // No fluid at this pixel
  if (fluidDepth < 0.0001) {
    return sceneColor;
  }

  // Scene object in front of fluid
  let sceneDepth = textureSampleLevel(sceneDepthTexture, pointSampler, input.uv, 0.0);
  if (sceneDepth > 0.0001 && fluidDepth < sceneDepth) {
    return sceneColor;
  }

  // Reconstruct view-space position
  let viewPos = viewPosFromDepth(input.uv, fluidDepth);

  // Compute normal from depth gradient
  let N_view = computeNormal(viewPos, input.uv);

  // View direction in view space
  let V = normalize(-viewPos);

  // Transform to world space for env map sampling
  let N_world = normalize((camera.viewMatrixInverse * vec4<f32>(N_view, 0.0)).xyz);
  let V_world = normalize((camera.viewMatrixInverse * vec4<f32>(V, 0.0)).xyz);

  // Fresnel (Schlick)
  let f0 = 0.04;
  let cosI = abs(dot(N_world, V_world));
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosI, 5.0);

  // Reflection
  let R = reflect(-V_world, N_world);
  let reflectionColor = textureSampleLevel(envMap, linearSampler, R, 0.0).rgb;

  // Refraction (env map as approximation of background)
  let eta = surfaceParams.z; // n1/n2 ≈ 0.752 for air→water
  let sinT2 = eta * eta * (1.0 - cosI * cosI);
  var fluidColor = reflectionColor;

  if (sinT2 < 1.0) {
    let cosT = sqrt(1.0 - sinT2);
    let T = normalize(eta * (-V_world) + (eta * cosI - cosT) * N_world);
    let refractionColor = textureSampleLevel(envMap, linearSampler, T, 0.0).rgb;
    fluidColor = mix(refractionColor, reflectionColor, vec3<f32>(fresnel));
  }

  // Specular highlight from light direction
  let lightDir = normalize(light.direction);
  let H = normalize(V_world + lightDir);
  let spec = pow(max(dot(N_world, H), 0.0), 256.0) * 0.3 * light.color;

  return vec4<f32>(fluidColor + spec, 1.0);
}
`;

export { vertexShader, fragmentShader };
