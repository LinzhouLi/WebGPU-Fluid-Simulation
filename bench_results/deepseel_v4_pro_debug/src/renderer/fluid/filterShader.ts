const fullScreenVertexShader = /* wgsl */`
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) vertexIdx: u32) -> VertexOutput {
  // Full-screen triangle: 3 vertices cover clip space [-1,1]² at z=0
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0,  3.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
  );
  var uv  = array<vec2<f32>, 3>(
    vec2<f32>(0.0, -1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
  );

  let p = pos[vertexIdx];
  return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv[vertexIdx]);
}
`;

const filterFragmentShader = /* wgsl */`
struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct FilterParams {
  texelSize: vec2<f32>,   // offset 0,  size 8
  isHorizontal: f32,       // offset 8,  size 4
  sigmaRange: f32,         // offset 12, size 4
  kernelRadius: f32,       // offset 16, size 4
  _pad: f32,               // offset 20, size 4 (struct rounds up to 24 for vec2<f32> alignment)
};

@group(0) @binding(0) var sourceDepth: texture_2d<f32>;
@group(0) @binding(1) var pointSampler: sampler;
@group(0) @binding(2) var<uniform> params: FilterParams;

@fragment
fn main(input: FragmentInput) -> @location(0) f32 {
  let centerDepth = textureSample(sourceDepth, pointSampler, input.uv).r;

  // If no fluid at this pixel (far plane / zero depth), pass through
  if (centerDepth <= 0.0001) {
    return centerDepth;
  }

  let sigmaSpatial = 2.0; // spatial sigma in pixels
  let sigmaR = params.sigmaRange;
  let kRadius = i32(params.kernelRadius); // cast f32 → i32 for loop
  let twoSigmaS2 = 2.0 * sigmaSpatial * sigmaSpatial;
  let twoSigmaR2 = 2.0 * sigmaR * sigmaR;

  var weightedSum = 0.0;
  var weightSum = 0.0;

  // Sample in the filter direction
  var step = vec2<f32>(0.0, 0.0);
  if (params.isHorizontal > 0.5) {
    step = vec2<f32>(params.texelSize.x, 0.0);
  } else {
    step = vec2<f32>(0.0, params.texelSize.y);
  }

  for (var i = -kRadius; i <= kRadius; i++) {
    let offset = f32(i);
    let sampleCoord = input.uv + step * offset;
    let sampleDepth = textureSampleLevel(sourceDepth, pointSampler, sampleCoord, 0.0).r;

    // Skip far-plane samples
    if (sampleDepth <= 0.0001) {
      continue;
    }

    let spatialW = exp(-offset * offset / twoSigmaS2);
    let depthDiff = abs(sampleDepth - centerDepth);
    let rangeW = exp(-depthDiff * depthDiff / twoSigmaR2);
    let w = spatialW * rangeW;

    weightedSum += sampleDepth * w;
    weightSum += w;
  }

  if (weightSum < 0.0001) {
    return centerDepth;
  }

  return weightedSum / weightSum;
}
`;

export { fullScreenVertexShader, filterFragmentShader };
