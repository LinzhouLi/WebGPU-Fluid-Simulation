import { ShaderStruct } from "../../../common/shader";

const NarrowRangeFilter = /* wgsl */`
  // Is depth too low? Keep filter symmetric and early out for both opposing values.
  if (any(sampleValue < vec2<f32>(thresholdLow))) { continue; }

  // Is depth too high? Clamp to upper bound.
  sampleValue = select(sampleValue, vec2<f32>(valueBoundHigh), sampleValue > vec2<f32>(thresholdHigh));

  // Dynamic depth range.
  thresholdLow = min(thresholdLow, min(sampleValue.x, sampleValue.y) - threshold);
  thresholdHigh = max(thresholdHigh, max(sampleValue.x, sampleValue.y) + threshold);

  // Add sample values
  weightSum += gaussianWeight * 2.0;
  valueSum += (sampleValue.x + sampleValue.y) * gaussianWeight;
`;


const computeShader = /* wgsl */`
const MaxFilterSize: u32 = 32;
const HalfMaxFilterSize: u32 = MaxFilterSize >> 1;
const SharedDataSize: u32 = MaxFilterSize << 1;

${ShaderStruct.RenderingOptions}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var destTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> filterDirection: vec2<i32>;
@group(0) @binding(3) var<uniform> options: RenderingOptions;

var<workgroup> sharedBuffer: array<f32, SharedDataSize>;

@compute @workgroup_size(MaxFilterSize, 1, 1)
fn main(
  @builtin(local_invocation_index) thread_id: u32,
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) block_id: vec3<u32>
) {

  let HalfFilterSize: u32 = options.filterSize >> 1;
  let Sigma: f32 = f32(HalfFilterSize) * 0.5;
  let GaussianK: f32 = 0.5 / (Sigma * Sigma);
  let threshold: f32 = options.radius * 5.0;
  let mu: f32 = options.radius;

  // preload to shared buffer
  let texCoord = 
    i32(block_id.x * MaxFilterSize + thread_id) * filterDirection + 
    i32(block_id.y) * (vec2<i32>(1) - filterDirection);
  sharedBuffer[thread_id] = textureLoad(srcTexture, texCoord - i32(HalfMaxFilterSize) * filterDirection, 0).r;
  sharedBuffer[thread_id + MaxFilterSize] = textureLoad(srcTexture, texCoord + i32(HalfMaxFilterSize) * filterDirection, 0).r;
  workgroupBarrier();

  // filter
  let sharedBufferCenterIndex = thread_id + HalfFilterSize;
  let centerSampleValue = sharedBuffer[sharedBufferCenterIndex];
  if (centerSampleValue > 0.0) {
    var valueSum: f32 = centerSampleValue;
    var weightSum: f32 = 1.0;

    var sampleValue: vec2<f32>; 
    var gaussianWeight: f32;
    var thresholdLow = centerSampleValue - threshold;
    var thresholdHigh = centerSampleValue + threshold;
    let valueBoundHigh = centerSampleValue + mu;
    for (var i: u32 = 1; i <= HalfFilterSize; i++) {
      sampleValue.x = sharedBuffer[sharedBufferCenterIndex - i];
      sampleValue.y = sharedBuffer[sharedBufferCenterIndex + i];

      gaussianWeight = exp(- f32(i * i) * GaussianK);
      ${NarrowRangeFilter}
    }

    valueSum /= weightSum;
    textureStore(destTexture, texCoord, vec4<f32>(valueSum));
  }
  return;

}
`;

export { computeShader };


// // spatial domain
// r = vec2<f32>(f32(i) * 0.01);
// sampleWeight = exp(-r * r);

// // range domain;
// r = (sampleValue - centerSampleValue) * 2.0;
// sampleWeight = sampleWeight * exp(-r * r);

// sampleWeight = select(sampleWeight, zeroVec2, sampleValue == zeroVec2);
// sampleValue *= sampleWeight;

// valueSum += sampleValue.x + sampleValue.y;
// weightSum += sampleWeight.x + sampleWeight.y;