import { ShaderStruct } from '../../common/shader';

// Narrow-range filter of
//   Truong & Yuksel, "A Narrow-Range Filter for Screen-Space Fluid Rendering",
//   i3D 2018.
//
// The filtered value is
//   z'_i = sum_j( w_ij * f(z_i, z_j) ) / sum_j( w_ij )                    (Eq.1)
// with the clamping function
//   f(z_i, z_j) = z_j                 if z_j >= z_i - delta_low           (Eq.2)
//                 z_i - mu            otherwise
// and the weights
//   w_ij = 0                          if z_j or its opposite pixel z_k
//                                     is closer than z_i + delta_high     (Eq.3/6)
//          G(p_i, p_j, sigma_i)       otherwise
//
// Depth values are negative eye space z (smaller means further away), so a value
// of 0 marks a pixel without fluid: it is never a valid depth and is handled by
// an explicit branch everywhere (it would compare as the closest possible value).

const FilterCommon = /* wgsl */`
${ShaderStruct.FluidOptions}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> options: FluidOptions;

struct FilterState {
  sum: f32,
  weight: f32,
  deltaLow: f32,
  deltaHigh: f32
};

fn isFluid(depth: f32) -> bool {
  return depth < 0.0;
}

fn loadDepth(coord: vec2<i32>, size: vec2<i32>) -> f32 {
  // Out of screen counts as background, which stops the kernel just like a
  // real background pixel does.
  if (any(coord < vec2<i32>(0)) || any(coord >= size)) { return 0.0; }
  return textureLoad(inputTexture, coord, 0).x;
}

// Eq.5: the world space filter size is projected to a screen space kernel size,
// so the amount of smoothing is independent of the camera distance.
fn screenKernelSigma(depth: f32) -> f32 {
  let sigma = ceil(
    options.screenHeight * options.filterSigma /
    (2.0 * abs(depth) * options.tanHalfFov)
  );
  return clamp(sigma, 1.0, options.maxFilterSigma);
}

// Accumulates a symmetric pair of neighbors. Both must be fluid pixels.
fn accumulatePair(
  stateIn: FilterState, center: f32,
  depthPositive: f32, depthNegative: f32, gaussian: f32,
  delta: f32, mu: f32
) -> FilterState {
  var state = stateIn;

  // Eq.6 bias correction: a pair is used together or dropped together, so the
  // kernel stays symmetric and background surfaces are not bent near occluders.
  if (depthPositive > center + state.deltaHigh || depthNegative > center + state.deltaHigh) {
    return state;
  }

  // Eq.2 clamping: neighbors behind the permitted range are clamped instead of
  // ignored, which is what produces curved edges rather than flattened ones.
  state.sum += gaussian * select(center - mu, depthPositive, depthPositive >= center - state.deltaLow);
  state.sum += gaussian * select(center - mu, depthNegative, depthNegative >= center - state.deltaLow);
  state.weight += 2.0 * gaussian;

  // Eq.8/9 dynamic range: neighbors already inside the accepted range widen it
  // for the pixels further out, which keeps flat surfaces smooth at grazing
  // angles. Callers walk from the closest pixels outwards.
  if (depthPositive >= center - state.deltaLow && depthPositive <= center + state.deltaHigh) {
    state.deltaLow = max(state.deltaLow, center - depthPositive + delta);
    state.deltaHigh = max(state.deltaHigh, depthPositive - center + delta);
  }
  if (depthNegative >= center - state.deltaLow && depthNegative <= center + state.deltaHigh) {
    state.deltaLow = max(state.deltaLow, center - depthNegative + delta);
    state.deltaHigh = max(state.deltaHigh, depthNegative - center + delta);
  }

  return state;
}
`;


// Separable approximation (paper section 3.4): the filter is applied as 1D
// passes with alternating directions.
const FilterShader1D = /* wgsl */`
${FilterCommon}

override FilterDirectionX: i32 = 1;
override FilterDirectionY: i32 = 0;
override MaxKernelRadius: i32 = 32;

@compute @workgroup_size(8, 8, 1)
fn main( @builtin(global_invocation_id) globalId: vec3<u32> ) {

  let size = vec2<i32>(textureDimensions(inputTexture));
  let coord = vec2<i32>(globalId.xy);
  if (any(coord >= size)) { return; }

  let center = textureLoad(inputTexture, coord, 0).x;
  if (!isFluid(center)) {
    textureStore(outputTexture, coord, vec4<f32>(0.0));
    return;
  }

  let sigma = screenKernelSigma(center);
  let kernelRadius = min(i32(3.0 * sigma), MaxKernelRadius);
  let twoSigmaSqr = 2.0 * sigma * sigma;
  let direction = vec2<i32>(FilterDirectionX, FilterDirectionY);

  var state = FilterState(center, 1.0, options.filterDelta, options.filterDelta);

  for (var offset = 1; offset <= kernelRadius; offset++) {

    let depthPositive = loadDepth(coord + direction * offset, size);
    let depthNegative = loadDepth(coord - direction * offset, size);

    // A background pixel on either side ends the kernel: reaching across a gap
    // would blend in a surface that is not connected to this one, and stopping
    // keeps the kernel symmetric for the bias correction.
    if (!isFluid(depthPositive) || !isFluid(depthNegative)) { break; }

    let gaussian = exp(-f32(offset * offset) / twoSigmaSqr);
    state = accumulatePair(
      state, center, depthPositive, depthNegative, gaussian,
      options.filterDelta, options.filterMu
    );

  }

  textureStore(outputTexture, coord, vec4<f32>(state.sum / state.weight, 0.0, 0.0, 0.0));

}
`;


// Final clean-up pass (paper section 3.4): a small fixed size 2D filter that
// removes the axis aligned streaks left by the separable approximation.
const FilterShaderCleanup = /* wgsl */`
${FilterCommon}

const CleanupRadius: i32 = 2;         // 5x5
const CleanupTwoSigmaSqr: f32 = 2.0;  // sigma = 1 pixel

@compute @workgroup_size(8, 8, 1)
fn main( @builtin(global_invocation_id) globalId: vec3<u32> ) {

  let size = vec2<i32>(textureDimensions(inputTexture));
  let coord = vec2<i32>(globalId.xy);
  if (any(coord >= size)) { return; }

  let center = textureLoad(inputTexture, coord, 0).x;
  if (!isFluid(center)) {
    textureStore(outputTexture, coord, vec4<f32>(0.0));
    return;
  }

  var state = FilterState(center, 1.0, options.filterDelta, options.filterDelta);

  // Half of the kernel, each offset paired with its opposite. Rows are visited
  // in increasing distance, which is close enough to "closest pixels first" for
  // a kernel this small.
  for (var dy = 0; dy <= CleanupRadius; dy++) {
    for (var dx = -CleanupRadius; dx <= CleanupRadius; dx++) {

      if (dy == 0 && dx <= 0) { continue; }
      let offset = vec2<i32>(dx, dy);

      let depthPositive = loadDepth(coord + offset, size);
      let depthNegative = loadDepth(coord - offset, size);
      // Unlike the 1D passes there is nothing to break out of, the remaining
      // offsets point in unrelated directions.
      if (!isFluid(depthPositive) || !isFluid(depthNegative)) { continue; }

      let gaussian = exp(-f32(dx * dx + dy * dy) / CleanupTwoSigmaSqr);
      state = accumulatePair(
        state, center, depthPositive, depthNegative, gaussian,
        options.filterDelta, options.filterMu
      );

    }
  }

  textureStore(outputTexture, coord, vec4<f32>(state.sum / state.weight, 0.0, 0.0, 0.0));

}
`;

export { FilterShader1D, FilterShaderCleanup };
