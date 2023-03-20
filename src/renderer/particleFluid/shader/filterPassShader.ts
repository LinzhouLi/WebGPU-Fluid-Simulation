
const computeShader = /* wgsl */`

override width: u32;
override height: u32;

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var destTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> filterDirection: vec2<i32>;
@group(0) @binding(3) var<uniform> filterSize: i32;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  if (global_index.x >= width || global_index.y >= height) { return; }
  let coord = vec2<i32>(global_index.xy);
  let val = textureLoad(srcTexture, coord, 0).r;

  if (val > 0.0) {
    var val_sum = 0.0; var weight_sum = 0.0;
    var sample_val: f32; var sample_weight: f32;
    var r: f32;
    for (var i: i32 = -filterSize; i <= filterSize; i++) {
      sample_val = textureLoad(srcTexture, coord + i * filterDirection, 0).r;
      if (sample_val == 0.0) { continue; }

      // spatial domain
      r = f32(i) * 0.01;
      sample_weight = exp(-r * r);

      // range domain;
      r = (sample_val - val) * 2.0;
      sample_weight = sample_weight * exp(-r * r);

      val_sum = val_sum + sample_val * sample_weight;
      weight_sum = weight_sum + sample_weight;
    }

    if (sample_weight > 0.0) {
      textureStore(
        destTexture, global_index.xy, 
        vec4<f32>(val_sum / weight_sum, 0.0, 0.0, 0.0)
      );
    }
  }
  else {
    textureStore( destTexture, global_index.xy, vec4<f32>(0.0) );
  }
  return;

}

`;

export { computeShader };