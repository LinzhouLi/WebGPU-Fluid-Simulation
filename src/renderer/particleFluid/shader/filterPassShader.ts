
const computeShader = /* wgsl */`

override width: u32;
override height: u32;

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var destTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> filterDirection: vec2<u32>;
@group(0) @binding(2) var<uniform> filterSize: f32;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  if (global_index.x >= width || global_index.y >= height) { return; }
  let val = textureLoad(srcTexture, global_index.xy, 0);
  textureStore(destTexture, global_index.xy, val);
  return;

}

`;

export { computeShader };