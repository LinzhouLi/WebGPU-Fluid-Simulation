// VertOutput.coord.xy range from (0, 0) to (1, 1). Used as uv sampling coord
// VertOutput.coord.zw range from (0, 0) to (screenWidth, screenHeight). Used as texture coord for textureLoad()

const vertexShader = /* wgsl */`

override screenWidth: f32;
override screenHeight: f32;

struct VertOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(linear, center) coord: vec4<f32>,
};

const coords = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0), // Bottom Left
  vec2<f32>( 1.0, -1.0), // Bottom Right
  vec2<f32>(-1.0,  1.0), // Top Left
  vec2<f32>( 1.0,  1.0)  // Top Right
);

@vertex
fn main(@builtin(vertex_index) index: u32) -> VertOutput {
  let coord = coords[index];
  let position = vec4<f32>(coord, 0.0, 1.0);
  let uv = coord * vec2<f32>(0.5, -0.5) + 0.5; // https://www.w3.org/TR/webgpu/#coordinate-systems
  let gbufferCoord = vec4<f32>(uv, uv * vec2<f32>(screenWidth, screenHeight));
  return VertOutput(
    position, gbufferCoord
  );
}

`;

export { vertexShader };