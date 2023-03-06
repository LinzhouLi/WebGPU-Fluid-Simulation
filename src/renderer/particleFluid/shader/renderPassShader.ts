import { ShaderStruct } from "../../../common/shaderStruct";
import { ShaderFunction } from "../../../common/shaderFunction";

const fragmentShader = /* wgsl */`

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}

struct FragInput {
  @location(0) @interpolate(linear, center) coord: vec4<f32>,
};

struct FragOutput {
  @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: DirectionalLight;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var fluidDepthMap: texture_2d<f32>;
@group(0) @binding(4) var fluidVolumeMap: texture_2d<f32>;

${ShaderFunction.sRGBGammaEncode}

fn linearEyeDepth(z: f32) -> f32 {
  return 1.0 / dot(camera.params.zw, vec2<f32>(z, 1.0));
}

fn uvToEye(uv: vec2<f32>, z: f32) -> vec3<f32> {
  let depthEye = linearEyeDepth(z); // Z coordinate of eye space is negtive
  return vec3<f32>(
    (uv - 0.5) * camera.params.xy * depthEye,
    -depthEye
  );
}

fn getNormal(uv: vec2<f32>, z: f32) -> vec4<f32> {

  let posEye = uvToEye(uv, z);
  let ddx = dpdx(posEye);
  let ddy = dpdy(posEye);

  let normalEye = normalize(cross(ddy, ddx));
  return camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0);

}

@fragment
fn main(input: FragInput) -> FragOutput {

  let frameCoord = vec2<u32>(floor(input.coord.zw));
  let z = textureLoad(fluidDepthMap, frameCoord, 0).r;
  if (z < 1e-4) { discard; }
  let fluidVolume = textureSample(fluidVolumeMap, linearSampler, input.coord.xy).r;
  let normalWorld = getNormal(input.coord.xy, z);

  // simple diffuse shading
  let NoL = saturate(dot(normalWorld.xyz, light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 *vec3<f32>(4,142,219)/256.0; // RECIPROCAL_PI

  return FragOutput(
    vec4<f32>(sRGBGammaEncode(diffuse), 1.0)
  );

}

`;

export { fragmentShader };