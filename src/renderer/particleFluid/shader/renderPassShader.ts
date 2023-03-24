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
@group(0) @binding(5) var envMap: texture_cube<f32>;

${ShaderFunction.sRGBGammaEncode}

// fn linearEyeDepth(z: f32) -> f32 {
//   return 1.0 / dot(camera.params.zw, vec2<f32>(z, 1.0));
// }

// fn uvToEye(uv: vec2<f32>, z: f32) -> vec3<f32> {
//   let depthEye = linearEyeDepth(z); // Z coordinate of eye space is negtive
//   return vec3<f32>(
//     (uv - 0.5) * camera.params.xy * depthEye,
//     -depthEye
//   );
// }

fn getNormal(positionEye: vec3<f32>) -> vec3<f32> {
  let ddx = dpdx(positionEye);
  let ddy = dpdy(positionEye);
  let normalEye = normalize(cross(-ddx, ddy));
  return normalEye;
}

fn getPosition(uv: vec2<f32>, depthEye: f32) -> vec3<f32> {
  return vec3<f32>(
    (uv - 0.5) * camera.params.xy * depthEye, -depthEye
  );
}

fn diffuseShading(normalWorld: vec3<f32>) -> vec3<f32> {
  // simple diffuse shading
  let NoL = saturate(dot(normalWorld, light.direction));
  let irradiance = NoL * light.color;
  return (irradiance + 0.02) * 0.3183098861837907 * vec3<f32>(4, 142, 219) / 256.0; // RECIPROCAL_PI
}

fn Fresnel_Schlick(F0: vec3<f32>, VoH: f32) -> vec3<f32> {
  let Fc = exp2((-5.55473 * VoH - 6.98316) * VoH);
  return saturate(50.0 * F0) * Fc + (1.0 - Fc) * F0;
}

fn shading(
  normalEye: vec3<f32>,
  positionEye: vec3<f32>,
  thickness: f32
) -> vec3<f32> {

  let viewDirEye = normalize(-positionEye);
  let VoN = saturate(dot(viewDirEye, normalEye));
  let fresnel = Fresnel_Schlick(vec3<f32>(0.02), VoN); // F0 of water is 0.02
  let attenuate = max(exp(-thickness * 0.5), 0.2);
  let tintColor = vec3<f32>(6.0, 105.0, 217.0) / 256.0;

  let reflectDirEye = reflect(-viewDirEye, normalEye);
  let refractDirEye = refract(-viewDirEye, normalEye, 0.7501875); // 1.0 / 1.333

  let normalWorld = (camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0)).xyz;
  let reflectDirWorld = (camera.viewMatrixInverse * vec4<f32>(reflectDirEye, 0.0)).xyz;
  let refractDirWorld = (camera.viewMatrixInverse * vec4<f32>(refractDirEye, 0.0)).xyz;

  let normalColor = textureSample(envMap, linearSampler, normalWorld).rgb;
  let reflectColor = textureSample(envMap, linearSampler, reflectDirWorld).rgb;
  let refractColor = textureSample(envMap, linearSampler, refractDirWorld).rgb;

  let color = mix(
    mix(tintColor, refractColor, attenuate),
    reflectColor, fresnel
  );
  return color;

}

@fragment
fn main(input: FragInput) -> FragOutput {

  let frameCoord = vec2<u32>(floor(input.coord.zw));
  let depthEye = textureLoad(fluidDepthMap, frameCoord, 0).r;
  if (depthEye == 0.0) { discard; }
  let fluidVolume = textureSample(fluidVolumeMap, linearSampler, input.coord.xy).r;
  let positionEye = getPosition(input.coord.xy, depthEye);
  let normalEye = getNormal(positionEye);
  let color = shading(normalEye, positionEye, fluidVolume);

  return FragOutput(
    vec4<f32>(sRGBGammaEncode(color), 1.0),
    // vec4<f32>(vec3<f32>(fluidVolume), 1.0)
  );

}

`;

export { fragmentShader };