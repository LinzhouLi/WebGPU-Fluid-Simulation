import { ShaderStruct, ShaderFunction } from "../../../common/shader";

const fragmentShader = /* wgsl */`

${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.RenderingOptions}

struct FragInput {
  @location(0) @interpolate(linear, center) coord: vec4<f32>,
};

struct FragOutput {
  @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: DirectionalLight;
@group(0) @binding(2) var<uniform> options: RenderingOptions;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var fluidDepthMap: texture_2d<f32>;
@group(0) @binding(5) var fluidVolumeMap: texture_2d<f32>;
@group(0) @binding(6) var envMap: texture_cube<f32>;

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

fn diffuseShading(normalEye: vec3<f32>) -> vec4<f32> {
  // simple diffuse shading
  let normalWorld = (camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0)).xyz;
  let NoL = saturate(dot(normalWorld, light.direction));
  let irradiance = NoL * light.color;
  let color = (irradiance + 0.05) * 0.3183098861837907 * options.tintColor; // RECIPROCAL_PI
  return vec4<f32>(color, 1.0);
}

fn Fresnel_Schlick(F0: vec3<f32>, VoH: f32) -> vec3<f32> {
  let Fc = exp2((-5.55473 * VoH - 6.98316) * VoH);
  return saturate(50.0 * F0) * Fc + (1.0 - Fc) * F0;
}

fn shading(
  normalEye: vec3<f32>,
  positionEye: vec3<f32>,
  thickness: f32
) -> vec4<f32> {

  const n_water = 1.333333;
  const n_water_inv = 1.0 / n_water;
  const F_0_t = (n_water - 1) / (n_water + 1);
  const F_0 = vec3<f32>(F_0_t * F_0_t);

  let viewDirEye = normalize(-positionEye);
  let VoN = saturate(dot(viewDirEye, normalEye));
  let fresnel = Fresnel_Schlick(F_0, VoN); // F0 of water is 0.02

  let normalWorld = (camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0)).xyz;

  let reflectDirEye = reflect(-viewDirEye, normalEye);
  let reflectDirWorld = (camera.viewMatrixInverse * vec4<f32>(reflectDirEye, 0.0)).xyz;
  let reflectColor = textureSample(envMap, linearSampler, reflectDirWorld).rgb;

  var color: vec4<f32>;

  if (options.mode == 0) {
    let refractDirEye = refract(-viewDirEye, normalEye, n_water_inv); // 1.0 / 1.333
    let refractDirWorld = (camera.viewMatrixInverse * vec4<f32>(refractDirEye, 0.0)).xyz;
    let refractColor = textureSample(envMap, linearSampler, refractDirWorld).rgb;

    let attenuate = exp(-options.opacity * thickness);
    color = vec4<f32>(
      mix(
        mix(options.tintColor, refractColor, attenuate),
        reflectColor, fresnel
      ),
      1.0
    );
  }
  else {
    let attenuate = exp(-5.0 * thickness);
    color = vec4<f32>(
      mix(
        options.tintColor,
        reflectColor, fresnel
      ),
      1.0 - attenuate
    );
  }

  return color;

}

@fragment
fn main(input: FragInput) -> FragOutput {

  let frameCoord = vec2<u32>(floor(input.coord.zw));
  let depthEye = textureLoad(fluidDepthMap, frameCoord, 0).r;
  let positionEye = getPosition(input.coord.xy, depthEye);
  let normalEye = getNormal(positionEye);
  if (depthEye == 0.0) { discard; }
  let fluidVolume = 0.02 * textureSample(fluidVolumeMap, linearSampler, input.coord.xy).r;

  var color: vec4<f32>;

  switch options.mode {
    case 2: {
      color = diffuseShading(normalEye);
      break;
    }
    case 3: {
      let normalWorld = (camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0)).xyz;
      color = vec4<f32>(normalWorld, 1.0);
      break;
    }
    case 4: {
      let posCam = vec4<f32>(0.0, 0.0, -depthEye, 1.0);
      let posClip = camera.projectionMatrix * posCam;
      color = vec4<f32>(vec3<f32>(posClip.z / posClip.w), 1.0); // camera depth
      // color = vec4<f32>(vec3<f32>(depthEye* 0.4), 1.0);
      break;
    }
    case 5: {
      color = vec4<f32>(vec3<f32>(fluidVolume), 1.0);
      break;
    }
    case 6: {
      color = vec4<f32>(vec3<f32>(positionEye), 1.0);
      break;
    }
    case 0, 1, default: {
      color = shading(normalEye, positionEye, fluidVolume);
    }
  }

  color = vec4<f32>(sRGBGammaEncode(color.rgb), color.w);
  return FragOutput( color );

}

`;

export { fragmentShader };