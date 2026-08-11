import { ShaderStruct, ShaderFunction, ShaderCode } from '../../common/shader';

// Shading pass of the screen space fluid renderer: rebuilds the fluid surface
// from the filtered depth map and shades it with environment reflection,
// refraction and Beer-Lambert absorption driven by the thickness map.
//
// The pass is a full screen triangle blended over the already rendered scene.
// Occlusion by scene geometry was resolved when the imposters were rasterized
// against the scene depth buffer, so pixels without fluid are simply discarded.

const CompositeShader = /* wgsl */`
${ShaderStruct.Camera}
${ShaderStruct.DirectionalLight}
${ShaderStruct.FluidOptions}
${ShaderCode.GlobalGroup}
${ShaderFunction.sRGBGammaEncode}

@group(1) @binding(0) var depthTexture: texture_2d<f32>;
@group(1) @binding(1) var thicknessTexture: texture_2d<f32>;
@group(1) @binding(2) var<uniform> options: FluidOptions;

@vertex
fn vertex( @builtin(vertex_index) vertexIndex: u32 ) -> @builtin(position) vec4<f32> {
  return vec4<f32>(
    select(-1.0, 3.0, vertexIndex == 1u),
    select(-1.0, 3.0, vertexIndex == 2u),
    0.0, 1.0
  );
}

// Eye space position of a pixel, w = 0 when the pixel holds no fluid.
// camera.params.xy = (aspect * height / near, -height / near), see globalResource.ts,
// so the negative y component already accounts for the downwards screen axis.
fn loadViewPosition(coord: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
  if (any(coord < vec2<i32>(0)) || any(coord >= size)) { return vec4<f32>(0.0); }
  let depth = textureLoad(depthTexture, coord, 0).x;
  if (depth >= 0.0) { return vec4<f32>(0.0); }
  let uv = (vec2<f32>(coord) + 0.5) / vec2<f32>(size);
  return vec4<f32>((uv - 0.5) * camera.params.xy * (-depth), depth, 1.0);
}

// Finite differences of the eye space position. Neighbors without fluid fall
// back to a one sided difference, otherwise the silhouette gets a bright rim.
fn viewNormal(coord: vec2<i32>, size: vec2<i32>, center: vec3<f32>) -> vec3<f32> {

  let left = loadViewPosition(coord - vec2<i32>(1, 0), size);
  let right = loadViewPosition(coord + vec2<i32>(1, 0), size);
  let down = loadViewPosition(coord - vec2<i32>(0, 1), size);
  let up = loadViewPosition(coord + vec2<i32>(0, 1), size);

  var ddx = vec3<f32>(1.0, 0.0, 0.0);
  if (left.w > 0.0 && right.w > 0.0) {
    let backward = center - left.xyz;
    let forward = right.xyz - center;
    // Prefer the side with the smaller depth change to avoid smearing the
    // normal across a discontinuity.
    ddx = select(forward, backward, abs(backward.z) < abs(forward.z));
  } else if (left.w > 0.0) { ddx = center - left.xyz; }
  else if (right.w > 0.0) { ddx = right.xyz - center; }

  var ddy = vec3<f32>(0.0, 1.0, 0.0);
  if (down.w > 0.0 && up.w > 0.0) {
    let backward = center - down.xyz;
    let forward = up.xyz - center;
    ddy = select(forward, backward, abs(backward.z) < abs(forward.z));
  } else if (down.w > 0.0) { ddy = center - down.xyz; }
  else if (up.w > 0.0) { ddy = up.xyz - center; }

  let tangentCross = cross(ddx, ddy);
  let lengthSqr = dot(tangentCross, tangentCross);
  if (lengthSqr < 1e-24) { return vec3<f32>(0.0, 0.0, 1.0); } // degenerate
  var normal = tangentCross * inverseSqrt(lengthSqr);
  if (normal.z < 0.0) { normal = -normal; } // always face the camera
  return normal;

}

// The thickness map is noisy since it is a raw sum of sphere chords, a few taps
// of bilinear filtering are enough to hide it.
fn sampleThickness(uv: vec2<f32>, texelSize: vec2<f32>) -> f32 {
  var total = 2.0 * textureSampleLevel(thicknessTexture, linearSampler, uv, 0.0).x;
  total += textureSampleLevel(thicknessTexture, linearSampler, uv + vec2<f32>( 1.5,  1.5) * texelSize, 0.0).x;
  total += textureSampleLevel(thicknessTexture, linearSampler, uv + vec2<f32>(-1.5,  1.5) * texelSize, 0.0).x;
  total += textureSampleLevel(thicknessTexture, linearSampler, uv + vec2<f32>( 1.5, -1.5) * texelSize, 0.0).x;
  total += textureSampleLevel(thicknessTexture, linearSampler, uv + vec2<f32>(-1.5, -1.5) * texelSize, 0.0).x;
  return total / 6.0;
}

@fragment
fn fragment( @builtin(position) position: vec4<f32> ) -> @location(0) vec4<f32> {

  let size = vec2<i32>(textureDimensions(depthTexture));
  let coord = vec2<i32>(position.xy);

  let viewPosition = loadViewPosition(coord, size);
  if (viewPosition.w == 0.0) { discard; }

  let sizeFloat = vec2<f32>(size);
  let uv = position.xy / sizeFloat;
  let thickness = sampleThickness(uv, 1.0 / sizeFloat);

  let normalView = viewNormal(coord, size, viewPosition.xyz);
  let normal = normalize((camera.viewMatrixInverse * vec4<f32>(normalView, 0.0)).xyz);
  let worldPosition = (camera.viewMatrixInverse * vec4<f32>(viewPosition.xyz, 1.0)).xyz;
  let viewDirection = normalize(camera.position - worldPosition);

  // Reflection and refraction both look up the environment cube map.
  let reflectionColor = textureSampleLevel(
    envMap, linearSampler, reflect(-viewDirection, normal), 0.0
  ).rgb;

  // A single refraction bends the ray far too much for thin fluid: light leaving
  // the back surface is refracted again and a thin sheet barely deflects it at
  // all. Fading towards the view direction as the fluid gets thinner keeps spray
  // and droplets looking transparent instead of sampling an unrelated (dark)
  // part of the environment map.
  var refractionDirection = refract(-viewDirection, normal, 1.0 / options.ior);
  if (all(refractionDirection == vec3<f32>(0.0))) { // total internal reflection
    refractionDirection = -viewDirection;
  }
  refractionDirection = normalize(mix(
    -viewDirection, refractionDirection,
    saturate(thickness / (8.0 * options.particleRadius))
  ));
  let refractionColor = textureSampleLevel(envMap, linearSampler, refractionDirection, 0.0).rgb;

  // Beer-Lambert absorption, with the fluid tint acting as the per channel
  // absorption coefficient, plus a cheap single scattering term: since
  // refraction only looks up the environment map, thick fluid would otherwise
  // just turn black instead of taking on its own color. The sky above is used
  // as the ambient estimate so this adapts to the environment.
  let transmittance = exp(-options.absorption * thickness * (1.0 - options.fluidColor));
  let ambient = textureSampleLevel(envMap, linearSampler, vec3<f32>(0.0, 1.0, 0.0), 0.0).rgb;
  let refraction = refractionColor * transmittance
    + options.fluidColor * ambient * (1.0 - transmittance);

  // Schlick, F0 = 0.02 for water. The sun highlight is part of the specular
  // reflection, so it is modulated by the same Fresnel term instead of being
  // added on top - otherwise ripples blow out to white.
  let fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(normal, viewDirection)), 5.0);
  let halfVector = normalize(viewDirection + light.direction);
  let specular = pow(saturate(dot(normal, halfVector)), 256.0) * light.color;

  let color = mix(refraction, reflectionColor + specular, fresnel);

  // Thin sheets and spray stay translucent and blend with the background.
  let alpha = saturate(1.0 - exp(-options.opacity * thickness));

  return vec4<f32>(sRGBGammaEncode(color), alpha);

}
`;

export { CompositeShader };
