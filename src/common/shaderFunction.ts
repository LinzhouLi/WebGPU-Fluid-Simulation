
const sRGBGammaEncode = /* wgsl */`
fn sRGBGammaEncode(color: vec3<f32>) -> vec3<f32> {
  return mix(
    color.rgb * 12.92,                                    // x <= 0.0031308
    pow(color.rgb, vec3<f32>(0.41666)) * 1.055 - 0.055,   // x >  0.0031308
    saturate(sign(color.rgb - 0.0031308))
  );
}
`;

const ShaderFunction = { sRGBGammaEncode };

export { ShaderFunction }