const KernalPoly6 = /* wgsl */`
fn kernalPoly6(r_len: f32) -> f32 {
  const KernelRadius2 = KernelRadius * KernelRadius;
  const coef = 315.0 / (64.0 * PI * pow(KernelRadius, 9));
  let t = KernelRadius2 - r_len * r_len;
  let x = select(
    0.0,
    t * t * t,
    r_len <= KernelRadius
  );
  return coef * x;
}
`;


const KernalCohesion = /* wgsl */`
fn kernalCohesion(r: vec3<f32>, r_len: f32) -> vec3<f32> {
  const halfKernelRadius = 0.5 * KernelRadius;
  const coef = 32.0 / (PI * pow(KernelRadius, 9));
  const bais = -pow(KernelRadius, 6) / 64.0;
  let tv = vec2<f32>(KernelRadius - r_len, r_len);
  let t3 = tv * tv * tv;
  let t = t3.x * t3.y; // (h - r)^3 * r^3
  var result = select(fma(2.0, t, bais), t, r_len > halfKernelRadius);
  result = select(0.0, result, r_len <= KernelRadius);
  let r_norm = select( vec3<f32>(0.0), r / r_len, r_len > EPS ); // handle r_len == 0
  return result * r_norm;
}
`;


const KernalSpikyGrad = /* wgsl */`
fn kernalSpikyGrad(r: vec3<f32>, r_len: f32) -> vec3<f32> {
  const coef = -45.0 / (PI * pow(KernelRadius, 6));
  let t = KernelRadius - r_len;
  let x = select( 0.0, t * t, r_len <= KernelRadius );
  let r_norm = select( vec3<f32>(0.0), r / r_len, r_len > EPS ); // handle r_len == 0
  return coef * x * r_norm; // normalize(r) = r / r_len
}
`;


const Boundary = /* wgsl */`
fn boundary(pos: vec3<f32>) -> vec3<f32> {
  const EPS = 1e-3;
  const UP = vec3<f32>(1.0 - EPS);
  const BOTTOM = vec3<f32>(EPS);
  return max(min(pos, UP), BOTTOM);
}

fn random(uv: vec2<f32>) -> f32 {
	return fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn random3(p: vec3<f32>) -> vec3<f32> {
  const a = vec2<f32>(12.9898, 78.233);
  let b = vec3<f32>(dot(p.yz, a), dot(p.xz, a), dot(p.xy, a));
  return fract(sin(b) * 43758.5453);
}

fn boundary_rand(pos: vec3<f32>) -> vec3<f32> {
  const EPS = 1e-4;
  let rand_vec3 = EPS * random3(pos);
  let bottom_bound = rand_vec3;
  let up_bound = 1.0 - rand_vec3;
  return max(min(pos, up_bound), bottom_bound);
}
`;

export { KernalPoly6, KernalCohesion, KernalSpikyGrad, Boundary };