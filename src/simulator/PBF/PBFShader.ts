import { PBFConfig } from './PBFConfig';

const NeighborStruct = /* wgsl */`
struct Neighbor {
  count: u32,
  particleIndex: array<u32, MaxNeighborCount>
};
`;


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

const KernalSpiky = /* wgsl */`
fn kernalSpiky(r_len: f32) -> f32 {
  const coef = 15.0 / (PI * pow(KernelRadius, 6));
  let t = KernelRadius - r_len;
  let x = select( 0.0, t * t * t, r_len <= KernelRadius );
  return coef * x;
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
  const EPS = 1e-3;
  let rand_vec3 = EPS * random3(pos);
  let bottom_bound = rand_vec3;
  let up_bound = 1.0 - rand_vec3;
  return max(min(pos, up_bound), bottom_bound);
}
`;


/********************* Advection *********************/
const ForceApplyShader = /* wgsl */`
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
override ParticleCount: u32;
override DeltaT: f32;
${Boundary}

@group(0) @binding(0) var<storage, read_write> position: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> positionPredict: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(0) @binding(3) var<uniform> gravity: vec3<f32>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let vel = velocity[particleIndex] + DeltaT * gravity;
  let pos = boundary_rand(position[particleIndex] + DeltaT * vel);
  positionPredict[particleIndex] = pos;
  return;
}
`;


/***************** Constrain Solving *****************/
const LambdaCalculationShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const EPS: f32 = 1e-6;
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const MaxNeighborCount: u32 = ${PBFConfig.MAX_NEIGHBOR_COUNT};
override ParticleCount: u32;
override ParticleVolume: f32;
override ParticleVolume2: f32;
override LambdaEPS: f32;
${NeighborStruct}
${KernalPoly6}
${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read_write> positionPredict: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(3) var<storage, read_write> neighborList: array<Neighbor>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = positionPredict[particleIndex];
  let neighbor = neighborList[particleIndex];
  let neighborCount = neighbor.count;
  var neighborIndex: u32;

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var grad_Pk: vec3<f32>;
  var grad_Pi_Ci = vec3<f32>();
  var sum_grad_Pj_Ci_2 = f32();
  var density = f32();

  for (var i: u32 = 0; i < neighborCount; i++) {
    neighborIndex = neighbor.particleIndex[i];
    positionDelta = selfPosition - positionPredict[neighborIndex];
    positionDeltaLength = length(positionDelta);

    grad_Pk = kernalSpikyGrad(positionDelta, positionDeltaLength);
    grad_Pi_Ci += grad_Pk;
    sum_grad_Pj_Ci_2 += dot(grad_Pk, grad_Pk);

    density += kernalPoly6(positionDeltaLength);
  }

  let constrain = max(0.0, density * ParticleVolume - 1.0);
  let sum_grad_Ci_2 = (dot(grad_Pi_Ci, grad_Pi_Ci) + sum_grad_Pj_Ci_2) * ParticleVolume2;
  lambda[particleIndex] = -constrain / (sum_grad_Ci_2 + LambdaEPS);
  return;
}
`;


const ConstrainSolveShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const EPS: f32 = 1e-6;
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const MaxNeighborCount: u32 = ${PBFConfig.MAX_NEIGHBOR_COUNT};
override ParticleCount: u32;
override ParticleVolume: f32;
override ScorrCoef: f32;
${NeighborStruct}
${KernalPoly6}
${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read_write> positionPredict: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> deltaPosition: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(3) var<storage, read_write> neighborList: array<Neighbor>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = positionPredict[particleIndex];
  let selfLambda = lambda[particleIndex];
  let neighbor = neighborList[particleIndex];
  let neighborCount = neighbor.count;
  var neighborIndex: u32;

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var neighborLambda: f32;
  var scorr: f32;
  var positionUpdate = vec3<f32>();

  for (var i: u32 = 0; i < neighborCount; i++) {
    neighborIndex = neighbor.particleIndex[i];
    positionDelta = selfPosition - positionPredict[neighborIndex];
    positionDeltaLength = length(positionDelta);
    neighborLambda = lambda[neighborIndex];

    scorr = kernalPoly6(positionDeltaLength); // suppose scorr_n == 4
    scorr *= scorr;
    scorr *= ScorrCoef * scorr;

    positionUpdate += (selfLambda + neighborLambda) * // + scorr) *
      kernalSpikyGrad(positionDelta, positionDeltaLength);
  }

  deltaPosition[particleIndex] = 0.5 * positionUpdate * ParticleVolume;
  return;
}
`;

const ConstrainApplyShader = /* wgsl */`
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
override ParticleCount: u32;
${Boundary}

@group(0) @binding(0) var<storage, read_write> positionPredict: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> deltaPosition: array<vec3<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let pos = boundary_rand(positionPredict[particleIndex] + deltaPosition[particleIndex]);
  positionPredict[particleIndex] = pos;
  return;
}
`;


/********************* Viscosity *********************/
const AttributeUpdateShader = /* wgsl */`
override ParticleCount: u32;
override InvDeltaT: f32;

@group(0) @binding(0) var<storage, read_write> position: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> positionPredict: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> velocityCopy: array<vec3<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let pos = position[particleIndex];
  let posPred = positionPredict[particleIndex];
  let vel = (posPred - pos) * InvDeltaT;
  velocityCopy[particleIndex] = vel;
  position[particleIndex] = posPred;
  return;
}
`;


const XSPHShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const MaxNeighborCount: u32 = ${PBFConfig.MAX_NEIGHBOR_COUNT};
override ParticleCount: u32;
override ParticleVolume: f32;
override XSPHCoef: f32;
${NeighborStruct}
${KernalPoly6}

@group(0) @binding(0) var<storage, read_write> position: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> velocityCopy: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read_write> neighborList: array<Neighbor>;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = position[particleIndex];
  let selfVelocity = velocityCopy[particleIndex];
  let neighbor = neighborList[particleIndex];
  let neighborCount = neighbor.count;
  var neighborIndex: u32;

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var velocityDelta: vec3<f32>;
  var velocityUpdate = vec3<f32>();

  for (var i: u32 = 0; i < neighborCount; i++) {
    neighborIndex = neighbor.particleIndex[i];
    positionDelta = selfPosition - position[neighborIndex];
    positionDeltaLength = length(positionDelta);
    velocityDelta = velocityCopy[neighborIndex] - selfVelocity;

    velocityUpdate += velocityDelta * kernalPoly6(positionDeltaLength);
  }

  velocity[particleIndex] = selfVelocity + XSPHCoef * velocityUpdate * ParticleVolume;
  return;
}
`;

export { 
  ForceApplyShader,       LambdaCalculationShader,
  ConstrainSolveShader,   ConstrainApplyShader,
  AttributeUpdateShader,  XSPHShader
};