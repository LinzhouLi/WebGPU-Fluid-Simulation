import { PBFConfig } from '../PBFConfig';
import { Boundary, KernalPoly6, KernalSpikyGrad } from './common';


const LambdaCalculationShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const GridSize: vec3<f32> = vec3<f32>(${PBFConfig.BOUNDARY_GRID[0]}, ${PBFConfig.BOUNDARY_GRID[1]}, ${PBFConfig.BOUNDARY_GRID[2]});
const GridSizeU: vec3<u32> = vec3<u32>(GridSize);
const GridSpaceSize: vec3<f32> = 1.0 / GridSize;

override ParticleCount: u32;
override ParticleVolume: f32;
override ParticleVolume2: f32;
override LambdaEPS: f32;

${KernalPoly6}
${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(0) @binding(1) var<storage, read> neighborList: array<u32>;

@group(1) @binding(0) var<storage, read_write> position2: array<vec3<f32>>;
@group(1) @binding(2) var<storage, read_write> lambda: array<f32>;
@group(1) @binding(3) var<storage, read_write> boundaryData: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = position2[particleIndex];
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var grad_Pk: vec3<f32>;
  var grad_Pi_Ci = vec3<f32>();
  var sum_grad_Pj_Ci_2 = f32();
  var density = f32();

  // fluid
  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    positionDelta = selfPosition - position2[nParticleIndex];
    positionDeltaLength = length(positionDelta);

    grad_Pk = kernalSpikyGrad(positionDelta, positionDeltaLength);
    grad_Pi_Ci += grad_Pk;
    sum_grad_Pj_Ci_2 += dot(grad_Pk, grad_Pk);
    density += kernalPoly6(positionDeltaLength);

    nListIndex++;
  }
  density *= ParticleVolume;
  grad_Pi_Ci *= ParticleVolume;
  sum_grad_Pj_Ci_2 *= ParticleVolume2;

  // boundary
  {
    let xj_vj = boundaryData[particleIndex];
    let xj = xj_vj.xyz; 
    let vj = xj_vj.w;

    positionDelta = selfPosition - xj;
    positionDeltaLength = length(positionDelta);

    grad_Pk = vj * kernalSpikyGrad(positionDelta, positionDeltaLength);
    grad_Pi_Ci += grad_Pk;
    density += vj * kernalPoly6(positionDeltaLength);
  }

  let constrain = max(0.0, density - 1.0);
  // let constrain = max(-0.1, density - 1.0);
  let sum_grad_Ci_2 = dot(grad_Pi_Ci, grad_Pi_Ci) + sum_grad_Pj_Ci_2;
  lambda[particleIndex] = -constrain / (sum_grad_Ci_2 + LambdaEPS);
  return;
}
`;

const ConstrainSolveShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};

override ParticleCount: u32;
override ParticleVolume: f32;
// override ScorrCoef: f32;

${KernalPoly6}
${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(0) @binding(1) var<storage, read> neighborList: array<u32>;

@group(1) @binding(0) var<storage, read_write> position2: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read_write> deltaPosition: array<vec3<f32>>;
@group(1) @binding(2) var<storage, read_write> lambda: array<f32>;
@group(1) @binding(3) var<storage, read_write> boundaryData: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = position2[particleIndex];
  let selfLambda = lambda[particleIndex];
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var neighborLambda: f32;
  var scorr: f32;
  var positionUpdate = vec3<f32>();

  // fluid
  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    positionDelta = selfPosition - position2[nParticleIndex];
    positionDeltaLength = length(positionDelta);
    neighborLambda = lambda[nParticleIndex];

    // scorr = kernalPoly6(positionDeltaLength); // suppose scorr_n == 4
    // scorr *= scorr;
    // scorr *= ScorrCoef * scorr;
    positionUpdate += (selfLambda + neighborLambda) * // + scorr) *
      kernalSpikyGrad(positionDelta, positionDeltaLength);

    nListIndex++;
  }
  positionUpdate *= ParticleVolume;

  // boundary
  {
    let xj_vj = boundaryData[particleIndex];
    let xj = xj_vj.xyz; 
    let vj = xj_vj.w;

    positionDelta = selfPosition - xj;
    positionDeltaLength = length(positionDelta);

    positionUpdate += vj * selfLambda * kernalSpikyGrad(positionDelta, positionDeltaLength);
  }

  deltaPosition[particleIndex] = 0.5 * positionUpdate;
  return;
}
`;

const ConstrainApplyShader = /* wgsl */`
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
override ParticleCount: u32;
${Boundary}

@group(1) @binding(0) var<storage, read_write> position2: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read_write> deltaPosition: array<vec3<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let pos = boundary_rand(position2[particleIndex] + deltaPosition[particleIndex]);
  position2[particleIndex] = pos;
  return;
}
`;


export { LambdaCalculationShader, ConstrainSolveShader, ConstrainApplyShader };