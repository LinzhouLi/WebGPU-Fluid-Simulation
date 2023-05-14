import { PBFConfig } from '../PBFConfig';
import { KernalPoly6, KernalSpikyGrad } from './common';


const XSPHShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const EPS: f32 = 1e-6;
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const MaxNeighborCount: u32 = ${PBFConfig.MAX_NEIGHBOR_COUNT};
override InvDeltaT: f32;
override ParticleCount: u32;
override ParticleVolume: f32;
override XSPHCoef: f32;
override VorticityCoef: f32;
${KernalPoly6}
${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read_write> neighborOffset: array<u32>;
@group(0) @binding(1) var<storage, read_write> neighborList: array<u32>;

@group(1) @binding(0) var<storage, read_write> position: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read_write> angularVelocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(1) @binding(3) var<storage, read_write> acceleration: array<vec3<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = position[particleIndex];
  let selfVelocity = velocity[particleIndex];
  let selfAngularVelocity = angularVelocity[particleIndex].xyz;
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var velocityDelta: vec3<f32>;
  var velocityUpdate = vec3<f32>();
  var etai = vec3<f32>();

  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    positionDelta = selfPosition - position[nParticleIndex]; // xi - xj
    positionDeltaLength = length(positionDelta);
    velocityDelta = velocity[nParticleIndex] - selfVelocity; // vj - vi

    velocityUpdate += velocityDelta * kernalPoly6(positionDeltaLength);
    etai += angularVelocity[nParticleIndex].w * kernalSpikyGrad(positionDelta, positionDeltaLength);

    nListIndex++;
  }

  acceleration[particleIndex] += ParticleVolume * (
    InvDeltaT * XSPHCoef * velocityUpdate +
    VorticityCoef * cross(etai, selfAngularVelocity)
  );
  return;
}
`;

export { XSPHShader };