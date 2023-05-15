import { PBFConfig } from '../PBFConfig';
import { KernalSpikyGrad } from './common';


const  VorticityConfinementShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const EPS: f32 = 1e-6;
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};

override ParticleCount: u32;
override ParticleVolume: f32;

${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(0) @binding(1) var<storage, read> neighborList: array<u32>;

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
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var velocityDelta: vec3<f32>;
  var omegai = vec3<f32>();

  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    positionDelta = selfPosition - position[nParticleIndex]; // xi - xj
    positionDeltaLength = length(positionDelta);
    velocityDelta = velocity[nParticleIndex] - selfVelocity; // vj - vi

    omegai += cross(
      velocityDelta,
      kernalSpikyGrad(positionDelta, positionDeltaLength)
    );

    nListIndex++;
  }

  omegai *= ParticleVolume;
  angularVelocity[particleIndex] = vec4<f32>(omegai, length(omegai));
  return;
}
`;

export { VorticityConfinementShader };