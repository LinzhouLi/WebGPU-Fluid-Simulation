import { PBFConfig } from '../PBFConfig';
import { KernalSpikyGrad } from './common';


const  VorticityConfinementShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};

override ParticleCount: u32;
override ParticleWeight: f32;

${KernalSpikyGrad}

@group(0) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(0) @binding(1) var<storage, read> neighborList: array<u32>;

@group(1) @binding(0) var<storage, read_write> position_density: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> angularVelocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(1) @binding(4) var<storage, read_write> normal: array<vec3<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let selfPosition = position_density[particleIndex].xyz;
  let selfVelocity = velocity[particleIndex];
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var pos_dens: vec4<f32>;
  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var velocityDelta: vec3<f32>;
  var volume: f32;
  var gradW: vec3<f32>;

  var omegai = vec3<f32>();
  var norm = vec3<f32>();

  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    pos_dens = position_density[nParticleIndex];
    positionDelta = selfPosition - pos_dens.xyz; // xi - xj
    positionDeltaLength = length(positionDelta);
    velocityDelta = velocity[nParticleIndex] - selfVelocity; // vj - vi

    volume = ParticleWeight / pos_dens.w; // m / rho_j
    gradW = kernalSpikyGrad(positionDelta, positionDeltaLength); // dW_ij

    omegai += volume * cross(velocityDelta, gradW); // - m / rho_j * cross(v_ij, dW_ij)
    norm += volume * gradW; // - m / rho_j * dW_ij

    nListIndex++;
  }

  angularVelocity[particleIndex] = vec4<f32>(omegai, length(omegai));
  normal[particleIndex] = KernelRadius * norm;

  return;
}
`;

export { VorticityConfinementShader };