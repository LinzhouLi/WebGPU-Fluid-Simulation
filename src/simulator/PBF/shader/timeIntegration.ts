import { PBFConfig } from '../PBFConfig';
import { Boundary } from './common';


const TimeIntegrationShader = /* wgsl */`
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
override ParticleCount: u32;
override DeltaT: f32;
${Boundary}

@group(0) @binding(0) var<storage, read_write> position: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> position2: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> acceleration: array<vec3<f32>>;
@group(0) @binding(4) var<uniform> gravity: vec3<f32>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  // semi implicit euler time integration
  let vel = velocity[particleIndex] + DeltaT * (gravity + acceleration[particleIndex]);
  // let vel = velocity[particleIndex] + DeltaT * (gravity);
  let pos = boundary_rand(position[particleIndex] + DeltaT * vel);
  position2[particleIndex] = pos;

  return;
}
`;

export { TimeIntegrationShader };