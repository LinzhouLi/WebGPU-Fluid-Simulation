import { ShaderStruct } from '../../../common/shader';
import { PBFConfig } from '../PBFConfig';
import { KernalPoly6 } from './common';


const AttributeUpdateShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const InvDeltaT: f32 = ${1 / PBFConfig.TIME_STEP};

override ParticleWeight: f32;

${ShaderStruct.SimulationOptions}
${KernalPoly6}

@group(0) @binding(0) var<uniform> options: SimulationOptions;

@group(1) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(1) @binding(1) var<storage, read> neighborList: array<u32>;

@group(2) @binding(0) var<storage, read_write> position_density: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> position2: array<vec3<f32>>;
@group(2) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= options.particleCount) { return; }

  let selfPosition = position2[particleIndex];
  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var density = f32();

  // fluid
  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    positionDelta = selfPosition - position2[nParticleIndex];
    positionDeltaLength = length(positionDelta);

    density += kernalPoly6(positionDeltaLength);

    nListIndex++;
  }
  density *= ParticleWeight;

  let oldPosition = position_density[particleIndex].xyz;
  let vel = (selfPosition - oldPosition) * InvDeltaT; // first order

  velocity[particleIndex] = vel;
  position_density[particleIndex] = vec4<f32>(selfPosition, density);

  return;
}
`;


export { AttributeUpdateShader };