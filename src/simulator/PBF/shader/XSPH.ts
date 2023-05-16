import { ShaderStruct } from '../../../common/shader';
import { PBFConfig } from '../PBFConfig';
import { KernalPoly6, KernalCohesion, KernalSpikyGrad } from './common';


const XSPHShader = /* wgsl */`
const PI: f32 = ${Math.PI};
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const InvDeltaT: f32 = ${1 / PBFConfig.TIME_STEP};

override ParticleWeight: f32;
override DoubleDensity0: f32;

${ShaderStruct.SimulationOptions}
${KernalPoly6}
${KernalCohesion}
${KernalSpikyGrad}


@group(0) @binding(0) var<uniform> options: SimulationOptions;

@group(1) @binding(0) var<storage, read> neighborOffset: array<u32>;
@group(1) @binding(1) var<storage, read> neighborList: array<u32>;

@group(2) @binding(0) var<storage, read_write> position_density: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> angularVelocity: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> velocity: array<vec3<f32>>;
@group(2) @binding(3) var<storage, read_write> acceleration: array<vec3<f32>>;
@group(2) @binding(4) var<storage, read_write> normal: array<vec3<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= options.particleCount) { return; }

  let selfPosition_density = position_density[particleIndex];
  let selfPosition = selfPosition_density.xyz;
  let selfDensity = selfPosition_density.w;
  let selfVelocity = velocity[particleIndex];
  let selfAngularVelocity = angularVelocity[particleIndex].xyz;
  let selfNormal = normal[particleIndex];

  var nListIndex = neighborOffset[particleIndex];
  let nListIndexEnd = neighborOffset[particleIndex + 1];
  var nParticleIndex = u32();

  var pos_dens: vec4<f32>;
  var positionDelta: vec3<f32>;
  var positionDeltaLength: f32;
  var velocityDelta: vec3<f32>;
  var Kij: f32;

  var velocityUpdate = vec3<f32>();
  var etai = vec3<f32>();
  var tension = vec3<f32>();

  while(nListIndex < nListIndexEnd) {
    nParticleIndex = neighborList[nListIndex];

    pos_dens = position_density[nParticleIndex];
    positionDelta = selfPosition - pos_dens.xyz; // xi - xj
    positionDeltaLength = length(positionDelta);
    velocityDelta = velocity[nParticleIndex] - selfVelocity; // vj - vi
    Kij = DoubleDensity0 / (selfDensity + pos_dens.w); // 2 * rho_0 / (rho_i + rho_j)

    // 1 / rho_j * v_ji * W_ij
    velocityUpdate += velocityDelta * kernalPoly6(positionDeltaLength) / pos_dens.w;

    // 1 / rho_j * ||omega|| * dW_ij
    etai += angularVelocity[nParticleIndex].w * kernalSpikyGrad(positionDelta, positionDeltaLength) / pos_dens.w;
    
    // a_tension = K_ij * (a_cohesion + a_curvature)
    tension -= Kij * (
      // cohesion     m * normaliz(x_ij) * W(x_ij)
      ParticleWeight * kernalCohesion(positionDelta, positionDeltaLength) +
      // curvature    n_ij
      (selfNormal - normal[nParticleIndex])
    );

    nListIndex++;
  }

  let etai_norm = select(vec3<f32>(0.0), normalize(etai), length(etai) > 0.0);

  acceleration[particleIndex] = (
    options.XSPHCoef * InvDeltaT * ParticleWeight * velocityUpdate +
    options.vorticityCoef * cross(etai_norm, selfAngularVelocity) +
    options.tensionCoef * tension
  );

  return;
}
`;

export { XSPHShader };