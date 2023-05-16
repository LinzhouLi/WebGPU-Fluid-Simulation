import { ShaderStruct } from '../../../common/shader';
import { PBFConfig } from '../PBFConfig';
import { BoundaryModel } from '../../boundary/volumeMap';
import { DiscreteField, ShapeFunction, Interpolation } from '../../boundary/discreteFieldShader';


const BoundaryVolumeShader = /* wgsl */`
const KernelRadius: f32 = ${PBFConfig.KERNEL_RADIUS};
const GridSize: vec3<f32> = vec3<f32>(${BoundaryModel.RESOLUTION[0]}, ${BoundaryModel.RESOLUTION[1]}, ${BoundaryModel.RESOLUTION[2]});
const GridSizeU: vec3<u32> = vec3<u32>(GridSize);
const GridSpaceSize: vec3<f32> = 1.0 / GridSize;

override ParticleRadius: f32;

${ShaderStruct.SimulationOptions}
${DiscreteField}
${ShapeFunction}
${Interpolation}

@group(0) @binding(0) var<uniform> options: SimulationOptions;

@group(2) @binding(0) var<storage, read_write> position2: array<vec3<f32>>;
@group(2) @binding(3) var<storage, read_write> boundaryData: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read> field: DiscreteField;

@compute @workgroup_size(256, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= options.particleCount) { return; }

  let x = position2[particleIndex];
  var N: array<vec4<f32>, 8>;
  var dN: array<mat4x3<f32>, 8>;
  var bData = vec4<f32>();

  getShapeFunction(x, &N, &dN);
  let normal_dist = interpolateSDF(x, &N, &dN);
  let dist = normal_dist.w;

  if (dist > 0.0 && dist < KernelRadius && length(normal_dist.xyz) > 1e-9) {
    let volume = interpolateVolumeMap(x, &N);
    if (volume > 0.0) {
      let normal = normalize(normal_dist.xyz);
      let d = max(                      // boundary point is 0.5 * ParticleRadius below the surface. 
        dist + 0.5 * ParticleRadius,    // Ensure that the particle is at least one particle diameter away 
        2.0 * ParticleRadius            // from the boundary X to avoid strong pressure forces.
      );
      bData = vec4<f32>(
        x - d * normal,
        volume
      );
    }
  }
  
  boundaryData[particleIndex] = bData;
  return;
}
`;

export { BoundaryVolumeShader }