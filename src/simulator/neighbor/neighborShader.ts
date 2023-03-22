
const GridStruct = /* wgsl */`
struct Grid {
  dimension: vec3<u32>,
  coord2index: vec3<u32>
};
`;

const NeighborStruct = /* wgsl */`
struct Neighbor {
  count: u32,
  particleIndex: array<u32, MaxNeighborCount>
};
`;


const ParticleInsertShader = /* wgsl */`
override ParticleCount: u32;
${GridStruct}

@group(0) @binding(0) var<storage, read_write> particlePosition: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> particlePositionSort: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> particleVelocity: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> particleVelocitySort: array<vec3<f32>>;

@group(0) @binding(5) var<storage, read_write> cellParticleCount: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cellOffset: array<u32>;
@group(0) @binding(7) var<storage, read_write> particleSortIndex: array<u32>;
@group(0) @binding(8) var<uniform> grid: Grid;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let cellCoord = vec3<u32>(floor(particlePosition[particleIndex] * vec3<f32>(grid.dimension)));
  let cellIndex = dot(cellCoord, grid.coord2index);
  particleSortIndex[particleIndex] = atomicAdd(&(cellParticleCount[cellIndex]), 1u);
  return;
}
`;


const CountingSortShader = /* wgsl */`
override ParticleCount: u32;
${GridStruct}

@group(0) @binding(0) var<storage, read_write> particlePosition: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> particlePositionSort: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> particleVelocity: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> particleVelocitySort: array<vec3<f32>>;

@group(0) @binding(5) var<storage, read_write> cellParticleCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> cellOffset: array<u32>;
@group(0) @binding(7) var<storage, read_write> particleSortIndex: array<u32>;
@group(0) @binding(8) var<uniform> grid: Grid;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }
  let position = particlePosition[particleIndex];
  let cellCoord = vec3<u32>(floor(position * vec3<f32>(grid.dimension)));
  let cellIndex = dot(cellCoord, grid.coord2index);
  let sortIndex = particleSortIndex[particleIndex] + cellOffset[cellIndex];
  particlePositionSort[sortIndex] = position;
  particleVelocitySort[sortIndex] = particleVelocity[particleIndex];
  return;
}
`;


const NeighborSearch = /* wgsl */`
  nCoord += cellCoord;
  if ( all(nCoord >= vec3<i32>(0)) && all(nCoord < GridDim) ) {
    nCellIdx = dot(nCoord, GridCoord2Index);
    nCellParticleCount = cellParticleCount[nCellIdx];
    nParticleIdxSort = cellOffset[nCellIdx];
    for (
      nCellParticleIdx = 0; 
      nCellParticleIdx < nCellParticleCount;
      nCellParticleIdx++
    ) {
      positionDelta = particlePositionSort[nParticleIdxSort] - sParticlePosition;
      if (dot(positionDelta, positionDelta) <= SearchRadiusSqr) { // within search radius
        neighborList[particleIndex].particleIndex[neighborCount] = nParticleIdxSort;
        neighborCount++;
        if (neighborCount > MaxNeighborCount) {
          neighborList[particleIndex].count = MaxNeighborCount;
          return;
        }
      }
      nParticleIdxSort++;
    }
  }
`;


function NeighborListShader(maxNeighborCount: number) {
  return /* wgsl */`
override ParticleCount: u32;
override SearchRadiusSqr: f32;
const MaxNeighborCount: u32 = ${maxNeighborCount};
${GridStruct}
${NeighborStruct}

@group(0) @binding(1) var<storage, read_write> particlePositionSort: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read_write> neighborList: array<Neighbor>;

@group(0) @binding(5) var<storage, read_write> cellParticleCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> cellOffset: array<u32>;
@group(0) @binding(8) var<uniform> grid: Grid;

@compute @workgroup_size(64, 1, 1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let particleIndex = global_id.x;
  if (particleIndex >= ParticleCount) { return; }

  let GridDim = vec3<i32>(grid.dimension);
  let GridCoord2Index = vec3<i32>(grid.coord2index);
  let sParticlePosition = particlePositionSort[particleIndex];
  let cellCoord = vec3<i32>(floor(sParticlePosition * vec3<f32>(grid.dimension)));

  var nCoord = vec3<i32>();         // neighbor cell coord(3d)
  var nCellIdx = i32();             // neighbor cell index(1d)
  var nCellParticleIdx = u32();     // particle index in a grid cell
  var nCellParticleCount = u32();   // particle count in a grid cell
  var nParticleIdxSort = u32();     // sorted index of a neighbor particle
  var positionDelta = vec3<f32>();  // position difference between this particle and a neighbor particle
  var neighborCount = u32();        // neighbor particle count

  nCoord.x = -1; nCoord.y = -1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x = -1; nCoord.y = -1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x = -1; nCoord.y = -1; nCoord.z =  1; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  0; nCoord.z = -1; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  0; nCoord.z =  0; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  0; nCoord.z =  1; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x = -1; nCoord.y =  1; nCoord.z =  1; ${NeighborSearch}

  nCoord.x =  0; nCoord.y = -1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  0; nCoord.y = -1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  0; nCoord.y = -1; nCoord.z =  1; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  0; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  0; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  0; nCoord.z =  1; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  0; nCoord.y =  1; nCoord.z =  1; ${NeighborSearch}

  nCoord.x =  1; nCoord.y = -1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  1; nCoord.y = -1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  1; nCoord.y = -1; nCoord.z =  1; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  0; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  0; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  0; nCoord.z =  1; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  1; nCoord.z = -1; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  1; nCoord.z =  0; ${NeighborSearch}
  nCoord.x =  1; nCoord.y =  1; nCoord.z =  1; ${NeighborSearch}

  neighborList[particleIndex].count = neighborCount;
  return;
}
`;
}

export { ParticleInsertShader, CountingSortShader, NeighborListShader };