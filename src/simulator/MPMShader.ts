import { wgsl } from '../3rd-party/wgsl-preprocessor';

function P2GComputeShader(n_particle: number, n_grid: number) {

const P2GScattering = wgsl/* wgsl */`
  dpos = (vec3<f32>(vec3<u32>(i, j, k)) - fx) * dx;
  weight = w[i][0] * w[j][1] * w[k][2];
  i = i + base.x; j = j + base.y; k = k + base.z; 

  if ( i >= 0 && i < ${n_grid} && j >= 0 && j < ${n_grid} && k >= 0 && k < ${n_grid} ) {
    let v_del = weight * (p_mass * F_v[p] + affine * dpos);
    let m_del = weight * p_mass;

    var old_val: f32; var new_val: f32;
    loop {
      let atomic_storage_ptr = &(F_grid_v[i][j][k][0]);
      old_val= bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.x;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(F_grid_v[i][j][k][1]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.y;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(F_grid_v[i][j][k][2]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.z;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(F_grid_m[i][j][k]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + m_del;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }
  }
`;

  return wgsl/* wgsl */`

override dx: f32;
override dt: f32;
// override gravity: f32;
// override bound: u32;

// override p_rho: f32;
override p_vol: f32;
override p_mass: f32;
override E: f32;

@group(0) @binding(0) var<storage, read_write> F_x: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(1) var<storage, read_write> F_v: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(2) var<storage, read_write> F_C: array<mat3x3<f32>, ${n_particle}>;
@group(0) @binding(3) var<storage, read_write> F_J: array<f32, ${n_particle}>;

@group(0) @binding(4) var<storage, read_write> F_grid_v: array<array<array<array<atomic<i32>, 4>, ${n_grid}>, ${n_grid}>, ${n_grid}>;
@group(0) @binding(5) var<storage, read_write> F_grid_m: array<array<array<atomic<i32>, ${n_grid}>, ${n_grid}>, ${n_grid}>;

@group(0) @binding(6) var<storage, read_write> gravity: vec3<f32>;

const identity_mat3x3 = mat3x3<f32>(
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0
);

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let p = global_index.x;
  if (p >= ${n_particle}) { return; }

  let Xp = F_x[p] / dx;
  let base = vec3<u32>(Xp - 0.5);
  let fx = Xp - vec3<f32>(base);
  let w = mat3x3<f32>(
    0.5 * (1.5 - fx) * (1.5 - fx), 
    0.75 - (fx - 1.0) * (fx - 1.0), 
    0.5 * (fx - 0.5) * (fx - 0.5)
  );

  let stress = -dt * 4.0 * E * p_vol * (F_J[p] - 1.0) / (dx * dx);
  let affine = stress * identity_mat3x3 + p_mass * F_C[p];

  var i: u32; var j: u32; var k: u32;
  var weight: f32; var dpos: vec3<f32>;

  i = 0; j = 0; k = 0; ${P2GScattering}
  i = 0; j = 0; k = 1; ${P2GScattering}
  i = 0; j = 0; k = 2; ${P2GScattering}
  i = 0; j = 1; k = 0; ${P2GScattering}
  i = 0; j = 1; k = 1; ${P2GScattering}
  i = 0; j = 1; k = 2; ${P2GScattering}
  i = 0; j = 2; k = 0; ${P2GScattering}
  i = 0; j = 2; k = 1; ${P2GScattering}
  i = 0; j = 2; k = 2; ${P2GScattering}
  
  i = 1; j = 0; k = 0; ${P2GScattering}
  i = 1; j = 0; k = 1; ${P2GScattering}
  i = 1; j = 0; k = 2; ${P2GScattering}
  i = 1; j = 1; k = 0; ${P2GScattering}
  i = 1; j = 1; k = 1; ${P2GScattering}
  i = 1; j = 1; k = 2; ${P2GScattering}
  i = 1; j = 2; k = 0; ${P2GScattering}
  i = 1; j = 2; k = 1; ${P2GScattering}
  i = 1; j = 2; k = 2; ${P2GScattering}

  i = 2; j = 0; k = 0; ${P2GScattering}
  i = 2; j = 0; k = 1; ${P2GScattering}
  i = 2; j = 0; k = 2; ${P2GScattering}
  i = 2; j = 1; k = 0; ${P2GScattering}
  i = 2; j = 1; k = 1; ${P2GScattering}
  i = 2; j = 1; k = 2; ${P2GScattering}
  i = 2; j = 2; k = 0; ${P2GScattering}
  i = 2; j = 2; k = 1; ${P2GScattering}
  i = 2; j = 2; k = 2; ${P2GScattering}

}

  `;
}

function GridComputeShader(n_particle: number, n_grid: number) {

  return wgsl/* wgsl */`

override dt: f32;
override bound: u32;

@group(0) @binding(0) var<storage, read_write> F_x: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(1) var<storage, read_write> F_v: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(2) var<storage, read_write> F_C: array<mat3x3<f32>, ${n_particle}>;
@group(0) @binding(3) var<storage, read_write> F_J: array<f32, ${n_particle}>;

@group(0) @binding(4) var<storage, read_write> F_grid_v: array<array<array<vec3<f32>, ${n_grid}>, ${n_grid}>, ${n_grid}>;
@group(0) @binding(5) var<storage, read_write> F_grid_m: array<array<array<f32, ${n_grid}>, ${n_grid}>, ${n_grid}>;

@group(0) @binding(6) var<storage, read_write> gravity: vec3<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let i = global_index.x; let j = global_index.y; let k = global_index.z;
  if (i >= ${n_grid} || j >= ${n_grid} || k >= ${n_grid}) { return; }

  // momentum -> velocity
  var velocity = vec3<f32>();
  if (F_grid_m[i][j][k] > 0.0) { 
    velocity = F_grid_v[i][j][k] / F_grid_m[i][j][k]; 
  }

  // apply gravity
  velocity = velocity + dt * gravity;

  // apply boundary condition
  let max_bound = ${n_grid} - bound;
  if ( (i < bound && velocity.x < 0) || (i >= max_bound && velocity.x > 0) ) {
    velocity.x = 0.0;
  }
  if ( (j < bound && velocity.y < 0) || (j >= max_bound && velocity.y > 0) ) {
    velocity.y = 0.0;
  }
  if ( (k < bound && velocity.z < 0) || (k >= max_bound && velocity.z > 0) ) {
    velocity.z = 0.0;
  }

  F_grid_v[i][j][k] = velocity;

}

  `;
}

function G2PComputeShader(n_particle: number, n_grid: number) {

const G2PGathering = wgsl/* wgsl */`
  dpos = (vec3<f32>(vec3<u32>(i, j, k)) - fx) * dx;
  weight = w[i][0] * w[j][1] * w[k][2];
  i = i + base.x; j = j + base.y; k = k + base.z; 
  velocity = F_grid_v[i][j][k];
  new_v = new_v + weight * velocity;
  new_C = new_C + inv4_dx2 * weight * mat3x3<f32>(
    velocity[0] * dpos,
    velocity[1] * dpos,
    velocity[2] * dpos
  );
`;

  return wgsl/* wgsl */`

override dx: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> F_x: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(1) var<storage, read_write> F_v: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(2) var<storage, read_write> F_C: array<mat3x3<f32>, ${n_particle}>;
@group(0) @binding(3) var<storage, read_write> F_J: array<f32, ${n_particle}>;

@group(0) @binding(4) var<storage, read_write> F_grid_v: array<array<array<vec3<f32>, ${n_grid}>, ${n_grid}>, ${n_grid}>;
@group(0) @binding(5) var<storage, read_write> F_grid_m: array<array<array<f32, ${n_grid}>, ${n_grid}>, ${n_grid}>;

@group(0) @binding(6) var<storage, read_write> gravity: vec3<f32>;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let p = global_index.x;
  if (p >= ${n_particle}) { return; }

  let Xp = F_x[p] / dx;
  let base = vec3<u32>(Xp - 0.5);
  let fx = Xp - vec3<f32>(base);
  let w = mat3x3<f32>(
    0.5 * (1.5 - fx) * (1.5 - fx), 
    0.75 - (fx - 1.0) * (fx - 1.0), 
    0.5 * (fx - 0.5) * (fx - 0.5)
  );

  var i: u32; var j: u32; var k: u32;
  var new_v = vec3<f32>(); var new_C = mat3x3<f32>();
  var weight: f32; var dpos: vec3<f32>;
  var velocity: vec3<f32>;
  let inv4_dx2 = 4.0 / (dx * dx);

  i = 0; j = 0; k = 0; ${G2PGathering}
  i = 0; j = 0; k = 1; ${G2PGathering}
  i = 0; j = 0; k = 2; ${G2PGathering}
  i = 0; j = 1; k = 0; ${G2PGathering}
  i = 0; j = 1; k = 1; ${G2PGathering}
  i = 0; j = 1; k = 2; ${G2PGathering}
  i = 0; j = 2; k = 0; ${G2PGathering}
  i = 0; j = 2; k = 1; ${G2PGathering}
  i = 0; j = 2; k = 2; ${G2PGathering}
  
  i = 1; j = 0; k = 0; ${G2PGathering}
  i = 1; j = 0; k = 1; ${G2PGathering}
  i = 1; j = 0; k = 2; ${G2PGathering}
  i = 1; j = 1; k = 0; ${G2PGathering}
  i = 1; j = 1; k = 1; ${G2PGathering}
  i = 1; j = 1; k = 2; ${G2PGathering}
  i = 1; j = 2; k = 0; ${G2PGathering}
  i = 1; j = 2; k = 1; ${G2PGathering}
  i = 1; j = 2; k = 2; ${G2PGathering}

  i = 2; j = 0; k = 0; ${G2PGathering}
  i = 2; j = 0; k = 1; ${G2PGathering}
  i = 2; j = 0; k = 2; ${G2PGathering}
  i = 2; j = 1; k = 0; ${G2PGathering}
  i = 2; j = 1; k = 1; ${G2PGathering}
  i = 2; j = 1; k = 2; ${G2PGathering}
  i = 2; j = 2; k = 0; ${G2PGathering}
  i = 2; j = 2; k = 1; ${G2PGathering}
  i = 2; j = 2; k = 2; ${G2PGathering}

  F_v[p] = new_v;
  F_C[p] = new_C;
  F_x[p] = F_x[p] + dt * new_v;
  F_J[p] = F_J[p] * (1.0 + dt * (new_C[0][0] + new_C[1][1] + new_C[2][2]));

}

`;
}

export { P2GComputeShader, GridComputeShader, G2PComputeShader };