import { wgsl } from '../3rd-party/wgsl-preprocessor';

function P2GComputeShader(n_particle: number, n_grid: number) {

const P2GScattering = wgsl/* wgsl */`
  dpos = (vec3<f32>(vec3<u32>(i, j, k)) - fx) * dx;
  weight = w[i][0] * w[j][1] * w[k][2];
  i = i + base.x; j = j + base.y; k = k + base.z; 

  if ( i >= 0 && i < ${n_grid} && j >= 0 && j < ${n_grid} && k >= 0 && k < ${n_grid} ) {
    v_del = weight * (p_mass * v_p + affine * dpos);
    m_del = weight * p_mass;

    loop {
      let atomic_storage_ptr = &(grid[i][j][k][0]);
      old_val= bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.x;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(grid[i][j][k][1]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.y;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(grid[i][j][k][2]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + v_del.z;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) { break; }
    }

    loop {
      let atomic_storage_ptr = &(grid[i][j][k][3]);
      old_val = bitcast<f32>(atomicLoad(atomic_storage_ptr));
      new_val = old_val + m_del;
      if ( 
        atomicCompareExchangeWeak(
          atomic_storage_ptr, bitcast<i32>(old_val), bitcast<i32>(new_val)
        ).exchanged
      ) {
        break; 
      }
    }
  }
`;

  return wgsl/* wgsl */`

override dx: f32;
override dt: f32;
override p_vol: f32;
override p_mass: f32;
override E: f32;

struct Particle {
  v: vec4<f32>,
  C: mat3x3<f32>
};

@group(0) @binding(0) var<storage, read_write> x: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(1) var<storage, read_write> particle_set: array<Particle, ${n_particle}>;
@group(0) @binding(2) var<storage, read_write> grid: array<array<array<array<atomic<i32>, 4>, ${n_grid}>, ${n_grid}>, ${n_grid}>;

const identity_mat3x3 = mat3x3<f32>(
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0
);

@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let p = global_index.x;
  if (p >= ${n_particle}) { return; }

  let particle = particle_set[p];
  let x_p = x[p];
  let v_p = particle.v.xyz;
  let C_p = particle.C;
  let J_p = particle.v.w * (1.0 + dt * (C_p[0][0] + C_p[1][1] + C_p[2][2]));

  let Xp = x_p / dx;
  let base = vec3<u32>(Xp - 0.5);
  let fx = Xp - vec3<f32>(base);
  let w = mat3x3<f32>(
    0.5 * (1.5 - fx) * (1.5 - fx), 
    0.75 - (fx - 1.0) * (fx - 1.0), 
    0.5 * (fx - 0.5) * (fx - 0.5)
  );

  let stress = -dt * 4.0 * E * p_vol * (J_p - 1.0) / (dx * dx);
  let affine = stress * identity_mat3x3 + p_mass * C_p;

  var i: u32; var j: u32; var k: u32;
  var weight: f32; var dpos: vec3<f32>;
  var old_val: f32; var new_val: f32;
  var v_del: vec3<f32>; var m_del: f32; 

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

  particle_set[p].v.w = J_p;

}

  `;
}

function GridComputeShader(n_particle: number, n_grid: number) {

  return wgsl/* wgsl */`

override dt: f32;
override bound: u32;

@group(0) @binding(0) var<storage, read_write> grid: array<array<array<vec4<f32>, ${n_grid}>, ${n_grid}>, ${n_grid}>;
@group(0) @binding(1) var<storage, read_write> gravity: vec4<f32>;

@compute @workgroup_size(2, 2, 2)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let i = global_index.x; let j = global_index.y; let k = global_index.z;
  if (i >= ${n_grid} || j >= ${n_grid} || k >= ${n_grid}) { return; }
  let grid_node = grid[i][j][k];

  // momentum -> velocity
  var velocity = vec4<f32>();
  if (grid_node.w > 0.0) { 
    velocity = grid_node / grid_node.w; 
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

  grid[i][j][k] = velocity;

}

  `;
}

function G2PComputeShader(n_particle: number, n_grid: number) {

const G2PGathering = wgsl/* wgsl */`
  dpos = (vec3<f32>(vec3<u32>(i, j, k)) - fx) * dx;
  weight = w[i][0] * w[j][1] * w[k][2];
  i = i + base.x; j = j + base.y; k = k + base.z; 
  velocity = grid[i][j][k];
  new_v = new_v + weight * velocity;
  new_C = new_C + inv4_dx2 * weight * mat3x3<f32>(
    velocity.x * dpos,
    velocity.y * dpos,
    velocity.z * dpos
  );
`;

  return wgsl/* wgsl */`

override dx: f32;
override dt: f32;

struct Particle {
  v: vec3<f32>,
  C: mat3x3<f32>
};

@group(0) @binding(0) var<storage, read_write> x: array<vec3<f32>, ${n_particle}>;
@group(0) @binding(1) var<storage, read_write> particle_set: array<Particle, ${n_particle}>;
@group(0) @binding(2) var<storage, read_write> grid: array<array<array<vec3<f32>, ${n_grid}>, ${n_grid}>, ${n_grid}>;

@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  let p = global_index.x;
  if (p >= ${n_particle}) { return; }

  let x_p = x[p];
  let Xp = x_p / dx;
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

  particle_set[p].v = new_v;
  particle_set[p].C = new_C;
  x[p] = x_p + dt * new_v;

}

`;
}

export { P2GComputeShader, GridComputeShader, G2PComputeShader };