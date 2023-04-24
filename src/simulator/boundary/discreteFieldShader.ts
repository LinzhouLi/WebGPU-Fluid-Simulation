const DiscreteField = /* wgsl */`
struct DiscreteField {
  corner: array< array< array< f32, GridSizeU.x + 1 > , GridSizeU.y + 1 >, GridSizeU.z + 1 >,
  edgex: array< array< array< f32, GridSizeU.x << 1 > , GridSizeU.y + 1 >, GridSizeU.z + 1 >,
  edgey: array< array< array< f32, GridSizeU.y << 1 > , GridSizeU.z + 1 >, GridSizeU.x + 1 >,
  edgez: array< array< array< f32, GridSizeU.z << 1 > , GridSizeU.x + 1 >, GridSizeU.y + 1 >
};
`;


const ShapeFunction = /* wgsl */`
fn getShapeFunction(
  xi: vec3<f32>, 
  N: ptr< function, array<vec4<f32>, 8> >,
  dN: ptr< function, array<mat4x3<f32>, 8> >
) {

  const _1d64 = f32(1.0 / 64.0);
  const _9d64 = f32(9.0 / 64.0);
  const _19d64 = f32(19.0 / 64.0);

  let s = 2.0 * fract(xi * GridSize) - 1.0; // x y z
	let s2 = s * s; // x2 y2 z2
  let s2sum = s2.x + s2.y + s2.z;

  let _1ms2 = 1.0 - s2; // _1mx2 _1my2 _1mz2
	let _1ms = 1.0 - s; // _1mx _1my _1mz
	let _1ps = 1.0 + s; // _1px _1py _1pz
	let _1m3s = 1.0 - 3.0 * s; // _1m3x _1m3y _1m3z
	let _1p3s = 1.0 + 3.0 * s; // _1p3x _1p3y _1p3z

	let _1mxt1my = _1ms.x * _1ms.y;
	let _1mxt1py = _1ms.x * _1ps.y;
	let _1pxt1my = _1ps.x * _1ms.y;
	let _1pxt1py = _1ps.x * _1ps.y;

	let _1mxt1mz = _1ms.x * _1ms.z;
	let _1mxt1pz = _1ms.x * _1ps.z;
	let _1pxt1mz = _1ps.x * _1ms.z;
	let _1pxt1pz = _1ps.x * _1ps.z;

	let _1myt1mz = _1ms.y * _1ms.z;
	let _1myt1pz = _1ms.y * _1ps.z;
	let _1pyt1mz = _1ps.y * _1ms.z;
	let _1pyt1pz = _1ps.y * _1ps.z;

	// Corner nodes.
  let fac = _9d64 * s2sum - _19d64;
	(*N)[0] = fac * _1ms.z * vec4<f32>(_1mxt1my, _1pxt1my, _1mxt1py, _1pxt1py);
	(*N)[1] = fac * _1ps.z * vec4<f32>(_1mxt1my, _1pxt1my, _1mxt1py, _1pxt1py);

	// Edge nodes.
  let _1mp3sx = vec4<f32>(_1m3s.x, _1p3s.x, _1m3s.x, _1p3s.x);
  let _1mp3sy = vec4<f32>(_1m3s.y, _1p3s.y, _1m3s.y, _1p3s.y);
  let _1mp3sz = vec4<f32>(_1m3s.z, _1p3s.z, _1m3s.z, _1p3s.z);
	(*N)[2] = _9d64 * _1ms2.x * _1mp3sx * vec4<f32>(_1myt1mz, _1myt1mz, _1myt1pz, _1myt1pz);
	(*N)[3] = _9d64 * _1ms2.x * _1mp3sx * vec4<f32>(_1pyt1mz, _1pyt1mz, _1pyt1pz, _1pyt1pz);
	(*N)[4] = _9d64 * _1ms2.y * _1mp3sy * vec4<f32>(_1mxt1mz, _1mxt1mz, _1pxt1mz, _1pxt1mz);
	(*N)[5] = _9d64 * _1ms2.y * _1mp3sy * vec4<f32>(_1mxt1pz, _1mxt1pz, _1pxt1pz, _1pxt1pz);
	(*N)[6] = _9d64 * _1ms2.z * _1mp3sz * vec4<f32>(_1mxt1my, _1mxt1my, _1mxt1py, _1mxt1py);
	(*N)[7] = _9d64 * _1ms2.z * _1mp3sz * vec4<f32>(_1pxt1my, _1pxt1my, _1pxt1py, _1pxt1py);

  let _3s2p = 2.0 * vec3<f32>(s2.x, s2.y, s2.z) + s2sum;
  let _9t3s2pm19 = 9.0 * _3s2p - 19.0; // _9t3x2py2pz2m19 _9tx2p3y2pz2m19 _9tx2py2p3z2m19
  let _2s = 2.0 * s; // _2x _2y _2z
  let _18s = 18.0 * s; // _18x _18y _18z
  let _3m9s2 = 3.0 - 9.0 * s2; // _3m9x2 _3m9y2 _3m9z2

  // Corner nodes.
  let _18sm9t3s2pm19 = _18s - _9t3s2pm19; // _18xm9t3x2py2pz2m19 _18ym9tx2p3y2pz2m19 _18zm9tx2py2p3z2m19
  let _18sp9t3s2pm19 = _18s + _9t3s2pm19; // _18xp9t3x2py2pz2m19 _18yp9tx2p3y2pz2m19 _18zp9tx2py2p3z2m19
  (*dN)[0] = _1d64 * mat4x3<f32>(
    vec3<f32>(_18sm9t3s2pm19.x * _1myt1mz, _18sm9t3s2pm19.y * _1mxt1mz, _18sm9t3s2pm19.z * _1mxt1my),
    vec3<f32>(_1myt1mz * _18sp9t3s2pm19.x, _18sm9t3s2pm19.y * _1pxt1mz, _18sm9t3s2pm19.z * _1pxt1my),
    vec3<f32>(_18sm9t3s2pm19.x * _1pyt1mz, _1mxt1mz * _18sp9t3s2pm19.y, _18sm9t3s2pm19.z * _1mxt1py),
    vec3<f32>(_1pyt1mz * _18sp9t3s2pm19.x, _1pxt1mz * _18sp9t3s2pm19.y, _18sm9t3s2pm19.z * _1pxt1py)
  );
  (*dN)[1] = _1d64 * mat4x3<f32>(
    vec3<f32>(_18sm9t3s2pm19.x * _1myt1pz, _18sm9t3s2pm19.y * _1mxt1pz, _1mxt1my * _18sp9t3s2pm19.z),
    vec3<f32>(_1myt1pz * _18sp9t3s2pm19.x, _18sm9t3s2pm19.y * _1pxt1pz, _1pxt1my * _18sp9t3s2pm19.z),
    vec3<f32>(_18sm9t3s2pm19.x * _1pyt1pz, _1mxt1pz * _18sp9t3s2pm19.y, _1mxt1py * _18sp9t3s2pm19.z),
    vec3<f32>(_1pyt1pz * _18sp9t3s2pm19.x, _1pxt1pz * _18sp9t3s2pm19.y, _1pxt1py * _18sp9t3s2pm19.z)
  );

  // Edge nodes.
  let _m3m9s2m2s = -_3m9s2 -  _2s; // _m3m9x2m2x _m3m9y2m2y _m3m9z2m2z
  let _p3m9s2m2s =  _3m9s2 -  _2s; // _p3m9x2m2x _p3m9y2m2y _p3m9z2m2z
  let _1ms2t1m3s =  _1ms2 * _1m3s; // _1mx2t1m3x _1my2t1m3y _1mz2t1m3z
  let _1ms2t1p3s =  _1ms2 * _1p3s; // _1mx2t1p3x _1my2t1p3y _1mz2t1p3z
  (*dN)[2] = _9d64 * mat4x3<f32>(
    vec3<f32>(_m3m9s2m2s.x * _1myt1mz, -_1ms2t1m3s.x * _1ms.z, -_1ms2t1m3s.x * _1ms.y),
    vec3<f32>(_p3m9s2m2s.x * _1myt1mz, -_1ms2t1p3s.x * _1ms.z, -_1ms2t1p3s.x * _1ms.y),
    vec3<f32>(_m3m9s2m2s.x * _1myt1pz, -_1ms2t1m3s.x * _1ps.z,  _1ms2t1m3s.x * _1ms.y),
    vec3<f32>(_p3m9s2m2s.x * _1myt1pz, -_1ms2t1p3s.x * _1ps.z,  _1ms2t1p3s.x * _1ms.y)
  );
  (*dN)[3] = _9d64 * mat4x3<f32>(
    vec3<f32>(_m3m9s2m2s.x * _1pyt1mz, _1ms2t1m3s.x * _1ms.z, -_1ms2t1m3s.x * _1ps.y),
    vec3<f32>(_p3m9s2m2s.x * _1pyt1mz, _1ms2t1p3s.x * _1ms.z, -_1ms2t1p3s.x * _1ps.y),
    vec3<f32>(_m3m9s2m2s.x * _1pyt1pz, _1ms2t1m3s.x * _1ps.z,  _1ms2t1m3s.x * _1ps.y),
    vec3<f32>(_p3m9s2m2s.x * _1pyt1pz, _1ms2t1p3s.x * _1ps.z,  _1ms2t1p3s.x * _1ps.y)
  );
  (*dN)[4] = _9d64 * mat4x3<f32>(
    vec3<f32>(-_1ms2t1m3s.y * _1ms.z, _m3m9s2m2s.y * _1mxt1mz, -_1ms2t1m3s.y * _1ms.x),
    vec3<f32>(-_1ms2t1p3s.y * _1ms.z, _p3m9s2m2s.y * _1mxt1mz, -_1ms2t1p3s.y * _1ms.x),
    vec3<f32>( _1ms2t1m3s.y * _1ms.z, _m3m9s2m2s.y * _1pxt1mz, -_1ms2t1m3s.y * _1ps.x),
    vec3<f32>( _1ms2t1p3s.y * _1ms.z, _p3m9s2m2s.y * _1pxt1mz, -_1ms2t1p3s.y * _1ps.x)
  );
  (*dN)[5] = _9d64 * mat4x3<f32>(
    vec3<f32>(-_1ms2t1m3s.y * _1ps.z, _m3m9s2m2s.y * _1mxt1pz, _1ms2t1m3s.y * _1ms.x),
    vec3<f32>(-_1ms2t1p3s.y * _1ps.z, _p3m9s2m2s.y * _1mxt1pz, _1ms2t1p3s.y * _1ms.x),
    vec3<f32>( _1ms2t1m3s.y * _1ps.z, _m3m9s2m2s.y * _1pxt1pz, _1ms2t1m3s.y * _1ps.x),
    vec3<f32>( _1ms2t1p3s.y * _1ps.z, _p3m9s2m2s.y * _1pxt1pz, _1ms2t1p3s.y * _1ps.x)
  );
  (*dN)[6] = _9d64 * mat4x3<f32>(
    vec3<f32>(-_1ms2t1m3s.z * _1ms.y, -_1ms2t1m3s.z * _1ms.x, _m3m9s2m2s.z * _1mxt1my),
    vec3<f32>(-_1ms2t1p3s.z * _1ms.y, -_1ms2t1p3s.z * _1ms.x, _p3m9s2m2s.z * _1mxt1my),
    vec3<f32>(-_1ms2t1m3s.z * _1ps.y,  _1ms2t1m3s.z * _1ms.x, _m3m9s2m2s.z * _1mxt1py),
    vec3<f32>(-_1ms2t1p3s.z * _1ps.y,  _1ms2t1p3s.z * _1ms.x, _p3m9s2m2s.z * _1mxt1py)
  );
  (*dN)[7] = _9d64 * mat4x3<f32>(
    vec3<f32>(_1ms2t1m3s.z * _1ms.y, -_1ms2t1m3s.z * _1ps.x, _m3m9s2m2s.z * _1pxt1my),
    vec3<f32>(_1ms2t1p3s.z * _1ms.y, -_1ms2t1p3s.z * _1ps.x, _p3m9s2m2s.z * _1pxt1my),
    vec3<f32>(_1ms2t1m3s.z * _1ps.y,  _1ms2t1m3s.z * _1ps.x, _m3m9s2m2s.z * _1pxt1py),
    vec3<f32>(_1ms2t1p3s.z * _1ps.y,  _1ms2t1p3s.z * _1ps.x, _p3m9s2m2s.z * _1pxt1py)
  );
}
`;


const Interpolation = /* wgsl */`
function interpolate(
  xi: vec3<f32>,
  field: DiscreteField,
  N: ptr< function, array<vec4<f32>, 8> >,
) -> f32 {
  let x = vec3<u32>(floor(xi * GridSize));
  let i = x.x; let j = x.y; let k = x.z;
  let i2 = i << 1; let j2 = j << 1; let k2 = k << 1; 

  let valvec0 = vec4<f32>(corner[k][j][i], corner[k][j][i + 1], corner[k][j + 1][i], corner[k][j + 1][i + 1]);
  let valvec1 = vec4<f32>(corner[k + 1][j][i], corner[k + 1][j][i + 1], corner[k + 1][j + 1][i], corner[k + 1][j + 1][i + 1]);

  let valvec2 = vec4<f32>(edgex[k][j][i2], edgex[k][j][i2 + 1], edgex[k + 1][j][i2], edgex[k + 1][j][i2]);
  let valvec3 = vec4<f32>(edgex[k][j + 1][i2], edgex[k][j + 1][i2 + 1], edgex[k + 1][j + 1][i2], edgex[k + 1][j + 1][i2]);

  let valvec4 = vec4<f32>(edgey[i][k][j2], edgey[i][k][j2 + 1], edgey[i + 1][k][j2], edgey[i + 1][k][j2 + 1]);
  let valvec5 = vec4<f32>(edgey[i][k + 1][j2], edgey[i][k + 1][j2 + 1], edgey[i + 1][k + 1][j2], edgey[i + 1][k + 1][j2 + 1]);

  let valvec6 = vec4<f32>(edgez[j][i][k2], edgez[j][i][k2], edgez[j + 1][i][k2], edgez[j + 1][i][k2]);
  let valvec7 = vec4<f32>(edgez[j][i + 1][k2], edgez[j][i + 1][k2], edgez[j + 1][i + 1][k2], edgez[j + 1][i + 1][k2]);

  let result0 = vec4<f32>(dot(valvec0, (*N)[0]), dot(valvec1, (*N)[1]), dot(valvec2, (*N)[2]), dot(valvec3, (*N)[3]));
  let result1 = vec4<f32>(dot(valvec4, (*N)[4]), dot(valvec5, (*N)[5]), dot(valvec6, (*N)[6]), dot(valvec7, (*N)[7]));
  let result2 = result0 + result1;
  return (result2.x + result2.y + result2.z + result2.w);
}

function interpolateWithGrad(
  xi: vec3<f32>,
  field: DiscreteField,
  N: ptr< function, array<vec4<f32>, 8> >,
  dN: ptr< function, array<mat4x3<f32>, 8> >
) -> vec4<f32> {
  let x = vec3<u32>(floor(xi * GridSize));
  let i = x.x; let j = x.y; let k = x.z;
  let i2 = i << 1; let j2 = j << 1; let k2 = k << 1; 

  let valvec0 = vec4<f32>(corner[k][j][i], corner[k][j][i + 1], corner[k][j + 1][i], corner[k][j + 1][i + 1]);
  let valvec1 = vec4<f32>(corner[k + 1][j][i], corner[k + 1][j][i + 1], corner[k + 1][j + 1][i], corner[k + 1][j + 1][i + 1]);

  let valvec2 = vec4<f32>(edgex[k][j][i2], edgex[k][j][i2 + 1], edgex[k + 1][j][i2], edgex[k + 1][j][i2]);
  let valvec3 = vec4<f32>(edgex[k][j + 1][i2], edgex[k][j + 1][i2 + 1], edgex[k + 1][j + 1][i2], edgex[k + 1][j + 1][i2]);

  let valvec4 = vec4<f32>(edgey[i][k][j2], edgey[i][k][j2 + 1], edgey[i + 1][k][j2], edgey[i + 1][k][j2 + 1]);
  let valvec5 = vec4<f32>(edgey[i][k + 1][j2], edgey[i][k + 1][j2 + 1], edgey[i + 1][k + 1][j2], edgey[i + 1][k + 1][j2 + 1]);

  let valvec6 = vec4<f32>(edgez[j][i][k2], edgez[j][i][k2], edgez[j + 1][i][k2], edgez[j + 1][i][k2]);
  let valvec7 = vec4<f32>(edgez[j][i + 1][k2], edgez[j][i + 1][k2], edgez[j + 1][i + 1][k2], edgez[j + 1][i + 1][k2]);

  let result0 = vec4<f32>(dot(valvec0, (*N)[0]), dot(valvec1, (*N)[1]), dot(valvec2, (*N)[2]), dot(valvec3, (*N)[3]));
  let result1 = vec4<f32>(dot(valvec4, (*N)[4]), dot(valvec5, (*N)[5]), dot(valvec6, (*N)[6]), dot(valvec7, (*N)[7]));
  let result2 = result0 + result1;
  let scaler_result = result2.x + result2.y + result2.z + result2.w;

  let result3 = (*dN)[0] * valvec0 + (*dN)[1] * valvec1 + (*dN)[2] * valvec2 + (*dN)[3] * valvec3;
  let result4 = (*dN)[4] * valvec4 + (*dN)[5] * valvec5 + (*dN)[6] * valvec6 + (*dN)[7] * valvec7;
  let grad_result = 2.0 * GridSize * (result3 + result4);
  return vec4<f32>(grad_result, scaler_result);
}
`;