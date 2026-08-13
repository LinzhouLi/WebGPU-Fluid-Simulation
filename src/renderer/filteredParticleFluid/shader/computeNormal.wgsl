struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};

@group(0) @binding(0) var depthMap: texture_2d<f32>;
@group(0) @binding(1) var normalMap: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> camera: Camera;

fn loadViewPosition(coord: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
  if (any(coord < vec2<i32>(0)) || any(coord >= size)) {
    return vec4<f32>(0.0);
  }

  let depth = textureLoad(depthMap, coord, 0).r;
  if (depth <= 0.0) {
    return vec4<f32>(0.0);
  }

  let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(size);
  let positionEye = vec3<f32>(
    (uv - vec2<f32>(0.5)) * camera.params.xy * depth,
    -depth
  );
  return vec4<f32>(positionEye, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sizeU = textureDimensions(depthMap);
  if (any(gid.xy >= sizeU)) {
    return;
  }

  let size = vec2<i32>(sizeU);
  let coord = vec2<i32>(gid.xy);
  let center = loadViewPosition(coord, size);
  if (center.w == 0.0) {
    textureStore(normalMap, coord, vec4<f32>(0.0));
    return;
  }

  let left = loadViewPosition(coord - vec2<i32>(1, 0), size);
  let right = loadViewPosition(coord + vec2<i32>(1, 0), size);
  let top = loadViewPosition(coord - vec2<i32>(0, 1), size);
  let bottom = loadViewPosition(coord + vec2<i32>(0, 1), size);

  var tangentX = vec3<f32>(0.0);
  var hasTangentX = false;
  if (left.w > 0.0 && right.w > 0.0) {
    let backward = center.xyz - left.xyz;
    let forward = right.xyz - center.xyz;
    tangentX = select(forward, backward, abs(backward.z) < abs(forward.z));
    hasTangentX = true;
  } else if (left.w > 0.0) {
    tangentX = center.xyz - left.xyz;
    hasTangentX = true;
  } else if (right.w > 0.0) {
    tangentX = right.xyz - center.xyz;
    hasTangentX = true;
  }

  var tangentY = vec3<f32>(0.0);
  var hasTangentY = false;
  if (top.w > 0.0 && bottom.w > 0.0) {
    let backward = center.xyz - top.xyz;
    let forward = bottom.xyz - center.xyz;
    tangentY = select(forward, backward, abs(backward.z) < abs(forward.z));
    hasTangentY = true;
  } else if (top.w > 0.0) {
    tangentY = center.xyz - top.xyz;
    hasTangentY = true;
  } else if (bottom.w > 0.0) {
    tangentY = bottom.xyz - center.xyz;
    hasTangentY = true;
  }

  var normalEye = vec3<f32>(0.0, 0.0, 1.0);
  if (hasTangentX && hasTangentY) {
    let tangentCross = cross(tangentX, tangentY);
    let lengthSqr = dot(tangentCross, tangentCross);
    if (lengthSqr > 1e-24) {
      normalEye = tangentCross * inverseSqrt(lengthSqr);
      if (normalEye.z < 0.0) {
        normalEye = -normalEye;
      }
    }
  }

  textureStore(normalMap, coord, vec4<f32>(normalEye, 1.0));
}
