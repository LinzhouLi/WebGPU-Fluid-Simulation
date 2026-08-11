const cameraStruct = /* wgsl */`
struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};
`;

const billboardVertexShader = /* wgsl */`
${cameraStruct}

struct BillboardParams {
  radius: f32,
  worldSigma: f32,
  delta: f32,
  mu: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPosition: vec2<f32>,
  @location(1) @interpolate(flat) centerView: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> particlePositions: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: BillboardParams;

@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
  );
  let local = corners[vertexIndex];
  let center = camera.viewMatrix * vec4<f32>(particlePositions[instanceIndex].xyz, 1.0);
  let billboardView = center.xyz + vec3<f32>(local * params.radius, 0.0);
  var output: VertexOutput;
  output.position = camera.projectionMatrix * vec4<f32>(billboardView, 1.0);
  output.localPosition = local;
  output.centerView = center.xyz;
  return output;
}
`;

const particleDepthFragmentShader = /* wgsl */`
${cameraStruct}

struct BillboardParams {
  radius: f32,
  worldSigma: f32,
  delta: f32,
  mu: f32,
};

struct FragmentInput {
  @location(0) localPosition: vec2<f32>,
  @location(1) @interpolate(flat) centerView: vec3<f32>,
};

struct FragmentOutput {
  @location(0) eyeDepth: f32,
  @builtin(frag_depth) deviceDepth: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> params: BillboardParams;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {
  let radiusSquared = dot(input.localPosition, input.localPosition);
  if (radiusSquared > 1.0) { discard; }
  let sphereZ = sqrt(max(0.0, 1.0 - radiusSquared)) * params.radius;
  let surfaceView = input.centerView + vec3<f32>(input.localPosition * params.radius, sphereZ);
  let clip = camera.projectionMatrix * vec4<f32>(surfaceView, 1.0);
  var output: FragmentOutput;
  output.eyeDepth = surfaceView.z;
  output.deviceDepth = clip.z / clip.w;
  return output;
}
`;

const particleThicknessFragmentShader = /* wgsl */`
struct BillboardParams {
  radius: f32,
  worldSigma: f32,
  delta: f32,
  mu: f32,
};

struct FragmentInput {
  @location(0) localPosition: vec2<f32>,
  @location(1) @interpolate(flat) centerView: vec3<f32>,
};

@group(0) @binding(2) var<uniform> params: BillboardParams;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.localPosition, input.localPosition);
  if (radiusSquared > 1.0) { discard; }
  let thickness = 2.0 * params.radius * sqrt(max(0.0, 1.0 - radiusSquared));
  return vec4<f32>(thickness, 0.0, 0.0, 0.0);
}
`;

const narrowRangeFilterShader = /* wgsl */`
${cameraStruct}

struct FilterParams {
  worldSigma: f32,
  delta: f32,
  mu: f32,
  padding: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var inputDepth: texture_2d<f32>;
@group(0) @binding(2) var outputDepth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> filterParams: FilterParams;

const INVALID_DEPTH: f32 = -100000.0;
const MAX_RADIUS: i32 = 20;

fn isFluid(z: f32) -> bool { return z > INVALID_DEPTH; }

fn loadDepth(pixel: vec2<i32>, size: vec2<i32>) -> f32 {
  return textureLoad(inputDepth, clamp(pixel, vec2<i32>(0), size - vec2<i32>(1)), 0).r;
}

fn filterAt(pixel: vec2<i32>, direction: vec2<i32>) -> f32 {
  let dimensions = vec2<i32>(textureDimensions(inputDepth));
  let zi = loadDepth(pixel, dimensions);
  if (!isFluid(zi)) { return INVALID_DEPTH; }

  let sigma = clamp(
    f32(dimensions.y) * filterParams.worldSigma * camera.projectionMatrix[1][1]
      / (2.0 * abs(zi)),
    0.75, f32(MAX_RADIUS) / 3.0
  );
  let radius = i32(clamp(ceil(3.0 * sigma), 1.0, f32(MAX_RADIUS)));
  var weightedDepth = zi;
  var weightSum = 1.0;
  var deltaLow = filterParams.delta;
  var deltaHigh = filterParams.delta;

  for (var step = 1; step <= MAX_RADIUS; step = step + 1) {
    if (step <= radius) {
      let zj = loadDepth(pixel + direction * step, dimensions);
      let zk = loadDepth(pixel - direction * step, dimensions);
      if (isFluid(zj) && zj >= zi - deltaLow && zj <= zi + deltaHigh) {
        deltaLow = max(deltaLow, zi - zj + filterParams.delta);
        deltaHigh = max(deltaHigh, zj - zi + filterParams.delta);
      }
      if (isFluid(zk) && zk >= zi - deltaLow && zk <= zi + deltaHigh) {
        deltaLow = max(deltaLow, zi - zk + filterParams.delta);
        deltaHigh = max(deltaHigh, zk - zi + filterParams.delta);
      }
      let foregroundJ = isFluid(zj) && zj > zi + deltaHigh;
      let foregroundK = isFluid(zk) && zk > zi + deltaHigh;
      if (!foregroundJ && !foregroundK) {
        let x = f32(step);
        let weight = exp(-(x * x) / (2.0 * sigma * sigma));
        let fj = select(zi - filterParams.mu, zj, isFluid(zj) && zj >= zi - deltaLow);
        let fk = select(zi - filterParams.mu, zk, isFluid(zk) && zk >= zi - deltaLow);
        weightedDepth = weightedDepth + weight * (fj + fk);
        weightSum = weightSum + 2.0 * weight;
      }
    }
  }
  return weightedDepth / weightSum;
}

@compute @workgroup_size(8, 8)
fn horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(outputDepth);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let pixel = vec2<i32>(id.xy);
  textureStore(outputDepth, pixel, vec4<f32>(filterAt(pixel, vec2<i32>(1, 0)), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn vertical(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(outputDepth);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let pixel = vec2<i32>(id.xy);
  textureStore(outputDepth, pixel, vec4<f32>(filterAt(pixel, vec2<i32>(0, 1)), 0.0, 0.0, 0.0));
}
`;

const cleanupFilterShader = /* wgsl */`
struct FilterParams {
  worldSigma: f32,
  delta: f32,
  mu: f32,
  padding: f32,
};

@group(0) @binding(1) var inputDepth: texture_2d<f32>;
@group(0) @binding(2) var outputDepth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> filterParams: FilterParams;
const INVALID_DEPTH: f32 = -100000.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let sizeU = textureDimensions(outputDepth);
  if (id.x >= sizeU.x || id.y >= sizeU.y) { return; }
  let size = vec2<i32>(sizeU);
  let pixel = vec2<i32>(id.xy);
  let zi = textureLoad(inputDepth, pixel, 0).r;
  if (zi <= INVALID_DEPTH) {
    textureStore(outputDepth, pixel, vec4<f32>(INVALID_DEPTH, 0.0, 0.0, 0.0));
    return;
  }
  var sum = 0.0;
  var weightSum = 0.0;
  for (var y = -2; y <= 2; y = y + 1) {
    for (var x = -2; x <= 2; x = x + 1) {
      let samplePixel = clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1));
      let z = textureLoad(inputDepth, samplePixel, 0).r;
      if (!(z > zi + filterParams.delta)) {
        let distanceSquared = f32(x * x + y * y);
        let weight = exp(-distanceSquared * 0.5);
        let clampedDepth = select(zi - filterParams.mu, z, z > INVALID_DEPTH && z >= zi - filterParams.delta);
        sum = sum + weight * clampedDepth;
        weightSum = weightSum + weight;
      }
    }
  }
  textureStore(outputDepth, pixel, vec4<f32>(sum / max(weightSum, 0.0001), 0.0, 0.0, 0.0));
}
`;

const compositeVertexShader = /* wgsl */`
@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
}
`;

const compositeFragmentShader = /* wgsl */`
${cameraStruct}

struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>,
};

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var surfaceDepth: texture_2d<f32>;
@group(0) @binding(2) var thicknessTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var environmentMap: texture_cube<f32>;
@group(0) @binding(5) var<uniform> light: DirectionalLight;
const INVALID_DEPTH: f32 = -100000.0;

fn eyePosition(pixel: vec2<i32>, eyeZ: f32, size: vec2<i32>) -> vec3<f32> {
  let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(size);
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return vec3<f32>(
    -eyeZ * ndc.x / camera.projectionMatrix[0][0],
    -eyeZ * ndc.y / camera.projectionMatrix[1][1],
    eyeZ
  );
}

fn validNeighbor(z: f32, center: f32) -> f32 {
  return select(center, z, z > INVALID_DEPTH && abs(z - center) < 0.12);
}

fn gammaEncode(linearColor: vec3<f32>) -> vec3<f32> {
  return pow(max(linearColor, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> FragmentOutput {
  let size = vec2<i32>(textureDimensions(surfaceDepth));
  let pixel = clamp(vec2<i32>(fragCoord.xy), vec2<i32>(0), size - vec2<i32>(1));
  let z = textureLoad(surfaceDepth, pixel, 0).r;
  if (z <= INVALID_DEPTH) { discard; }

  let leftZ  = validNeighbor(textureLoad(surfaceDepth, clamp(pixel + vec2<i32>(-1,  0), vec2<i32>(0), size - vec2<i32>(1)), 0).r, z);
  let rightZ = validNeighbor(textureLoad(surfaceDepth, clamp(pixel + vec2<i32>( 1,  0), vec2<i32>(0), size - vec2<i32>(1)), 0).r, z);
  let upZ    = validNeighbor(textureLoad(surfaceDepth, clamp(pixel + vec2<i32>( 0, -1), vec2<i32>(0), size - vec2<i32>(1)), 0).r, z);
  let downZ  = validNeighbor(textureLoad(surfaceDepth, clamp(pixel + vec2<i32>( 0,  1), vec2<i32>(0), size - vec2<i32>(1)), 0).r, z);

  let centerView = eyePosition(pixel, z, size);
  let dx = eyePosition(pixel + vec2<i32>(1, 0), rightZ, size)
         - eyePosition(pixel + vec2<i32>(-1, 0), leftZ, size);
  let dy = eyePosition(pixel + vec2<i32>(0, -1), upZ, size)
         - eyePosition(pixel + vec2<i32>(0, 1), downZ, size);
  let normalView = normalize(cross(dx, dy));

  let worldPosition = (camera.viewMatrixInverse * vec4<f32>(centerView, 1.0)).xyz;
  var normalWorld = normalize((camera.viewMatrixInverse * vec4<f32>(normalView, 0.0)).xyz);
  let viewDirection = normalize(camera.position - worldPosition);
  if (dot(normalWorld, viewDirection) < 0.0) { normalWorld = -normalWorld; }

  let incident = -viewDirection;
  let reflectedDirection = reflect(incident, normalWorld);
  var refractedDirection = refract(incident, normalWorld, 1.0 / 1.333);
  if (dot(refractedDirection, refractedDirection) < 0.001) { refractedDirection = reflectedDirection; }
  let reflected = textureSampleLevel(environmentMap, linearSampler, reflectedDirection, 0.0).rgb;
  let refracted = textureSampleLevel(environmentMap, linearSampler, refractedDirection, 0.0).rgb;
  let uv = fragCoord.xy / vec2<f32>(size);
  let thickness = textureSampleLevel(thicknessTexture, linearSampler, uv, 0.0).r;
  let transmission = exp(-vec3<f32>(5.0, 1.4, 0.55) * thickness);
  let waterBodyColor = vec3<f32>(0.015, 0.18, 0.24);
  let refractionColor = refracted * transmission + waterBodyColor * (1.0 - transmission);

  let NoV = clamp(dot(normalWorld, viewDirection), 0.0, 1.0);
  let fresnel = 0.02037 + (1.0 - 0.02037) * pow(1.0 - NoV, 5.0);
  let halfVector = normalize(viewDirection + normalize(light.direction));
  let sunSpecular = pow(max(dot(normalWorld, halfVector), 0.0), 120.0) * light.color * 0.35;
  let linearColor = mix(refractionColor, reflected, fresnel) + sunSpecular;

  let clip = camera.projectionMatrix * vec4<f32>(centerView, 1.0);
  var output: FragmentOutput;
  output.color = vec4<f32>(gammaEncode(linearColor), 1.0);
  output.depth = clip.z / clip.w;
  return output;
}
`;

export {
  billboardVertexShader,
  particleDepthFragmentShader,
  particleThicknessFragmentShader,
  narrowRangeFilterShader,
  cleanupFilterShader,
  compositeVertexShader,
  compositeFragmentShader,
};
