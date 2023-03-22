const THREAD_COUNT = 256;


const ScanShaderCode = /* wgsl */`
const THREAD_COUNT: u32 = ${THREAD_COUNT};
const DOUBLE_THREAD_COUNT: u32 = 2 * THREAD_COUNT;
const BANK_COUNT: u32 = 32;
const LOG_BANK_COUNT: u32 = 5;
const SHARED_DATA_SIZE: u32 = DOUBLE_THREAD_COUNT + ((DOUBLE_THREAD_COUNT - 1) >> LOG_BANK_COUNT);

@group(0) @binding(0) var<storage, read_write> scanSource: array<u32>;
@group(0) @binding(1) var<storage, read_write> scanResult: array<u32>;

var<workgroup> sharedData: array<u32, SHARED_DATA_SIZE>;

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main(
  @builtin(local_invocation_index) thread_id: u32,
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) block_id: vec3<u32>
) {

  let arrayIndex = global_id.x * 2;
  
  var ai: u32 = thread_id * 2;
  var bi: u32 = ai + 1;
  sharedData[ai + (ai >> LOG_BANK_COUNT)] = scanSource[arrayIndex]; // Avoid Bank Conflicts
  sharedData[bi + (bi >> LOG_BANK_COUNT)] = scanSource[arrayIndex + 1];

  // up sweep
  var offset: u32 = 1;
  var d: u32;
  for (d = THREAD_COUNT; d > 0; d >>= 1) {
    workgroupBarrier();
    if (thread_id < d) {
      ai = offset * (2 * thread_id + 1) - 1;
      bi = ai + offset;
      ai += ai >> LOG_BANK_COUNT; // Avoid Bank Conflicts
      bi += bi >> LOG_BANK_COUNT;
      sharedData[bi] += sharedData[ai];
    }
    offset <<= 1;
  }

  if (thread_id == 0) { sharedData[SHARED_DATA_SIZE - 1] = 0; }

  // down sweep
  var temp: u32;
  for (d = 1; d < DOUBLE_THREAD_COUNT; d <<= 1) {
    offset >>= 1;
    workgroupBarrier();
    if (thread_id < d) {
      ai = offset * (2 * thread_id + 1) - 1;
      bi = ai + offset;
      ai += ai >> LOG_BANK_COUNT; // Avoid Bank Conflicts
      bi += bi >> LOG_BANK_COUNT;

      temp = sharedData[ai];
      sharedData[ai] = sharedData[bi];
      sharedData[bi] += temp;
    }
  }

  workgroupBarrier();
  ai = thread_id * 2;
  bi = ai + 1;
  scanResult[arrayIndex] = sharedData[ai + (ai >> LOG_BANK_COUNT)]; // Avoid Bank Conflicts
  scanResult[arrayIndex + 1] = sharedData[bi + (bi >> LOG_BANK_COUNT)];

}
`;


const CopyShaderCode = /* wgsl */`
const THREAD_COUNT: u32 = ${THREAD_COUNT};
const STRIDE = 2 * THREAD_COUNT;

@group(0) @binding(0) var<storage, read_write> scanSource: array<u32>;
@group(0) @binding(1) var<storage, read_write> scanResult: array<u32>;
@group(1) @binding(0) var<storage, read_write> tempScanSource: array<u32>;

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main ( @builtin(global_invocation_id) global_id: vec3<u32> ) {
  let tempScanArrayIndex = global_id.x;
  let scanArrayIndex = STRIDE * (tempScanArrayIndex + 1) - 1;
  // block sum = block exclusive scan result[-1] + block[-1]
  tempScanSource[tempScanArrayIndex] = scanResult[scanArrayIndex] + scanSource[scanArrayIndex];
}
`;


const GathreShaderCode = /* wgsl */`
const THREAD_COUNT: u32 = ${THREAD_COUNT};
const STRIDE = 2 * THREAD_COUNT;

@group(0) @binding(1) var<storage, read_write> tempScanResult: array<u32>;
@group(1) @binding(1) var<storage, read_write> scanResult: array<u32>;

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn main (
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) block_id: vec3<u32>
) {
  let arrayIndex = global_id.x;
  let blockIndex = block_id.x >> 1; // block_id.x / 2
  scanResult[arrayIndex] += tempScanResult[blockIndex];
}
`;


export { ScanShaderCode, CopyShaderCode, GathreShaderCode };