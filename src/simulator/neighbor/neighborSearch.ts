import { device } from '../../controller';
import { ExclusiveScan } from './exclusiveScan';
import { ParticleInsertShader, CountingSortShader, NeighborListShader } from './neighborShader';

class NeighborSearch {

  public static MIN_SEARCH_RADIUS = 1.0 / Math.floor( Math.cbrt(ExclusiveScan.MAX_ARRAY_LENGTH) );

  private searchRadius: number;
  private particleCount: number;
  private gridDimension: number;
  private gridCellCount: number;
  private gridCellCountAlignment: number;

  private cellParticleCount: GPUBuffer;
  private cellOffset: GPUBuffer;
  private particleSortIndex: GPUBuffer;
  private gridInfo: GPUBuffer;
  // private debugBuffer: GPUBuffer;

  private bindGroup: GPUBindGroup;

  private particleInsertPipeline: GPUComputePipeline;
  private countingSortPipeline: GPUComputePipeline;
  private neighborListPipeline: GPUComputePipeline;

  private scan: ExclusiveScan;

  constructor(
    particleCount: number,
    searchRadius: number
  ) {

    this.searchRadius = searchRadius;
    this.particleCount = particleCount;

    this.gridDimension = Math.ceil(1.0 / this.searchRadius);
    this.gridCellCount = Math.pow(this.gridDimension, 3);
    this.gridCellCountAlignment = Math.ceil(this.gridCellCount / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT;

    if (searchRadius < NeighborSearch.MIN_SEARCH_RADIUS)
      throw new Error(`Search Radius should greater than ${NeighborSearch.MIN_SEARCH_RADIUS}!`);

  }

  private createStorageData() {

    // create GPU Buffers
    // grid cell buffer x2
    const cellBufferDesp = {
      size: this.gridCellCountAlignment * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    } as GPUBufferDescriptor;
    this.cellParticleCount = device.createBuffer(cellBufferDesp);
    this.cellOffset = device.createBuffer(cellBufferDesp);

    // particle buffer x1
    const particleBufferDesp = {
      size: this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    } as GPUBufferDescriptor;
    this.particleSortIndex = device.createBuffer(particleBufferDesp);

    // grid info buffer
    this.gridInfo = device.createBuffer({
      size: 2 * 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const gridInfoArray = new Uint32Array(8);
    gridInfoArray.set([
      this.gridDimension, this.gridDimension, this.gridDimension, 0,
      this.gridDimension * this.gridDimension, this.gridDimension, 1, 0
    ]);
    device.queue.writeBuffer( this.gridInfo, 0, gridInfoArray, 0 );

    // this.debugBuffer = device.createBuffer({
    //   size: this.gridCellCount * Uint32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    // });

  }

  public async initResource(
    particlePosition: GPUBuffer, particlePositionSort: GPUBuffer,
    neighborList: GPUBuffer
  ) {

    this.createStorageData();

    // bind group
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ]
    });

    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: particlePosition } },
        { binding: 1, resource: { buffer: particlePositionSort } },
        { binding: 2, resource: { buffer: neighborList } },
        { binding: 3, resource: { buffer: this.cellParticleCount } },
        { binding: 4, resource: { buffer: this.cellOffset } },
        { binding: 5, resource: { buffer: this.particleSortIndex } },
        { binding: 6, resource: { buffer: this.gridInfo } },
      ]
    });

    // pipeline
    this.particleInsertPipeline = await device.createComputePipelineAsync({
      label: 'Particle Insert Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: ParticleInsertShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount
        }
      }
    });

    this.countingSortPipeline = await device.createComputePipelineAsync({
      label: 'Counting Sort Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: CountingSortShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount
        }
      }
    });

    this.neighborListPipeline = await device.createComputePipelineAsync({
      label: 'Neighbor List Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: NeighborListShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          SearchRadiusSqr: this.searchRadius * this.searchRadius
        }
      }
    })

    this.scan = new ExclusiveScan(this.cellParticleCount, this.cellOffset, this.gridCellCountAlignment);
    await this.scan.initResource();

  }

  public execute(passEncoder: GPUComputePassEncoder) {

    const workgroupCount = Math.ceil(this.particleCount / 64);
    passEncoder.setBindGroup(0, this.bindGroup);

    // particle insert
    passEncoder.setPipeline(this.particleInsertPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // exclusive scan: CellOffset[] = prefix sum of CellParticleCount[]
    this.scan.execute(passEncoder);

    passEncoder.setBindGroup(0, this.bindGroup);

    // counting sort
    passEncoder.setPipeline(this.countingSortPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // neighbor list
    passEncoder.setPipeline(this.neighborListPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

  }

  public async debug() {

    // const ce = device.createCommandEncoder();
    // ce.copyBufferToBuffer(
    //   this.cellOffset, 0,
    //   this.debugBuffer, 0,
    //   this.gridCellCount * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce.finish() ]);
    // await this.debugBuffer.mapAsync(GPUMapMode.READ);
    // const buffer = this.debugBuffer.getMappedRange();
    // const array = new Uint32Array(buffer);
    // console.log(array);

    // let t = [];
    // let x = 0;
    // let right = true;
    // for (let i = 6; i < 26; i++) {
    //   for (let j = 6; j < 26; j++) {
    //     for (let k = 6; k < 26; k++) {
    //       let idx = i*40*40 + j*40 + k;
    //       let m = array[idx];
    //       t.push([i,j,k, idx, m, x]);
    //       if (m!=x) { right = false; }
    //       x += 8;
    //     }
    //   }
    // }
    // console.log(t)
    // console.log(right)

    await this.scan.debug()

  }

}

export { NeighborSearch };