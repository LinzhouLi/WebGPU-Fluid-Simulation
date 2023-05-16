import * as THREE from 'three';
import { device } from '../../controller';
import { SPH } from '../SPH';
import { PBFConfig } from '../PBF/PBFConfig';
import { ExclusiveScan } from './exclusiveScan';
import { ParticleInsertShader, CountingSortShader, NeighborCountShader, NeighborListShader } from './neighborShader';

class NeighborSearch {

  public static MIN_SEARCH_RADIUS = 1.0 / Math.floor( Math.cbrt(ExclusiveScan.MAX_ARRAY_LENGTH) );
  public static SEARCH_RADIUS = SPH.KERNEL_RADIUS;
  public static GRID_DIM = Math.ceil(1.0 / NeighborSearch.SEARCH_RADIUS);
  public static GRID_COUNT_ALIGNMENT = 
    Math.ceil(Math.pow(NeighborSearch.GRID_DIM, 3) / ExclusiveScan.ARRAY_ALIGNMENT) 
    * ExclusiveScan.ARRAY_ALIGNMENT;

  private simulator: SPH;

  // input
  private particlePosition: GPUBuffer;

  // output
  public neighborOffset: GPUBuffer;
  public neighborList: GPUBuffer;

  private neighborCount: GPUBuffer;
  private cellParticleCount: GPUBuffer;
  private cellOffset: GPUBuffer;
  private particleSortIndex: GPUBuffer;
  private particleSortIndexCopy: GPUBuffer;
  private gridInfo: GPUBuffer;

  private particleInsertBindGroupLayout: GPUBindGroupLayout;
  private countingBindGroupLayout: GPUBindGroupLayout;
  private neighborListBindGroupLayout: GPUBindGroupLayout;

  private particleInsertBindGroup: GPUBindGroup;
  private countingBindGroup: GPUBindGroup;
  private neighborListBindGroup: GPUBindGroup;

  private particleInsertPipeline: GPUComputePipeline;
  private countingSortPipeline: GPUComputePipeline;
  private neighborCountPipeline: GPUComputePipeline;
  private neighborListPipeline: GPUComputePipeline;

  private cellScan: ExclusiveScan;
  private particleScan: ExclusiveScan;

  private static debug = false;
  private debugBuffer1: GPUBuffer;
  private debugBuffer2: GPUBuffer;
  private debugBuffer3: GPUBuffer;

  constructor( simulator: SPH ) {

    this.simulator = simulator;

    if (NeighborSearch.SEARCH_RADIUS < NeighborSearch.MIN_SEARCH_RADIUS)
      throw new Error(`Search Radius should greater than ${NeighborSearch.MIN_SEARCH_RADIUS}!`);

  }

  public reset(commandEncoder: GPUCommandEncoder) {

    commandEncoder.clearBuffer(this.neighborCount);
    commandEncoder.clearBuffer(this.cellParticleCount);
    commandEncoder.clearBuffer(this.cellOffset);
    commandEncoder.clearBuffer(this.particleSortIndex);
    commandEncoder.clearBuffer(this.particleSortIndexCopy);

    this.cellScan.reset(commandEncoder);
    this.particleScan.reset(commandEncoder);

  }

  private createStorageData() {

    // create GPU Buffers
    // neighbor list buffer
    let bufferSize = SPH.MAX_NEIGHBOR_COUNT * SPH.MAX_PARTICAL_NUM * Uint32Array.BYTES_PER_ELEMENT
    this.neighborList = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    // neighbor count/offset buffer
    bufferSize = Math.ceil(SPH.MAX_PARTICAL_NUM / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT * Float32Array.BYTES_PER_ELEMENT;
    this.neighborCount = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.neighborOffset = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    // grid cell count/offset buffer
    bufferSize = NeighborSearch.GRID_COUNT_ALIGNMENT * Uint32Array.BYTES_PER_ELEMENT;
    this.cellParticleCount = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.cellOffset = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE });

    // particle index buffer
    bufferSize = SPH.MAX_PARTICAL_NUM * Uint32Array.BYTES_PER_ELEMENT;
    this.particleSortIndex = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE });
    this.particleSortIndexCopy = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE });

    // grid info buffer
    bufferSize = 2 * 4 * Uint32Array.BYTES_PER_ELEMENT;
    this.gridInfo = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const gridInfoArray = new Uint32Array(8);
    gridInfoArray.set([
      NeighborSearch.GRID_DIM, NeighborSearch.GRID_DIM, NeighborSearch.GRID_DIM, 0,
      NeighborSearch.GRID_DIM * NeighborSearch.GRID_DIM, NeighborSearch.GRID_DIM, 1, 0
    ]);
    device.queue.writeBuffer( this.gridInfo, 0, gridInfoArray, 0 );

    if (NeighborSearch.debug) {
      this.debugBuffer1 = device.createBuffer({
        size: 4 * this.simulator.particleCount * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      this.debugBuffer2 = device.createBuffer({
        size: SPH.MAX_NEIGHBOR_COUNT * this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      this.debugBuffer3 = device.createBuffer({
        size: this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
    }

  }

  private createBindGroup() {

    // particle insert
    this.particleInsertBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ]
    });

    this.particleInsertBindGroup = device.createBindGroup({
      layout: this.particleInsertBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePosition } },
        { binding: 1, resource: { buffer: this.cellParticleCount } },
        { binding: 2, resource: { buffer: this.particleSortIndex } },
        { binding: 3, resource: { buffer: this.gridInfo } },
      ]
    });

    // counting sort + neighbor count
    this.countingBindGroupLayout = device.createBindGroupLayout({
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

    this.countingBindGroup = device.createBindGroup({
      layout: this.countingBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePosition } },
        { binding: 1, resource: { buffer: this.neighborCount } },
        { binding: 2, resource: { buffer: this.cellParticleCount } },
        { binding: 3, resource: { buffer: this.cellOffset } },
        { binding: 4, resource: { buffer: this.particleSortIndex } },
        { binding: 5, resource: { buffer: this.particleSortIndexCopy } },
        { binding: 6, resource: { buffer: this.gridInfo } },
      ]
    });

    // neighbor search
    this.neighborListBindGroupLayout = device.createBindGroupLayout({
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

    this.neighborListBindGroup = device.createBindGroup({
      layout: this.neighborListBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePosition } },
        { binding: 1, resource: { buffer: this.neighborOffset } },
        { binding: 2, resource: { buffer: this.neighborList } },
        { binding: 3, resource: { buffer: this.cellParticleCount } },
        { binding: 4, resource: { buffer: this.cellOffset } },
        { binding: 5, resource: { buffer: this.particleSortIndexCopy } },
        { binding: 6, resource: { buffer: this.gridInfo } },
      ]
    })

  }

  private async createPipeline( groupLayout: GPUBindGroupLayout ) {

    this.particleInsertPipeline = await device.createComputePipelineAsync({
      label: 'Particle Insert Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [groupLayout, this.particleInsertBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: ParticleInsertShader }),
        entryPoint: 'main'
      }
    });

    this.countingSortPipeline = await device.createComputePipelineAsync({
      label: 'Counting Sort Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [groupLayout, this.countingBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: CountingSortShader }),
        entryPoint: 'main'
      }
    });

    this.neighborCountPipeline = await device.createComputePipelineAsync({
      label: 'Neighbor Count Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [groupLayout, this.countingBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: NeighborCountShader }),
        entryPoint: 'main',
        constants: {
          SearchRadiusSqr: NeighborSearch.SEARCH_RADIUS * NeighborSearch.SEARCH_RADIUS
        }
      }
    });

    this.neighborListPipeline = await device.createComputePipelineAsync({
      label: 'Neighbor List Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [groupLayout, this.neighborListBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: NeighborListShader }),
        entryPoint: 'main',
        constants: {
          SearchRadiusSqr: NeighborSearch.SEARCH_RADIUS * NeighborSearch.SEARCH_RADIUS
        }
      }
    });

  }

  public async initResource( groupLayout: GPUBindGroupLayout, particlePosition: GPUBuffer ) {

    this.particlePosition = particlePosition;

    this.createStorageData();
    this.createBindGroup();
    await this.createPipeline(groupLayout);

    const particleCountAlignment = Math.ceil(SPH.MAX_PARTICAL_NUM / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT;
    this.cellScan = new ExclusiveScan(this.cellParticleCount, this.cellOffset, NeighborSearch.GRID_COUNT_ALIGNMENT);
    this.particleScan = new ExclusiveScan(this.neighborCount, this.neighborOffset, particleCountAlignment);
    await this.cellScan.initResource(groupLayout);
    await this.particleScan.initResource(groupLayout);

  }

  public clearBuffer(commandEncoder: GPUCommandEncoder) {

    commandEncoder.clearBuffer(this.cellParticleCount);

  }

  public execute(passEncoder: GPUComputePassEncoder) {

    const workgroupCount = Math.ceil(this.simulator.particleCount / 256);

    // particle insert
    passEncoder.setBindGroup(1, this.particleInsertBindGroup);
    passEncoder.setPipeline(this.particleInsertPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // CellOffset[] = prefix sum of CellParticleCount[]
    this.cellScan.execute(passEncoder);

    passEncoder.setBindGroup(1, this.countingBindGroup);

    // counting sort
    passEncoder.setPipeline(this.countingSortPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // neighbor count
    passEncoder.setPipeline(this.neighborCountPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // neighborOffset[] = prefix sum of neighborCount[]
    const particleCountAlignment = Math.ceil((this.simulator.particleCount + 1) / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT;
    this.particleScan.execute(passEncoder, particleCountAlignment);

    // neighbor list
    passEncoder.setBindGroup(1, this.neighborListBindGroup);
    passEncoder.setPipeline(this.neighborListPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

  }

  public async debug() {

    if (NeighborSearch.debug) {
      
      const ce = device.createCommandEncoder();
      ce.copyBufferToBuffer(
        this.particlePosition, 0,
        this.debugBuffer1, 0,
        4 * this.simulator.particleCount * Float32Array.BYTES_PER_ELEMENT
      );
      ce.copyBufferToBuffer(
        this.neighborList, 0,
        this.debugBuffer2, 0,
        SPH.MAX_NEIGHBOR_COUNT * this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT
      );
      ce.copyBufferToBuffer(
        this.neighborOffset, 0,
        this.debugBuffer3, 0,
        this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT
      );
      device.queue.submit([ ce.finish() ]);

      await this.debugBuffer1.mapAsync(GPUMapMode.READ);
      const buffer1 = this.debugBuffer1.getMappedRange(0, 4 * this.simulator.particleCount * Float32Array.BYTES_PER_ELEMENT);
      const postionArray = new Float32Array(buffer1);
      
      await this.debugBuffer2.mapAsync(GPUMapMode.READ);
      const buffer2 = this.debugBuffer2.getMappedRange(0, SPH.MAX_NEIGHBOR_COUNT * this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT);
      const neighborListArray = new Uint32Array(buffer2);

      await this.debugBuffer3.mapAsync(GPUMapMode.READ);
      const buffer3 = this.debugBuffer3.getMappedRange(0, this.simulator.particleCount * Uint32Array.BYTES_PER_ELEMENT);
      const neighborOffsetArray = new Uint32Array(buffer3);

      const index = new Array(5).fill(0).map(_ => Math.floor(Math.random() * this.simulator.particleCount));
      const position = new Array(5).fill(0).map((_, i) => new THREE.Vector3().set( 
        postionArray[4 * index[i]], postionArray[4 * index[i] + 1], postionArray[4 * index[i] + 2] 
      ));
      
      let neighborPos = new THREE.Vector3();
      let deltaPos = new THREE.Vector3();
      const neighborList = new Array(5).fill(0).map(_ => new Array());
      for (let i = 0; i < this.simulator.particleCount; i++) {
        neighborPos.set( postionArray[4 * i], postionArray[4 * i + 1], postionArray[4 * i + 2] );
        position.forEach((pos, j) => {
          deltaPos.copy(neighborPos).sub(pos);
          if (deltaPos.length() <= NeighborSearch.SEARCH_RADIUS) {
            neighborList[j].push(i);
          }
        });
      }

      let right = true;
      neighborList.forEach((list, i) => {
        const start = neighborOffsetArray[index[i]];
        const end = neighborOffsetArray[index[i] + 1];
        
        let neiborListTest = neighborListArray.slice(
          start, end
        ).sort((a, b) => a - b);

        const neighborCountTrue = list.length;
        let neiborListTrue = list.sort((a, b) => a - b);

        if (neighborCountTrue != (end - start)) right = false;
        else neiborListTrue.forEach((val, j) => {
          if (val != neiborListTest[j]) right = false;
        });
      });
      console.log(right);

    }
    
    await this.cellScan.debug()

  }

}

export { NeighborSearch };