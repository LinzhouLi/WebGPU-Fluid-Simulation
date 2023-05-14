import * as THREE from 'three';
import { device } from '../../controller';
import { LagrangianSimulator } from '../LagrangianSimulator';
import { PBFConfig } from '../PBF/PBFConfig';
import { ExclusiveScan } from './exclusiveScan';
import { ParticleInsertShader, CountingSortShader, NeighborCountShader, NeighborListShader } from './neighborShader';

class NeighborSearch {

  public static MIN_SEARCH_RADIUS = 1.0 / Math.floor( Math.cbrt(ExclusiveScan.MAX_ARRAY_LENGTH) );

  private searchRadius: number;
  private particleCount: number;
  private gridDimension: number;
  private gridCellCount: number;
  private gridCellCountAlignment: number;

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

  constructor( simulator: LagrangianSimulator ) {

    this.searchRadius = LagrangianSimulator.KERNEL_RADIUS;
    this.particleCount = simulator.particleCount;

    this.gridDimension = Math.ceil(1.0 / this.searchRadius);
    this.gridCellCount = Math.pow(this.gridDimension, 3);
    this.gridCellCountAlignment = Math.ceil(this.gridCellCount / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT;

    if (this.searchRadius < NeighborSearch.MIN_SEARCH_RADIUS)
      throw new Error(`Search Radius should greater than ${NeighborSearch.MIN_SEARCH_RADIUS}!`);

  }

  private createStorageData() {

    // create GPU Buffers
    // neighbor list buffer
    this.neighborList = device.createBuffer({
      size: LagrangianSimulator.MAX_NEIGHBOR_COUNT * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    });
    let attributeBufferDesp = {
      size: 
        Math.ceil((this.particleCount + 1) / ExclusiveScan.ARRAY_ALIGNMENT) * 
        ExclusiveScan.ARRAY_ALIGNMENT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    } as GPUBufferDescriptor;
    this.neighborCount = device.createBuffer(attributeBufferDesp);
    this.neighborOffset = device.createBuffer(attributeBufferDesp);

    // grid cell buffer x2
    this.cellParticleCount = device.createBuffer({
      size: this.gridCellCountAlignment * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST // for clearBuffer()
    });
    this.cellOffset = device.createBuffer({
      size: this.gridCellCountAlignment * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    });

    // particle buffer x1
    const particleBufferDesp = {
      size: this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    } as GPUBufferDescriptor;
    this.particleSortIndex = device.createBuffer(particleBufferDesp);
    this.particleSortIndexCopy = device.createBuffer(particleBufferDesp);

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

    if (NeighborSearch.debug) {
      this.debugBuffer1 = device.createBuffer({
        size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      this.debugBuffer2 = device.createBuffer({
        size: (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
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

  private async createPipeline() {

    this.particleInsertPipeline = await device.createComputePipelineAsync({
      label: 'Particle Insert Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.particleInsertBindGroupLayout] }),
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
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.countingBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: CountingSortShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount
        }
      }
    });

    this.neighborCountPipeline = await device.createComputePipelineAsync({
      label: 'Neighbor Count Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.countingBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: NeighborCountShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          SearchRadiusSqr: this.searchRadius * this.searchRadius
        }
      }
    });

    this.neighborListPipeline = await device.createComputePipelineAsync({
      label: 'Neighbor List Pipeline (Neighbor Search)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.neighborListBindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: NeighborListShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          SearchRadiusSqr: this.searchRadius * this.searchRadius
        }
      }
    });

  }

  public async initResource( particlePosition: GPUBuffer ) {

    this.particlePosition = particlePosition;

    this.createStorageData();
    this.createBindGroup();
    await this.createPipeline();

    const particleCountAlignment = Math.ceil((this.particleCount + 1) / ExclusiveScan.ARRAY_ALIGNMENT) * ExclusiveScan.ARRAY_ALIGNMENT;
    this.cellScan = new ExclusiveScan(this.cellParticleCount, this.cellOffset, this.gridCellCountAlignment);
    this.particleScan = new ExclusiveScan(this.neighborCount, this.neighborOffset, particleCountAlignment);
    await this.cellScan.initResource();
    await this.particleScan.initResource();

  }

  public clearBuffer(commandEncoder: GPUCommandEncoder) {

    commandEncoder.clearBuffer(this.cellParticleCount);

  }

  public execute(passEncoder: GPUComputePassEncoder) {

    const workgroupCount = Math.ceil(this.particleCount / 256);

    // particle insert
    passEncoder.setBindGroup(0, this.particleInsertBindGroup);
    passEncoder.setPipeline(this.particleInsertPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // CellOffset[] = prefix sum of CellParticleCount[]
    this.cellScan.execute(passEncoder);

    passEncoder.setBindGroup(0, this.countingBindGroup);

    // counting sort
    passEncoder.setPipeline(this.countingSortPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // neighbor count
    passEncoder.setPipeline(this.neighborCountPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    // neighborOffset[] = prefix sum of neighborCount[]
    this.particleScan.execute(passEncoder);

    // neighbor list
    passEncoder.setBindGroup(0, this.neighborListBindGroup);
    passEncoder.setPipeline(this.neighborListPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

  }

  public async debug() {

    if (NeighborSearch.debug) {
      
      const ce = device.createCommandEncoder();
      ce.copyBufferToBuffer(
        this.particlePosition, 0,
        this.debugBuffer1, 0,
        4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT
      );
      ce.copyBufferToBuffer(
        this.neighborList, 0,
        this.debugBuffer2, 0,
        (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT
      );
      device.queue.submit([ ce.finish() ]);

      await this.debugBuffer1.mapAsync(GPUMapMode.READ);
      const buffer1 = this.debugBuffer1.getMappedRange(0, 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
      const postionArray = new Float32Array(buffer1);
      // console.log(postionArray)
      
      await this.debugBuffer2.mapAsync(GPUMapMode.READ);
      const buffer2 = this.debugBuffer2.getMappedRange(0, (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
      const neighborListArray = new Uint32Array(buffer2);
      console.log(neighborListArray)

      const index = new Array(5).fill(0).map(_ => Math.floor(Math.random() * this.particleCount));
      const position = new Array(5).fill(0).map((_, i) => new THREE.Vector3().set( 
        postionArray[4 * index[i]], postionArray[4 * index[i] + 1], postionArray[4 * index[i] + 2] 
      ));
      
      let neighborPos = new THREE.Vector3();
      let deltaPos = new THREE.Vector3();
      const neighborList = new Array(5).fill(0).map(_ => new Array());
      for (let i = 0; i < this.particleCount; i++) {
        neighborPos.set( postionArray[4 * i], postionArray[4 * i + 1], postionArray[4 * i + 2] );
        position.forEach((pos, j) => {
          deltaPos.copy(neighborPos).sub(pos);
          if (deltaPos.length() <= this.searchRadius) {
            neighborList[j].push(i);
          }
        });
      }
      

      let right = true;
      const alignment = PBFConfig.MAX_NEIGHBOR_COUNT + 1;
      neighborList.forEach((list, i) => {
        const start = index[i] * alignment;
        const neighborCountTest = neighborListArray[start];
        let neiborListTest = neighborListArray.slice(
          start + 1, start + 1 + neighborCountTest
        ).sort((a, b) => a - b);

        const neighborCountTrue = list.length;
        let neiborListTrue = list.sort((a, b) => a - b);

        if (neighborCountTrue != neighborCountTest) right = false;
        else neiborListTrue.forEach((val, j) => {
          if (val != neiborListTest[j]) right = false;
        });
        console.log(neiborListTest, neiborListTrue)
      });
      console.log(right);

    }
    
    await this.cellScan.debug()

  }

}

export { NeighborSearch };