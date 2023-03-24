import * as THREE from 'three';
import { device } from '../../controller';
import { PBFConfig } from '../PBF/PBFConfig';
import { ExclusiveScan } from './exclusiveScan';
import { ParticleInsertShader, CountingSortShader, NeighborListShader } from './neighborShader';

class NeighborSearch {

  public static MIN_SEARCH_RADIUS = 1.0 / Math.floor( Math.cbrt(ExclusiveScan.MAX_ARRAY_LENGTH) );

  private searchRadius: number;
  private particleCount: number;
  private gridDimension: number;
  private gridCellCount: number;
  private gridCellCountAlignment: number;

  private particlePosition: GPUBuffer;
  private neighborList: GPUBuffer;
  private cellParticleCount: GPUBuffer;
  private cellOffset: GPUBuffer;
  private particleSortIndex: GPUBuffer;
  private particleSortIndexCopy: GPUBuffer;
  private gridInfo: GPUBuffer;

  private bindGroup: GPUBindGroup;

  private particleInsertPipeline: GPUComputePipeline;
  private countingSortPipeline: GPUComputePipeline;
  private neighborListPipeline: GPUComputePipeline;

  private scan: ExclusiveScan;

  // private debugBuffer1: GPUBuffer;
  // private debugBuffer2: GPUBuffer;

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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    } as GPUBufferDescriptor;
    this.cellParticleCount = device.createBuffer(cellBufferDesp);
    this.cellOffset = device.createBuffer(cellBufferDesp);

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

    // this.debugBuffer1 = device.createBuffer({
    //   size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    // });
    // this.debugBuffer2 = device.createBuffer({
    //   size: (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    // });

  }

  public async initResource(
    particlePosition: GPUBuffer, // particlePositionSort: GPUBuffer,
    neighborList: GPUBuffer
  ) {

    this.particlePosition = particlePosition;
    this.neighborList = neighborList;
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
        // { binding: 1, resource: { buffer: particlePositionSort } },
        { binding: 1, resource: { buffer: neighborList } },
        { binding: 2, resource: { buffer: this.cellParticleCount } },
        { binding: 3, resource: { buffer: this.cellOffset } },
        { binding: 4, resource: { buffer: this.particleSortIndex } },
        { binding: 5, resource: { buffer: this.particleSortIndexCopy } },
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
    //   this.particlePosition, 0,
    //   this.debugBuffer1, 0,
    //   4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT
    // );
    // ce.copyBufferToBuffer(
    //   this.neighborList, 0,
    //   this.debugBuffer2, 0,
    //   (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce.finish() ]);

    // await this.debugBuffer1.mapAsync(GPUMapMode.READ);
    // const buffer1 = this.debugBuffer1.getMappedRange(0, 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
    // const postionArray = new Float32Array(buffer1);
    
    // await this.debugBuffer2.mapAsync(GPUMapMode.READ);
    // const buffer2 = this.debugBuffer2.getMappedRange(0, (PBFConfig.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
    // const neighborListArray = new Uint32Array(buffer2);

    // const index = new Array(5).fill(0).map(_ => Math.floor(Math.random() * this.particleCount));
    // const position = new Array(5).fill(0).map((_, i) => new THREE.Vector3().set( 
    //   postionArray[4 * index[i]], postionArray[4 * index[i] + 1], postionArray[4 * index[i] + 2] 
    // ));
    
    // let neighborPos = new THREE.Vector3();
    // let deltaPos = new THREE.Vector3();
    // const neighborList = new Array(5).fill(0).map(_ => new Array());
    // for (let i = 0; i < this.particleCount; i++) {
    //   neighborPos.set( postionArray[4 * i], postionArray[4 * i + 1], postionArray[4 * i + 2] );
    //   position.forEach((pos, j) => {
    //     deltaPos.copy(neighborPos).sub(pos);
    //     if (deltaPos.length() <= this.searchRadius) {
    //       neighborList[j].push(i);
    //     }
    //   });
    // }
    

    // let right = true;
    // const alignment = PBFConfig.MAX_NEIGHBOR_COUNT + 1;
    // neighborList.forEach((list, i) => {
    //   const start = index[i] * alignment;
    //   const neighborCountTest = neighborListArray[start];
    //   let neiborListTest = neighborListArray.slice(
    //     start + 1, start + 1 + neighborCountTest
    //   ).sort((a, b) => a - b);

    //   const neighborCountTrue = list.length;
    //   let neiborListTrue = list.sort((a, b) => a - b);

    //   if (neighborCountTrue != neighborCountTest) right = false;
    //   else neiborListTrue.forEach((val, j) => {
    //     if (val != neiborListTest[j]) right = false;
    //   });
    // });
    // console.log(right);

    await this.scan.debug()

  }

}

export { NeighborSearch };