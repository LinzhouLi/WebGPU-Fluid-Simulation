import * as THREE from 'three';
import { device } from '../../controller';
import { LagrangianSimulator } from "../LagrangianSimulator";
import { NeighborSearch } from '../neighbor/neighborSearch';

class PBF extends LagrangianSimulator {

  private maxNeighborCount = 47;
  private kernelRadius = 0.025;

  private neighborSearch: NeighborSearch;

  private bindGroup: GPUBindGroup;

  private neighborList: GPUBuffer;
  private debugBuffer: GPUBuffer;

  private particleArray: Float32Array;

  constructor() {

    super(40 * 40 * 40, 1);

  }

  private createStorageData() {

    // set initial particle position
    let particleArray = new Float32Array(4 * this.particleCount);
    this.particleArray = particleArray;
    let position = new THREE.Vector3();
    const range = 40;
    for (let i = 0; i < range; i++) {
      for (let j = 0; j < range; j++) {
        for (let k = 0; k < range; k++) {
          position.set(i, j, k).multiplyScalar(0.5 / range).addScalar(0.15);
          particleArray.set(
            position.toArray(),
            (i * range * range + j * range + k) * 4
          );
        }
      }
    }
    device.queue.writeBuffer(
      this.particlePositionBuffer, 0,
      particleArray, 0,
      4 * this.particleCount
    );

    // create GPU Buffers
    this.neighborList = device.createBuffer({
      size: (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    this.debugBuffer = device.createBuffer({
      size: (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

  }

  public async initResource() {

    this.createStorageData();

    // bind group
    const bindgroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    this.bindGroup = device.createBindGroup({
      layout: bindgroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePositionBuffer } },
        { binding: 1, resource: { buffer: this.neighborList } }
      ]
    });

    // neighbor search
    this.neighborSearch = new NeighborSearch(
      this.particlePositionBuffer,  this.neighborList,
      this.particleCount,           this.maxNeighborCount,
      this.kernelRadius
    );
    await this.neighborSearch.initResource();

  }

  public enableInteraction() { }

  public async initComputePipeline() { }

  public run(commandEncoder: GPUCommandEncoder) {

    const passEncoder = commandEncoder.beginComputePass();

    this.neighborSearch.execute(passEncoder);

    passEncoder.end();

  }

  public update() { }

  public async debug() {

    // const ce = device.createCommandEncoder();
    // ce.copyBufferToBuffer(
    //   this.neighborList, 0,
    //   this.debugBuffer, 0,
    //   (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce.finish() ]);
    // await this.debugBuffer.mapAsync(GPUMapMode.READ);
    // const buffer = this.debugBuffer.getMappedRange();
    // const array = new Uint32Array(buffer);
    // console.log(array);

    // let neighbors = [];
    // let k = (20 * 40 * 40 + 20 * 40 + 20);
    // let pos = [this.particleArray[4*k], this.particleArray[4*k+1], this.particleArray[4*k+2]];
    // for (let i = 0; i < this.particleCount; i++) {
    //   let npos = [
    //     this.particleArray[4*i] - pos[0], 
    //     this.particleArray[4*i+1] - pos[1], 
    //     this.particleArray[4*i+2] - pos[2]
    //   ];
    //   if (npos[0]*npos[0] + npos[1]*npos[1] + npos[2]*npos[2] < this.kernelRadius*this.kernelRadius) {
    //     neighbors.push(i)
    //   }
    // }
    // console.log(array.slice(k*48, (k+1)* 48));
    // console.log(neighbors);

    await this.neighborSearch.debug();

  }

}

export { PBF };