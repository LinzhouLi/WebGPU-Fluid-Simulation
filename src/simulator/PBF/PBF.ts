import * as THREE from 'three';
import { device } from '../../controller';
import { LagrangianSimulator } from "../LagrangianSimulator";
import { NeighborSearch } from '../neighbor/neighborSearch';

class PBF extends LagrangianSimulator {

  private maxNeighborCount = 47;
  private kernelRadius = 0.025;

  private neighborList: GPUBuffer;
  private positionCopy: GPUBuffer;
  private deltaPosition: GPUBuffer;
  private velocity: GPUBuffer;
  private velocityCopy: GPUBuffer;
  private gravity: GPUBuffer;
  // private debugBuffer: GPUBuffer;
  // private debugBuffer2: GPUBuffer;

  private bindGroup1: GPUBindGroup;
  private bindGroup2: GPUBindGroup;

  private neighborSearch: NeighborSearch;

  private gravityArray: Float32Array;

  constructor() {

    super(40 * 40 * 40, 1);

  }

  private createStorageData() {

    // set initial particle position
    let particleArray = new Float32Array(4 * this.particleCount);
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
    // neighbor list buffer
    this.neighborList = device.createBuffer({
      size: (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // vec3/vec4 particle attribute buffer
    const attributeBufferDesp = {
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    } as GPUBufferDescriptor;
    this.positionCopy = device.createBuffer(attributeBufferDesp);
    this.deltaPosition = device.createBuffer(attributeBufferDesp);
    this.velocity = device.createBuffer(attributeBufferDesp);
    this.velocityCopy = device.createBuffer(attributeBufferDesp);

    // gravity buffer
    this.gravity = device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.gravityArray = new Float32Array(4);
    this.gravityArray.set([0, -9.8, 0, 0]);
    device.queue.writeBuffer(
      this.gravity, 0,
      this.gravityArray, 0
    );

    // this.debugBuffer = device.createBuffer({
    //   size: (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    // });
    // this.debugBuffer2 = device.createBuffer({
    //   size: 4 * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    // });

  }

  public async initResource() {

    this.createStorageData();

    // bind group
    // const bindgroupLayout = device.createBindGroupLayout({
    //   entries: [
    //     { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    //     { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    //   ]
    // });

    // this.bindGroup1 = device.createBindGroup({
    //   layout: bindgroupLayout,
    //   entries: [
    //     { binding: 0, resource: { buffer: this.particlePositionBuffer } },
    //     { binding: 1, resource: { buffer: this.neighborList } },
    //     { binding: 2, resource: { buffer: this.neighborList } },
    //   ]
    // });

    // neighbor search
    this.neighborSearch = new NeighborSearch(
      this.particleCount,
      this.maxNeighborCount,
      this.kernelRadius
    );
    await this.neighborSearch.initResource(
      this.particlePositionBuffer,  this.positionCopy,
      this.velocity,                this.velocityCopy,
      this.neighborList
    );

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

    // const ce1 = device.createCommandEncoder();
    // ce1.copyBufferToBuffer(
    //   this.neighborList, 0,
    //   this.debugBuffer, 0,
    //   (this.maxNeighborCount + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce1.finish() ]);
    // await this.debugBuffer.mapAsync(GPUMapMode.READ);
    // const buffer1 = this.debugBuffer.getMappedRange();
    // const neighborListArray = new Uint32Array(buffer1);
    // console.log(neighborListArray);

    // const ce2 = device.createCommandEncoder();
    // ce2.copyBufferToBuffer(
    //   this.positionCopy, 0,
    //   this.debugBuffer2, 0,
    //   4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce2.finish() ]);
    // await this.debugBuffer2.mapAsync(GPUMapMode.READ);
    // const buffer2 = this.debugBuffer2.getMappedRange(0, 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
    // const positionArray = new Float32Array(buffer2);
    // console.log(positionArray);

    // let neighbors = [];
    // let k = (20 * 40 * 40 + 20 * 40 + 20);
    // let pos = [positionArray[4*k], positionArray[4*k+1], positionArray[4*k+2]];
    // for (let i = 0; i < this.particleCount; i++) {
    //   let npos = [
    //     positionArray[4*i] - pos[0], 
    //     positionArray[4*i+1] - pos[1], 
    //     positionArray[4*i+2] - pos[2]
    //   ];
    //   if (npos[0]*npos[0] + npos[1]*npos[1] + npos[2]*npos[2] < this.kernelRadius*this.kernelRadius) {
    //     neighbors.push(i)
    //   }
    // }
    // console.log(neighborListArray.slice(k*48, (k+1)* 48));
    // console.log(neighbors);

    await this.neighborSearch.debug();

  }

}

export { PBF };