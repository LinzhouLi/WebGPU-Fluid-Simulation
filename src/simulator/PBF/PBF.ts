import * as THREE from 'three';
import { device } from '../../controller';
import { PBFConfig } from './PBFConfig';
import { NeighborSearch } from '../neighbor/neighborSearch';
import { 
  ForceApplyShader,       LambdaCalculationShader,
  ConstrainSolveShader,   ConstrainApplyShader,
  AttributeUpdateShader,  XSPHShader
} from './PBFShader';


function kernalPoly6(r_len: number) {
  if (r_len <= PBF.KERNEL_RADIUS) {
    const coef = 315.0 / (64.0 * Math.PI * Math.pow(PBF.KERNEL_RADIUS, 9));
    const x = PBF.KERNEL_RADIUS * PBF.KERNEL_RADIUS - r_len * r_len;
    return coef * Math.pow(x, 3);
  }
  else {
    return 0;
  }
}

class PBF extends PBFConfig {

  private neighborList: GPUBuffer;
  private positionPredict: GPUBuffer;
  // private positionPredictCopy: GPUBuffer;
  private deltaPosition: GPUBuffer;
  private lambda: GPUBuffer;
  private velocity: GPUBuffer;
  private velocityCopy: GPUBuffer;
  private gravity: GPUBuffer;

  private advectionBindGroupLayout: GPUBindGroupLayout;
  private constrainBindGroupLayout: GPUBindGroupLayout;
  private viscosityBindGroupLayout: GPUBindGroupLayout;
  private advectionBindGroup: GPUBindGroup;
  private constrainBindGroup: GPUBindGroup;
  private viscosityBindGroup: GPUBindGroup;

  private forceApplyPipeline: GPUComputePipeline;
  private lambdaCalculationPipeline: GPUComputePipeline;
  private constrainSolvePipeline: GPUComputePipeline;
  private constrainApplyPipeline: GPUComputePipeline;
  private attributeUpdatePipeline: GPUComputePipeline;
  private XSPHPipeline: GPUComputePipeline;

  private neighborSearch: NeighborSearch;

  private gravityArray: Float32Array;

  private tempBuffer: GPUBuffer;
  private debugBuffer1: GPUBuffer;
  private debugBuffer2: GPUBuffer;

  constructor() {

    super(25 * 25 * 25);

  }

  private createStorageData() {

    const particlePerDim = 25;
    const range = 0.5;
    this.restDensity = this.particleCount / (range * range * range);

    // set initial particle position
    let positionArray = new Float32Array(4 * this.particleCount);
    let position = new THREE.Vector3();
    for (let i = 0; i < particlePerDim; i++) {
      for (let j = 0; j < particlePerDim; j++) {
        for (let k = 0; k < particlePerDim; k++) {
          position.set(i, j, k).multiplyScalar(range / particlePerDim).addScalar(0.15);
          positionArray.set(
            position.toArray(),
            (i * particlePerDim * particlePerDim + j * particlePerDim + k) * 4
          );
        }
      }
    }
    device.queue.writeBuffer(
      this.particlePositionBuffer, 0,
      positionArray, 0,
      4 * this.particleCount
    );

    let density = 0;
    let k = (10 * particlePerDim * particlePerDim + 10 * particlePerDim + 10);
    let pos = [positionArray[4*k], positionArray[4*k+1], positionArray[4*k+2]];
    for (let i = 0; i < this.particleCount; i++) {
      let npos = [
        positionArray[4*i] - pos[0], 
        positionArray[4*i+1] - pos[1], 
        positionArray[4*i+2] - pos[2]
      ];
      let len = Math.sqrt(npos[0]*npos[0] + npos[1]*npos[1] + npos[2]*npos[2]);
      density += kernalPoly6(len);
    }
    console.log(density, this.restDensity);

    // create GPU Buffers
    // neighbor list buffer
    this.neighborList = device.createBuffer({
      size: (PBF.MAX_NEIGHBOR_COUNT + 1) * this.particleCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // vec3/vec4 particle attribute buffer
    const attributeBufferDesp = {
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    } as GPUBufferDescriptor;
    this.positionPredict = device.createBuffer(attributeBufferDesp);
    // this.positionPredictCopy = device.createBuffer(attributeBufferDesp);
    this.deltaPosition = device.createBuffer(attributeBufferDesp);
    this.velocity = device.createBuffer(attributeBufferDesp);
    this.velocityCopy = device.createBuffer(attributeBufferDesp);

    // f32 particle attribute buffer
    this.lambda = device.createBuffer({
      size: this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

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

    this.tempBuffer = device.createBuffer({
      size: this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    this.debugBuffer1 = device.createBuffer({
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    this.debugBuffer2 = device.createBuffer({
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

  }

  private createBindGroup() {

    // advection
    this.advectionBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ]
    });

    this.advectionBindGroup = device.createBindGroup({
      layout: this.advectionBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePositionBuffer } },
        { binding: 1, resource: { buffer: this.positionPredict } },
        { binding: 2, resource: { buffer: this.velocity } },
        { binding: 3, resource: { buffer: this.gravity } },
      ]
    });

    // constrain
    this.constrainBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // !!!
      ]
    });

    this.constrainBindGroup = device.createBindGroup({
      layout: this.constrainBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.positionPredict } },
        { binding: 1, resource: { buffer: this.deltaPosition } },
        { binding: 2, resource: { buffer: this.lambda } },
        { binding: 3, resource: { buffer: this.neighborList } },
        { binding: 4, resource: { buffer: this.tempBuffer } },
      ]
    });

    // viscosity
    this.viscosityBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });

    this.viscosityBindGroup = device.createBindGroup({
      layout: this.viscosityBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particlePositionBuffer } },
        { binding: 1, resource: { buffer: this.positionPredict } },
        { binding: 2, resource: { buffer: this.velocity } },
        { binding: 3, resource: { buffer: this.velocityCopy } },
        { binding: 4, resource: { buffer: this.neighborList } },
      ]
    });

  }

  public async initResource() {

    this.createStorageData();
    this.createBindGroup();

    // neighbor search
    this.neighborSearch = new NeighborSearch(
      this.particleCount,
      PBF.KERNEL_RADIUS
    );
    await this.neighborSearch.initResource(
      this.positionPredict,  // this.positionPredictCopy,
      this.neighborList
    );

  }

  private getScorrCoefficient() {

    let poly6 = kernalPoly6(this.scorrCoefDq * PBF.KERNEL_RADIUS);
    return -this.scorrCoefK / Math.pow(poly6, this.scorrCoefN);

  }

  public enableInteraction() { }

  public async initComputePipeline() {

    // advection
    const advectionPipelineLayout = device.createPipelineLayout({ 
      bindGroupLayouts: [this.advectionBindGroupLayout] 
    });
    this.forceApplyPipeline = await device.createComputePipelineAsync({
      label: 'Force Apply Pipeline (PBF)',
      layout: advectionPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: ForceApplyShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          DeltaT: this.timeStep
        }
      }
    });

    // constrain
    const constrainPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.constrainBindGroupLayout]
    });
    this.lambdaCalculationPipeline = await device.createComputePipelineAsync({
      label: 'Lambda Calculation Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: LambdaCalculationShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          InvRestDensity: 1 / this.restDensity,
          InvRestDensity2: 1 / (this.restDensity * this.restDensity),
          LambdaEPS: this.lambdaEPS
        }
      }
    });



    this.constrainSolvePipeline = await device.createComputePipelineAsync({
      label: 'Constrain Solve Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: ConstrainSolveShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          InvRestDensity: 1 / this.restDensity,
          ScorrCoef: this.getScorrCoefficient()
        }
      }
    });

    this.constrainApplyPipeline = await device.createComputePipelineAsync({
      label: 'Constrain Apply Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: ConstrainApplyShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount
        }
      }
    });

    // viscosity
    const viscosityPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.viscosityBindGroupLayout]
    });
    this.attributeUpdatePipeline = await device.createComputePipelineAsync({
      label: 'Attribute Update Pipeline (PBF)',
      layout: viscosityPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: AttributeUpdateShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          InvDeltaT: 1 / this.timeStep
        }
      }
    });

    this.XSPHPipeline = await device.createComputePipelineAsync({
      label: 'XSPH Pipeline (PBF)',
      layout: viscosityPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: XSPHShader }),
        entryPoint: 'main',
        constants: {
          ParticleCount: this.particleCount,
          InvRestDensity: 1 / this.restDensity,
          XSPHCoef: this.XSPHCoef
        }
      }
    });

  }

  public run(commandEncoder: GPUCommandEncoder) {

    this.neighborSearch.clearBuffer(commandEncoder);
    
    const passEncoder = commandEncoder.beginComputePass();
    const workgroupCount = Math.ceil(this.particleCount / 64);

    passEncoder.setBindGroup(0, this.advectionBindGroup);
    passEncoder.setPipeline(this.forceApplyPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    this.neighborSearch.execute(passEncoder);

    passEncoder.setBindGroup(0, this.constrainBindGroup);
    for (let i = 0; i < this.constrainIterationCount; i++) {
      passEncoder.setPipeline(this.lambdaCalculationPipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);

      passEncoder.setPipeline(this.constrainSolvePipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);

      passEncoder.setPipeline(this.constrainApplyPipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);
    }

    passEncoder.setBindGroup(0, this.viscosityBindGroup);
    passEncoder.setPipeline(this.attributeUpdatePipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.setPipeline(this.XSPHPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.end();
  
  }

  public update() { }

  public async debug() {

    const ce = device.createCommandEncoder();
    ce.copyBufferToBuffer(
      this.lambda, 0,
      this.debugBuffer1, 0,
      this.particleCount * Float32Array.BYTES_PER_ELEMENT
    );
    ce.copyBufferToBuffer(
      this.tempBuffer, 0,
      this.debugBuffer2, 0,
      this.particleCount * Float32Array.BYTES_PER_ELEMENT
    );
    device.queue.submit([ ce.finish() ]);

    await this.debugBuffer1.mapAsync(GPUMapMode.READ);
    const buffer1 = this.debugBuffer1.getMappedRange(0, this.particleCount * Float32Array.BYTES_PER_ELEMENT);
    const array1 = new Float32Array(buffer1);

    await this.debugBuffer2.mapAsync(GPUMapMode.READ);
    const buffer2 = this.debugBuffer2.getMappedRange(0, this.particleCount * Float32Array.BYTES_PER_ELEMENT);
    const array2 = new Float32Array(buffer2);

    let min = array1[0]; let max = array1[0];
    let min_index = 0; let max_index = 0;
    array1.forEach((val, i) => {
      if (val >= max) { max = val; max_index = i; }
      if (val <= min) { min = val; min_index = i; }
    });
    console.log(min_index, min);
    console.log(max_index, max);

    min = array2[0]; max = array2[0];
    min_index = 0; max_index = 0;
    array2.forEach((val, i) => {
      if (val >= max) { max = val; max_index = i; }
      if (val <= min) { min = val; min_index = i; }
    });
    console.log(min_index, min);
    console.log(max_index, max);

    await this.neighborSearch.debug();

  }

}

export { PBF };