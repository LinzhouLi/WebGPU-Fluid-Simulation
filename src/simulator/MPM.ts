import { device } from '../controller';
import type { ResourceType } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';
import { BaseSimulator } from "./baseSimulator";
import { resourceFactory, bindGroupFactory } from '../common/base';
import { P2GComputeShader, GridComputeShader, G2PComputeShader } from './MPMShader';

class MPM extends BaseSimulator {

  private static _ResourceFormats = {
    particleVelocity: {
      type: 'buffer' as ResourceType,
      label: 'Particle Velocity',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    particleVelocityGradient: {
      type: 'buffer' as ResourceType,
      label: 'Particle Velocity Gradient',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    particlePlasticDeformation: {
      type: 'buffer' as ResourceType,
      label: 'Particle Plastic Deformation',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    gridVelocity: { // gridMomentum
      type: 'buffer' as ResourceType,
      label: 'Grid Velocity',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // commandEncoder.clearBuffer() needs COPY_DST ?
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    gridMass: {
      type: 'buffer' as ResourceType,
      label: 'Grid Mass',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    gravity: {
      type: 'buffer' as ResourceType,
      label: 'Gravity',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    debug: {
      type: 'buffer' as ResourceType,
      label: 'Debug',
      usage:  GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    },
  };

  private timeStep: number;
  private gridCount: number;
  private gridLength: number;
  private bound: number;

  private material: {
    particleDensity: number;
    particleVolume: number;
    particleMass: number;
    E: number;
  };

  private gravityArray: Float32Array;

  private resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>;

  private P2GComputePipeline: GPUComputePipeline;
  private GridComputePipeline: GPUComputePipeline;
  private G2PComputePipeline: GPUComputePipeline;

  private bindGroup: { layout: GPUBindGroupLayout, group: GPUBindGroup };

  constructor() {

    const n_grid = 32;
    super( 9000, 25 );

    this.timeStep = 1e-4;
    this.gridCount = n_grid;
    this.gridLength = 1 / n_grid;
    this.bound = 3.0;

    const rho = 1;
    const vol = Math.pow(this.gridLength * 0.5, 2);
    this.material = {
      particleDensity: rho,
      particleVolume: vol,
      particleMass: rho * vol,
      E: 400
    };

    this.gravityArray = new Float32Array(4);
    
  }

  public static _RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(MPM._ResourceFormats);
  }

  public async initResource() {

    this.resource = await resourceFactory.createResource(
      [ 
        'particleVelocity', 'particleVelocityGradient', 'particlePlasticDeformation',
        'gridVelocity', 'gridMass', 'gravity'
      ],
      {
        particleVelocity: { size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT },
        particleVelocityGradient: { size: 4 * 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT },
        particlePlasticDeformation: { size: this.particleCount * Float32Array.BYTES_PER_ELEMENT },
        gridVelocity: { size: 4 * this.gridCount * this.gridCount * this.gridCount * Float32Array.BYTES_PER_ELEMENT },
        gridMass: { size: this.gridCount * this.gridCount * this.gridCount * Float32Array.BYTES_PER_ELEMENT },
        gravity: { size: 4 * Float32Array.BYTES_PER_ELEMENT },
      }
    );
    this.resource.particlePosition = this.particlePositionBuffer;
    
    // set initial particle position
    let particlePositionArray = new Float32Array(4 * this.particleCount);
    let tempArray = new Array(3).fill(0);
    for (let particleIndex = 0; particleIndex < this.particleCount; particleIndex++) {
      particlePositionArray.set(
        tempArray.map(_ => Math.random() * 0.4 + 0.15),
        particleIndex * 4
      );
    }
    device.queue.writeBuffer(
      this.particlePositionBuffer, 0,
      particlePositionArray, 0
    );

    // set initial particle plastic deformation
    let particlePlasticDeformationArray = new Float32Array(this.particleCount).fill(1.0);
    device.queue.writeBuffer(
      this.resource.particlePlasticDeformation as GPUBuffer, 0,
      particlePlasticDeformationArray, 0
    );

    // set default gravity
    this.gravityArray.set([0, -9.8, 0, 0]);
    device.queue.writeBuffer(
      this.resource.gravity as GPUBuffer, 0,
      this.gravityArray, 0
    );

  }

  public enableInteraction() {

    document.addEventListener('keydown', event => {
      if (event.key.toUpperCase() === 'W') {
        this.gravityArray.set([-9.8, 0, 0, 0]);
      }
      else if (event.key.toUpperCase() === 'A') {
        this.gravityArray.set([0, 0, 9.8, 0]);
      }
      else if (event.key.toUpperCase() === 'S') {
        this.gravityArray.set([9.8, 0, 0, 0]);
      }
      else if (event.key.toUpperCase() === 'D') {
        this.gravityArray.set([0, 0, -9.8, 0]);
      }
      else if (event.key.toUpperCase() === 'Q') {
        this.gravityArray.set([0, 9.8, 0, 0]);
      }
      else if (event.key.toUpperCase() === 'E') {
        this.gravityArray.set([0, -9.8, 0, 0]);
      }
    });

  }

  private async initP2GComputePipeline() {

    this.P2GComputePipeline = await device.createComputePipelineAsync({
      label: 'MPM P2G Compute Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroup.layout] }),
      compute: {
        module: device.createShaderModule({ code: P2GComputeShader(this.particleCount, this.gridCount) }),
        constants: {
          dx: this.gridLength,
          dt: this.timeStep,
          p_vol: this.material.particleVolume,
          p_mass: this.material.particleMass,
          E: this.material.E
        },
        entryPoint: 'main'
      }
    });

  }

  private async initGridComputePipeline() {

    this.GridComputePipeline = await device.createComputePipelineAsync({
      label: 'MPM Grid Compute Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroup.layout] }),
      compute: {
        module: device.createShaderModule({ code: GridComputeShader(this.particleCount, this.gridCount) }),
        constants: {
          dt: this.timeStep,
          bound: this.bound,
        },
        entryPoint: 'main'
      }
    });

  }

  private async initG2PComputePipeline() {

    this.G2PComputePipeline = await device.createComputePipelineAsync({
      label: 'MPM G2P Compute Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroup.layout] }),
      compute: {
        module: device.createShaderModule({ code: G2PComputeShader(this.particleCount, this.gridCount) }),
        constants: {
          dx: this.gridLength,
          dt: this.timeStep
        },
        entryPoint: 'main'
      }
    });

  }

  public async initComputePipeline() {

    const attributes = [
      'particlePosition', 'particleVelocity', 'particleVelocityGradient', 'particlePlasticDeformation',
      'gridVelocity', 'gridMass', 'gravity'
    ];
    this.bindGroup = bindGroupFactory.create(attributes, this.resource);

    this.initP2GComputePipeline();
    this.initGridComputePipeline();
    this.initG2PComputePipeline();

  }

  public run(commandEncoder: GPUCommandEncoder) {

    // clear grid
    commandEncoder.clearBuffer(this.resource.gridVelocity as GPUBuffer);
    commandEncoder.clearBuffer(this.resource.gridMass as GPUBuffer);

    const passEncoder = commandEncoder.beginComputePass();

    // P2G pass
    passEncoder.setPipeline(this.P2GComputePipeline);
    passEncoder.setBindGroup(0, this.bindGroup.group);
    passEncoder.dispatchWorkgroups(Math.ceil(this.particleCount / 16));

    // grid pass
    passEncoder.setPipeline(this.GridComputePipeline);
    passEncoder.setBindGroup(0, this.bindGroup.group);
    passEncoder.dispatchWorkgroups(
      Math.ceil(this.gridCount / 4),
      Math.ceil(this.gridCount / 4),
      Math.ceil(this.gridCount / 4)
    );

    // G2P pass
    passEncoder.setPipeline(this.G2PComputePipeline);
    passEncoder.setBindGroup(0, this.bindGroup.group);
    passEncoder.dispatchWorkgroups(Math.ceil(this.particleCount / 16));

    passEncoder.end();

    // commandEncoder.copyBufferToBuffer( // for debug
    //   this.resource.gridVelocity as GPUBuffer, 0,
    //   this.resource.debug as GPUBuffer, 0,
    //   4 * this.gridCount * this.gridCount * this.gridCount * Float32Array.BYTES_PER_ELEMENT
    // );

  }

  public update() {

    device.queue.writeBuffer(
      this.resource.gravity as GPUBuffer, 0,
      this.gravityArray, 0
    );

  }

  public async debug() {

    await (this.resource.debug as GPUBuffer).mapAsync(GPUMapMode.READ);
    const buffer = (this.resource.debug as GPUBuffer).getMappedRange();
    console.log(buffer);
    const array = new Float32Array(buffer);
    console.log(array);
    for(let i = 0; i < 32; i++) {
      if (i == 15) console.log(array.slice(i * 4 * 32 * 32, (i+1) * 4 * 32 * 32))
      // console.log(array.slice(i * 32 * 32, (i+1) * 32 * 32))
      // console.log(array.slice(i * 4 * 256, (i+1) * 4 * 256));
    }
    // for (let i in array) {
    //   if (array[i] > 0) console.log(array[i]);
    // }
    // console.log(array.slice(4 * (14 * 32 ** 2 + 14 * 32 + 14), 4 * (14 * 32 ** 2 + 14 * 32 + 17)))
    
    (this.resource.debug as GPUBuffer).unmap();

  }

};

export { MPM };