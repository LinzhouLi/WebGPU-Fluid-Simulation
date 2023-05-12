import { device } from '../../controller';
import type { ResourceType } from '../../common/resourceFactory';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from "../LagrangianSimulator";
import { resourceFactory, bindGroupFactory } from '../../common/base';
import { P2GComputeShader, GridComputeShader, G2PComputeShader } from './MPMShader';

class MPM extends LagrangianSimulator {

  private static _ResourceFormats = {
    particle: {
      type: 'buffer' as ResourceType,
      label: 'Particle Velocity and Plastic Deformation',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      layout: { type: 'storage' as GPUBufferBindingType } as GPUBufferBindingLayout
    },
    grid: {
      type: 'buffer' as ResourceType,
      label: 'Grid Velocity and Mass',
      visibility: GPUShaderStage.COMPUTE,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, // commandEncoder.clearBuffer() needs COPY_DST ?
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

  private resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>;

  private P2GComputePipeline: GPUComputePipeline;
  private GridComputePipeline: GPUComputePipeline;
  private G2PComputePipeline: GPUComputePipeline;

  private P2GBindGroup: { layout: GPUBindGroupLayout, group: GPUBindGroup };
  private GridBindGroup: { layout: GPUBindGroupLayout, group: GPUBindGroup };

  constructor() {

    const n_grid = 32;
    super( 9000, 15 );

    this.timeStep = 0.0008;
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
    
  }

  public static _RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(MPM._ResourceFormats);
  }

  public async initResource() {

    this.resource = await resourceFactory.createResource(
      [ 'particle', 'grid' ],
      {
        particle: { size: 4 * 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT },
        grid: { size: 4 * this.gridCount * this.gridCount * this.gridCount * Float32Array.BYTES_PER_ELEMENT },
      }
    );
    this.resource.particlePosition = this.particlePositionBuffer;
    
    // set initial particle position
    let particleArray = new Float32Array(16 * this.particleCount);
    let tempArray = new Array(3).fill(0);
    for (let particleIndex = 0; particleIndex < this.particleCount; particleIndex++) {
      particleArray.set(
        tempArray.map(_ => Math.random() * 0.4 + 0.15),
        particleIndex * 4
      );
    }
    device.queue.writeBuffer(
      this.particlePositionBuffer, 0,
      particleArray, 0,
      4 * this.particleCount
    );

    // set initial particle plastic deformation = 1
    for (let particleIndex = 0; particleIndex < this.particleCount; particleIndex++) {
      particleArray.set( [1.0], particleIndex * 16 + 3 );
    }
    device.queue.writeBuffer(
      this.resource.particle as GPUBuffer, 0,
      particleArray, 0
    );

  }

  private async initP2GComputePipeline() {

    this.P2GComputePipeline = await device.createComputePipelineAsync({
      label: 'MPM P2G Compute Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.P2GBindGroup.layout] }),
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
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.GridBindGroup.layout] }),
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
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.P2GBindGroup.layout] }),
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

    this.P2GBindGroup = bindGroupFactory.create(
      ['particlePosition', 'particle', 'grid'],
      this.resource,
    );
    this.GridBindGroup = bindGroupFactory.create(
      ['grid', 'gravity'], 
      this.resource,
    );

    await this.initP2GComputePipeline();
    await this.initGridComputePipeline();
    await this.initG2PComputePipeline();

  }

  public run(commandEncoder: GPUCommandEncoder) {

    if (this.pause) return;

    // clear grid
    commandEncoder.clearBuffer(this.resource.grid as GPUBuffer);

    const passEncoder = commandEncoder.beginComputePass();

    // P2G pass
    passEncoder.setPipeline(this.P2GComputePipeline);
    passEncoder.setBindGroup(0, this.P2GBindGroup.group);
    passEncoder.dispatchWorkgroups(Math.ceil(this.particleCount / 4));

    // grid pass
    passEncoder.setPipeline(this.GridComputePipeline);
    passEncoder.setBindGroup(0, this.GridBindGroup.group);
    passEncoder.dispatchWorkgroups(
      Math.ceil(this.gridCount / 2),
      Math.ceil(this.gridCount / 2),
      Math.ceil(this.gridCount / 2)
    );

    // G2P pass
    passEncoder.setPipeline(this.G2PComputePipeline);
    passEncoder.setBindGroup(0, this.P2GBindGroup.group);
    passEncoder.dispatchWorkgroups(Math.ceil(this.particleCount / 4));

    passEncoder.end();

    // commandEncoder.copyBufferToBuffer( // for debug
    //   this.resource.gridVelocity as GPUBuffer, 0,
    //   this.resource.debug as GPUBuffer, 0,
    //   4 * this.gridCount * this.gridCount * this.gridCount * Float32Array.BYTES_PER_ELEMENT
    // );

  }

  public update() { }

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