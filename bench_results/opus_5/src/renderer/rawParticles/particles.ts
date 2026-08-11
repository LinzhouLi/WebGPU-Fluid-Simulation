import * as THREE from 'three';
import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { device, canvasFormat } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import type { SPH } from '../../simulator/SPH';
import { vertexShader, fragmentShader } from './shader';

// Debug view of the raw particle data: one sphere imposter per particle, drawn
// as a 4 vertex triangle strip that reads its center from the simulator's
// position buffer. No vertex buffers and no index buffer are needed.

class RawParticles {

  private static ResourceFormats = {
    material: {
      type: 'buffer' as ResourceType,
      label: 'Material Structure',
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      layout: {
        type: 'uniform' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },
  };

  protected renderPipeline: GPURenderPipeline;
  protected bindGroup: GPUBindGroup;
  protected simulator: SPH;

  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  constructor(simulator: SPH) {

    this.simulator = simulator;

  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(RawParticles.ResourceFormats);
  }

  public async initResource(globalBindGroupLayout: GPUBindGroupLayout) {

    await this.initGroupResource();
    await this.initPipeline(globalBindGroupLayout);

  }

  private async initGroupResource() {

    // struct Material { color: vec4<f32>, radius: f32 }
    const material = new Float32Array(8);
    material.set([ ...new THREE.Color(0x049ef4).toArray(), 1 ]);
    material[4] = this.simulator.particleRadius;

    this.resourceCPUData = { material: { value: material } };
    this.resource = await resourceFactory.createResource(['material'], this.resourceCPUData);

  }

  protected async initPipeline(globalBindGroupLayout: GPUBindGroupLayout) {

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Bind Group Layout',
      entries: [{ // instance positions
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' }
      }, { // material
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }]
    });

    this.bindGroup = device.createBindGroup({
      label: 'Particle Rendering Bind Group',
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.simulator.position }
      }, {
        binding: 1,
        resource: { buffer: this.resource.material as GPUBuffer }
      }]
    });

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Particle Render Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [globalBindGroupLayout, bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: vertexShader }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: fragmentShader }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }],
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater', // reverse Z
        format: 'depth32float'
      }
    });

  }

  public render(renderPassEncoder: GPURenderPassEncoder) {

    renderPassEncoder.setPipeline(this.renderPipeline);
    renderPassEncoder.setBindGroup(1, this.bindGroup);
    renderPassEncoder.draw(4, this.simulator.particleCount);

  }

}

export { RawParticles };
