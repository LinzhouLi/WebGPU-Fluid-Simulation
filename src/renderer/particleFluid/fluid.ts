import * as THREE from 'three';
import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { canvasFormat, device } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { vertexShader, fragmentShader } from './shader';

class ParticleFluid {

  private static ResourceFormats = {
    sphereMaterial: {
      type: 'buffer' as ResourceType,
      label: 'Sphere Material Structure', 
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      layout: { 
        type: 'uniform' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },
  };

  protected radius: number;
  protected spriteMesh: THREE.Mesh;
  protected simulator: LagrangianSimulator;

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected resourceAttributes: string[]; // resource name
  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  protected vertexBufferLayout: GPUVertexBufferLayout[];
  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;
  protected renderPipeline: GPURenderPipeline;

  constructor(simulator: LagrangianSimulator) {

    this.vertexShaderCode = vertexShader;
    this.fragmentShaderCode = fragmentShader;
    this.simulator = simulator;
    this.radius = 0.02;
    this.spriteMesh = new THREE.Mesh(
      new THREE.PlaneGeometry( 1, 1 ),
      new THREE.MeshLambertMaterial({ color: 0x049ef4 })
    );
    
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(ParticleFluid.ResourceFormats);
  }

  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    await this.initGroupResource();
    this.initBindGroup(globalResource);
    await this.initPipeline()

  }

  protected async initGroupResource() {

    const material = this.spriteMesh.material as THREE.MeshPhysicalMaterial;

    this.resourceAttributes = ['sphereMaterial', 'particlePosition'];
    this.resourceCPUData = {
      sphereMaterial: { 
        value: new Float32Array([
          this.radius, 0, 0, 0,
          ...material.color.toArray(), 0 // fix bug: bind group is too small!
        ])
      }
    };
    
    this.resource = await resourceFactory.createResource(['sphereMaterial'], this.resourceCPUData);
    this.resource.particlePosition = this.simulator.particlePositionBuffer;
    
  }

  protected initBindGroup(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {
    
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // material
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // instance positions
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' }
      }, { // light
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }]
    });

    this.bindGroup = device.createBindGroup({
      label: 'Particle Rendering Pipeline Bind Group',
      layout: this.bindGroupLayout,
      entries: [{ // camera
        binding: 0,
        resource: { buffer: globalResource.camera as GPUBuffer },
      }, { // material
        binding: 1,
        resource: { buffer: this.resource.sphereMaterial as GPUBuffer }
      }, { // instance positions
        binding: 2,
        resource: { buffer: this.simulator.particlePositionBuffer }
      }, { // light
        binding: 3,
        resource: { buffer: globalResource.directionalLight as GPUBuffer }
      }]
    })
  }

  protected async initPipeline() {

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        entryPoint: 'main',
        buffers: this.vertexBufferLayout
      },
      fragment: {
        module: device.createShaderModule({ code: this.fragmentShaderCode }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }],
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      }, 
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

  }

  protected async setRenderBundle(
    bundleEncoder: GPURenderBundleEncoder
  ) {
    
    bundleEncoder.setPipeline(this.renderPipeline);

    // set bind group
    bundleEncoder.setBindGroup(0, this.bindGroup);

    // draw
    bundleEncoder.draw(4, this.simulator.particleCount);
    
  }

}

export { ParticleFluid }