import * as THREE from 'three';
import type { TypedArray } from '../../common/base';
import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { device, canvasFormat } from '../../controller';
import { resourceFactory, vertexBufferFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { SPH } from '../../simulator/SPH';
import { vertexShader, fragmentShader } from './shader';

class RawParticles {

  private static ResourceFormats = {
    material: {
      type: 'buffer' as ResourceType,
      label: 'Material Structure', 
      visibility: GPUShaderStage.FRAGMENT,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      layout: { 
        type: 'uniform' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },
  };


  protected mesh: THREE.Mesh;
  protected renderPipeline: GPURenderPipeline;
  protected bindGroup: GPUBindGroup;
  protected simulator: SPH;

  protected vertexCount: number;
  protected vertexBufferAttributes: string[]; // resource name
  protected vertexBufferData: Record<string, TypedArray>; // resource in CPU
  protected vertexBuffers: Record<string, GPUBuffer>; // resource in GPU
  protected indexFormat: GPUIndexFormat;

  protected resourceAttributes: string[]; // resource name
  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  constructor(simulator: SPH) {

    this.simulator = simulator;
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.007, 8, 8), // radius, width segment, height segment
      new THREE.MeshLambertMaterial({ color: 0x049ef4 })
    );
    
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(RawParticles.ResourceFormats);
  }

  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.initVertexBuffer();
    await this.initGroupResource();
    await this.initPipeline(globalResource);

  }

  public initVertexBuffer() {

    this.vertexBufferAttributes = ['position', 'normal', 'uv'];
    this.vertexBufferData = {
      position: this.mesh.geometry.attributes.position.array as TypedArray,
      normal: this.mesh.geometry.attributes.normal.array as TypedArray,
      uv: this.mesh.geometry.attributes.uv.array as TypedArray,
    };

    if (!!this.mesh.geometry.index) {
      this.vertexBufferAttributes.push('index');
      this.vertexBufferData.index = this.mesh.geometry.index.array as TypedArray;
      this.vertexCount = this.mesh.geometry.index.count;
      this.indexFormat = this.mesh.geometry.index.array instanceof Uint32Array ? 'uint32' : 'uint16';
    }
    else {
      this.vertexCount = this.mesh.geometry.attributes.position.count;
    }

    this.vertexBuffers = vertexBufferFactory.createResource(this.vertexBufferAttributes, this.vertexBufferData);

  }

  public async initGroupResource() {

    const material = this.mesh.material as THREE.MeshLambertMaterial;

    this.resourceAttributes = ['material', 'particlePosition'];
    this.resourceCPUData = {
      material: {
        value: new Float32Array([
          ...material.color.toArray(), 1
        ])
      }
    };
    
    this.resource = await resourceFactory.createResource(['material'], this.resourceCPUData);
    this.resource.particlePosition = this.simulator.position;
    
  }

  protected async initPipeline(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {
    
    const vertexLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' }
      }, { // material
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
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
      layout: bindGroupLayout,
      entries: [{ // camera
        binding: 0,
        resource: { buffer: globalResource.camera as GPUBuffer },
      }, { // material
        binding: 1,
        resource: { buffer: this.resource.material as GPUBuffer }
      }, { // instance positions
        binding: 2,
        resource: { buffer: this.simulator.position }
      }, { // light
        binding: 3,
        resource: { buffer: globalResource.directionalLight as GPUBuffer }
      }]
    })
    
    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: vertexShader }),
        entryPoint: 'main',
        buffers: vertexLayout
      },
      fragment: {
        module: device.createShaderModule({ code: fragmentShader }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back'
      }, 
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });
    
  }

  public render(renderPassEncoder: GPURenderPassEncoder) {

    renderPassEncoder.setPipeline(this.renderPipeline);

    // set vertex and index buffers
    let loction = 0;
    let indexed = false;
    for (const attribute of this.vertexBufferAttributes) {
      if (attribute === 'index') {
        renderPassEncoder.setIndexBuffer(this.vertexBuffers.index, this.indexFormat);
        indexed = true;
      }
      else {
        renderPassEncoder.setVertexBuffer(loction, this.vertexBuffers[attribute]);
        loction++;
      }
    }

    // set bind group
    renderPassEncoder.setBindGroup(0, this.bindGroup);

    // draw
    if (indexed) renderPassEncoder.drawIndexed(this.vertexCount, this.simulator.particleCount);
    else renderPassEncoder.draw(this.vertexCount, this.simulator.particleCount);
    
  }

}

export { RawParticles };
