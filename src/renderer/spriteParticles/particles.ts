import * as THREE from 'three';
import type { TypedArray } from '../../common/base';
import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { canvasFormat, device } from '../../controller';
import { resourceFactory, vertexBufferFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { vertexShader, fragmentShader } from './shader';

class SpriteParticles {

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

  protected radius: number;
  protected spriteMesh: THREE.Mesh;
  protected simulator: LagrangianSimulator;

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected vertexCount: number;
  protected vertexBufferAttributes: string[]; // resource name
  protected vertexBufferData: Record<string, TypedArray>; // resource in CPU
  protected vertexBuffers: Record<string, GPUBuffer>; // resource in GPU

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
    this.radius = 0.03;
    this.spriteMesh = new THREE.Mesh(
      new THREE.PlaneGeometry( 1, 1 ),
      new THREE.MeshLambertMaterial({ color: 0x049ef4 })
    );
    
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(SpriteParticles.ResourceFormats);
  }

  public initVertexBuffer() {

    this.vertexBufferAttributes = ['position', 'uv'];
    this.vertexBufferData = {
      position: this.spriteMesh.geometry.attributes.position.array as TypedArray,
      uv: this.spriteMesh.geometry.attributes.uv.array as TypedArray,
    };

    if (!!this.spriteMesh.geometry.index) {
      this.vertexBufferAttributes.push('index');
      this.vertexBufferData.index = this.spriteMesh.geometry.index.array as TypedArray;
      this.vertexCount = this.spriteMesh.geometry.index.count;
    }
    else {
      this.vertexCount = this.spriteMesh.geometry.attributes.position.count;
    }

    this.vertexBufferLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    this.vertexBuffers = vertexBufferFactory.createResource(this.vertexBufferAttributes, this.vertexBufferData);

  }

  public async initGroupResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    const material = this.spriteMesh.material as THREE.MeshPhysicalMaterial;

    this.resourceAttributes = ['material', 'particlePosition'];
    this.resourceCPUData = {
      material: { 
        value: new Float32Array([
          this.radius,
          material.metalness, 
          material.specularIntensity, 
          material.roughness, // fix bug: don't need alignment
          ...material.color.toArray(), 0 // fix bug: bind group is too small!
        ])
      }
    };
    
    this.resource = await resourceFactory.createResource(['material'], this.resourceCPUData);
    this.resource.particlePosition = this.simulator.particlePositionBuffer;
    
    this.initBindGroup(globalResource);
    
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
        resource: { buffer: this.resource.material as GPUBuffer }
      }, { // instance positions
        binding: 2,
        resource: { buffer: this.simulator.particlePositionBuffer }
      }, { // light
        binding: 3,
        resource: { buffer: globalResource.directionalLight as GPUBuffer }
      }]
    })
  }

  public async initPipeline() {

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
        topology: 'triangle-list',
        cullMode: 'none' // do not need cull
      }, 
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

  }

  public async setRenderBundle(
    bundleEncoder: GPURenderBundleEncoder
  ) {
    
    bundleEncoder.setPipeline(this.renderPipeline);

    // set vertex and index buffers
    let loction = 0;
    let indexed = false;
    for (const attribute of this.vertexBufferAttributes) {
      if (attribute === 'index') {
        bundleEncoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
        indexed = true;
      }
      else {
        bundleEncoder.setVertexBuffer(loction, this.vertexBuffers[attribute]);
        loction++;
      }
    }

    // set bind group
    bundleEncoder.setBindGroup(0, this.bindGroup);

    // draw
    if (indexed) bundleEncoder.drawIndexed(this.vertexCount, this.simulator.particleCount);
    else bundleEncoder.draw(this.vertexCount, this.simulator.particleCount);
    
  }

}

export { SpriteParticles }