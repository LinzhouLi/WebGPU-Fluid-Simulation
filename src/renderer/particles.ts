import * as THREE from 'three';
import type { TypedArray } from '../common/base';
import type { ResourceType, BufferData } from '../common/resourceFactory';
import { device, canvasFormat } from '../controller';
import { resourceFactory, vertexBufferFactory } from '../common/base';
import { ResourceFactory } from '../common/resourceFactory';
import { BaseSimulator } from '../simulator/baseSimulator';
import { vertexShader, fragmentShader } from './shader';

class Particles {

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
  protected simulator: BaseSimulator;

  protected vertexCount: number;
  protected vertexBufferAttributes: string[]; // resource name
  protected vertexBufferData: Record<string, TypedArray>; // resource in CPU
  protected vertexBuffers: Record<string, GPUBuffer>; // resource in GPU

  protected resourceAttributes: string[]; // resource name
  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  constructor(simulator: BaseSimulator) {

    this.simulator = simulator;
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 8, 8), // radius, width segment, height segment
      new THREE.MeshLambertMaterial({ color: 0x049ef4 })
    );
    
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(Particles.ResourceFormats);
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
    }
    else {
      this.vertexCount = this.mesh.geometry.attributes.position.count;
    }

    this.vertexBuffers = vertexBufferFactory.createResource(this.vertexBufferAttributes, this.vertexBufferData);

  }

  public async initGroupResource() {

    const material = this.mesh.material as THREE.MeshPhysicalMaterial;

    this.resourceAttributes = ['material', 'particlePosition'];
    this.resourceCPUData = {
      material: { 
        value: new Float32Array([
          material.metalness, 
          material.specularIntensity, 
          material.roughness,
          0, // for alignment
          ...material.color.toArray(), 0 // fix bug: bind group is too small!
        ])
      }
    };
    
    this.resource = await resourceFactory.createResource(['material'], this.resourceCPUData);
    this.resource.particlePosition = this.simulator.particlePositionBuffer;
    
  }

  public async setRenderBundle(
    bundleEncoder: GPURenderBundleEncoder,
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {
    
    const vertexLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
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

    const bindGroup = device.createBindGroup({
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
        resource: { buffer: this.simulator.particlePositionBuffer }
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
    bundleEncoder.setBindGroup(0, bindGroup);

    // draw
    if (indexed) bundleEncoder.drawIndexed(this.vertexCount, this.simulator.particleCount);
    else bundleEncoder.draw(this.vertexCount, this.simulator.particleCount);
    
  }

}

export { Particles };