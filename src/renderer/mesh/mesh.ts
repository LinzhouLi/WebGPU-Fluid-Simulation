import * as THREE from 'three';
import type { ResourceType, BufferData, TextureData } from '../../common/resourceFactory';
import type { TypedArray } from '../../common/base';
import { device, canvasFormat } from '../../controller';
import { vertexBufferFactory, resourceFactory, bindGroupFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { VertexShader, FragmentShader } from './shader';


class Mesh {

  private static ResourceFormats = {
    transform: {
      type: 'buffer' as ResourceType,
      label: 'Transform Matrix', 
      visibility: GPUShaderStage.VERTEX,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      layout: { 
        type: 'uniform' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },

    meshMaterial: {
      type: 'buffer' as ResourceType,
      label: 'Mesh Material Structure', 
      visibility: GPUShaderStage.FRAGMENT,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      layout: { 
        type: 'uniform' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },
  };

  protected mesh: THREE.Mesh;

  protected vertexCount: number;
  protected vertexBufferAttributes: Array<string>;
  protected vertexBufferData: Record<string, TypedArray>;
  protected vertexBuffers: Record<string, GPUBuffer>;

  protected resourceAttributes: Array<string>;
  protected resourceData: Record<string, BufferData | TextureData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  protected bindgroupLayout: GPUBindGroupLayout;
  protected bindgroup: GPUBindGroup;
  protected pipeline: GPURenderPipeline;

  protected vertexShader: string;
  protected fragmentShader: string;

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(Mesh.ResourceFormats);
  }

  constructor() {

    this.vertexShader = VertexShader;
    this.fragmentShader = FragmentShader;

    this.vertexBufferAttributes = ['position', 'normal', 'uv'];
    this.resourceAttributes = ['transform', 'meshMaterial'];

  }

  protected initVertexBuffer() {

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

  protected async initGroupResource() {

    const material = this.mesh.material as THREE.MeshPhongMaterial;
    this.mesh.normalMatrix.getNormalMatrix(this.mesh.matrixWorld);
    const normalMatArray = this.mesh.normalMatrix.toArray();

    this.resourceAttributes = ['transform', 'meshMaterial'];
    this.resourceData = {
      transform: {
        value: new Float32Array([
          ...this.mesh.matrixWorld.toArray(),
          ...normalMatArray.slice(0, 3), 0,
          ...normalMatArray.slice(3, 6), 0,
          ...normalMatArray.slice(6, 9), 0
        ])
      },
      meshMaterial: {
        value: new Float32Array([
          material.shininess, 0, 0, 0, // ???
          ...material.color.toArray(), 0 // fix bug: bind group is too small!
        ])
      }
    };

    this.resource = await resourceFactory.createResource(this.resourceAttributes, this.resourceData);

  }

  protected initBindGroup() {

    const layout_group = bindGroupFactory.create(this.resourceAttributes, this.resource);
    this.bindgroupLayout = layout_group.layout;
    this.bindgroup = layout_group.group;

  }
  
  public async initPipeline(globalBindGroupLayout: GPUBindGroupLayout) {

    const vertexLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    const bindgroupLayout = bindGroupFactory.createLayout(this.resourceAttributes);

    this.pipeline = await device.createRenderPipelineAsync({
      label: 'Mesh Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [globalBindGroupLayout, bindgroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: 
          this.vertexShader
        }),
        entryPoint: 'main',
        buffers: vertexLayout
      },
      fragment: {
        module: device.createShaderModule({ code: 
          this.fragmentShader
        }),
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

  public async setMesh(mesh: THREE.Mesh) {

    this.mesh = mesh;
    this.initVertexBuffer();
    await this.initGroupResource();
    this.initBindGroup();

  }

  public render(
    encoder: GPURenderBundleEncoder | GPURenderPassEncoder
  ) {
    
    // set vertex and index buffers
    let loction = 0;
    let indexed = false;
    for (const attribute of this.vertexBufferAttributes) {
      if (attribute === 'index') {
        encoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
        indexed = true;
      }
      else {
        encoder.setVertexBuffer(loction, this.vertexBuffers[attribute]);
        loction++;
      }
    }

    // set bind group
    encoder.setBindGroup(1, this.bindgroup);
    encoder.setPipeline(this.pipeline);

    // draw
    if (indexed) encoder.drawIndexed(this.vertexCount);
    else encoder.draw(this.vertexCount);

  }

}

export { Mesh }; 