import * as THREE from 'three';
import type { ResourceType, BufferData, TextureData } from '../../common/resourceFactory';
import type { TypedArray } from '../../common/base';
import { device } from '../../controller';
import { vertexBufferFactory, resourceFactory, bindGroupFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { VertexShader, FragmentShader } from './shader';


class Mesh {

  private static ResourceFormats = {
    material: {
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

  protected vertexShader: string;
  protected fragmentShader: string;

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(Mesh.ResourceFormats);
  }

  constructor(mesh: THREE.Mesh) {

    this.mesh = mesh;
    this.vertexShader = VertexShader;
    this.fragmentShader = FragmentShader;

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

  public async initGroupResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    const material = this.mesh.material as THREE.MeshPhongMaterial;
    this.mesh.normalMatrix.getNormalMatrix(this.mesh.matrixWorld);
    const normalMatArray = this.mesh.normalMatrix.toArray();

    this.resourceAttributes = ['transform', 'material'];
    this.resourceData = {
      transform: {
        value: new Float32Array([
          ...this.mesh.matrixWorld.toArray(),
          ...normalMatArray.slice(0, 3), 0,
          ...normalMatArray.slice(3, 6), 0,
          ...normalMatArray.slice(6, 9), 0
        ])
      },
      material: {
        value: new Float32Array([
          material.shininess, // ???
          ...material.color.toArray(), 0 // fix bug: bind group is too small!
        ])
      }
    };

    if (!!material.map) {
      this.resourceAttributes.push('baseMap');
      this.resourceData.baseMap = { 
        value: await resourceFactory.toBitmap(material.map.image), 
        flipY: material.map.flipY 
      };
    }

    this.resource = await resourceFactory.createResource(this.resourceAttributes, this.resourceData);

  }

}

export { Mesh }; 