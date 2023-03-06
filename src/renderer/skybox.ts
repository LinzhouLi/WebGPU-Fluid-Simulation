import { canvasFormat, device } from '../controller';
import {
  vertexBufferFactory,
  bindGroupFactory
} from '../common/base';
import { ShaderFunction } from "../common/shaderFunction";

const skyboxVertexShader = /* wgsl */`
struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) vPosition: vec3<f32>,
};

@vertex
fn main(@location(0) position: vec3<f32>) -> VertOutput {
  let positionCamera = camera.viewMatrix * vec4<f32>(position, 0.0);
  let positionNDC = camera.projectionMatrix * vec4<f32>(positionCamera.xyz, 1.0);
  let positionReverseZ = vec4<f32>(positionNDC.xy, 0.0, positionNDC.w);
  return VertOutput(positionReverseZ, position);
}
`;


const skyboxFragmentShader = /* wgsl */`
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var envMap: texture_cube<f32>;

${ShaderFunction.sRGBGammaEncode}

@fragment
fn main(
  @builtin(position) position : vec4<f32>,
  @location(0) fragPosition : vec3<f32>,
) -> @location(0) vec4<f32> {
  let color_linear = textureSampleLevel(envMap, linearSampler, fragPosition, 0);
  return vec4<f32>(sRGBGammaEncode(color_linear.xyz), 1.0);
}
`;


class Skybox {

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected vertexCount: number;
  protected vertexBufferAttributes: string[]; // resource name
  protected vertexBuffers: { [x: string]: GPUBuffer }; // resource in GPU

  protected vertexBufferLayout: GPUVertexBufferLayout[];
  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;
  protected renderPipeline: GPURenderPipeline;

  constructor() {

    this.vertexShaderCode = skyboxVertexShader;
    this.fragmentShaderCode = skyboxFragmentShader;

  };

  public initVertexBuffer() {

    const attributes = ['position', 'index'];
    const position = new Float32Array([
      -1.0,  1.0, -1.0,
      -1.0,  1.0,  1.0,
       1.0,  1.0,  1.0,
       1.0,  1.0, -1.0,
  
      -1.0, -1.0, -1.0,
      -1.0, -1.0,  1.0,
       1.0, -1.0,  1.0,
       1.0, -1.0, -1.0
    ]);
    const index = new Uint16Array([
      3, 6, 7,   3, 2, 6, // +x
      0, 5, 1,   0, 4, 5, // -x
      0, 1, 2,   0, 2, 3, // +y
      4, 6, 5,   4, 7, 6, // -y
      1, 5, 6,   1, 6, 2, // +z
      0, 7, 4,   0, 3, 7  // -z
    ]);

    this.renderPipeline = undefined;
    this.vertexCount = 36;
    this.vertexBufferAttributes = attributes;

    this.vertexBufferLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    this.vertexBuffers = vertexBufferFactory.createResource(attributes, { position, index });

  }

  public async initGroupResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    const layout_group = bindGroupFactory.create(
      ['camera', 'linearSampler', 'envMap'],
      globalResource
    );
    this.bindGroupLayout = layout_group.layout;
    this.bindGroup = layout_group.group;

  }

  public async initPipeline() {

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Skybox Render Pipeline',
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
        targets:[{ format: canvasFormat }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'front' // 天空盒使用正面剔除
      }, 
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater-equal', // 深度比较改为 less-equal
        format: 'depth32float'
      }
    });

  }

  public async setRenderBundle(
    bundleEncoder: GPURenderBundleEncoder
  ) {
    
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
    bundleEncoder.setVertexBuffer(0, this.vertexBuffers.position);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.drawIndexed(this.vertexCount);

  }

}

export { Skybox };