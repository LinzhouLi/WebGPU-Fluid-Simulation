import { device } from '../controller';
import {
  vertexBufferFactory,
  bindGroupFactory
} from '../common/base';
import { canvasFormat } from '../controller'

const skyboxVertexShader = /* wgsl */`
struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>
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

fn sRGBGammaEncode(color: vec3<f32>) -> vec3<f32> {
  return mix(
    color.rgb * 12.92,                                    // x <= 0.0031308
    pow(color.rgb, vec3<f32>(0.41666)) * 1.055 - 0.055,   // x >  0.0031308
    saturate(sign(color.rgb - 0.0031308))
  );
}

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

  private renderPipeline: GPURenderPipeline;
  private vertexCount: number;
  private vertexBufferAttributes: string[]; // resource name
  private vertexBuffers: { [x: string]: GPUBuffer }; // resource in GPU

  constructor() {  };

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
    this.vertexBuffers = vertexBufferFactory.createResource(attributes, { position, index });

  }

  public async setRenderBundle(
    bundleEncoder: GPURenderBundleEncoder,
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    const vertexBufferLayout = vertexBufferFactory.createLayout(this.vertexBufferAttributes);
    const { layout, group } = bindGroupFactory.create(
      ['camera', 'linearSampler', 'envMap'],
      globalResource
    );
    
    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Skybox Render Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module: device.createShaderModule({ code: skyboxVertexShader }),
        entryPoint: 'main',
        buffers: vertexBufferLayout
      },
      fragment: {
        module: device.createShaderModule({ code: skyboxFragmentShader }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }]
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
    
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
    bundleEncoder.setVertexBuffer(0, this.vertexBuffers.position);
    bundleEncoder.setBindGroup(0, group);
    bundleEncoder.drawIndexed(this.vertexCount);

  }

}

export { Skybox };