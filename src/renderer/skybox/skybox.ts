import { canvasFormat, device } from '../../controller';
import { vertexBufferFactory } from '../../common/base';
import { VertexShader, FragmentShader } from './shader';


class Skybox {

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected vertexCount: number;
  protected vertexBufferAttributes: string[]; // resource name
  protected vertexBuffers: { [x: string]: GPUBuffer }; // resource in GPU
  protected vertexBufferLayout: GPUVertexBufferLayout[];

  protected renderPipeline: GPURenderPipeline;

  constructor() {

    this.vertexShaderCode = VertexShader;
    this.fragmentShaderCode = FragmentShader;

  };

  protected initVertexBuffer() {

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

  protected async initPipeline(globalBindGroupLayout: GPUBindGroupLayout) {

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Skybox Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [globalBindGroupLayout] 
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

  public async initResouce(globalBindGroupLayout: GPUBindGroupLayout) {

    this.initVertexBuffer();
    await this.initPipeline(globalBindGroupLayout);

  }

  public render(
    encoder: GPURenderBundleEncoder | GPURenderPassEncoder
  ) {
    
    encoder.setPipeline(this.renderPipeline);
    encoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
    encoder.setVertexBuffer(0, this.vertexBuffers.position);
    encoder.drawIndexed(this.vertexCount);

  }

}

export { Skybox };