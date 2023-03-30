import { device, canvasSize, canvasFormat } from '../../controller';
import { bindGroupFactory } from '../../common/base';
import { vertexShader } from './shader/screenVertexShader';
import { fragmentShader } from './shader/renderPassShader';

class ScreenSpaceRenderer {

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;
  protected renderPipeline: GPURenderPipeline;
  protected renderBundle: GPURenderBundle;

  constructor() {

    this.vertexShaderCode = vertexShader;
    this.fragmentShaderCode = fragmentShader;

  }

  public async initResource(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.initBindGroup(resource);
    await this.initPipeline();
    this.initRenderBundle();

  }

  private initBindGroup(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    const layout_group = bindGroupFactory.create(
      [ 
        'camera', 'directionalLight', 'linearSampler', 
        'fluidDepthMap', 'fluidVolumeMap', 'envMap'
      ],
      resource
    );
    this.bindGroupLayout = layout_group.layout;
    this.bindGroup = layout_group.group;

  }

  private async initPipeline() {

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Postprocess Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        constants: {
          screenWidth: canvasSize.width,
          screenHeight: canvasSize.height
        },
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: this.fragmentShaderCode }),
        entryPoint: 'main',
        targets: [{ 
          format: canvasFormat,
          writeMask: GPUColorWrite.RED | GPUColorWrite.GREEN | GPUColorWrite.BLUE,
          blend: {
            color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }
          }
        }]
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      }
    });

  }

  private initRenderBundle() {

    let bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ canvasFormat ]}
    );
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.draw(4);
    this.renderBundle = bundleEncoder.finish();

  }

  public execute(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    const renderPassEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: ctxTextureView,
        loadOp: 'load',
        storeOp: 'store'
      }]
    });
    renderPassEncoder.executeBundles([this.renderBundle]);
    renderPassEncoder.end();

  }

}

export { ScreenSpaceRenderer };