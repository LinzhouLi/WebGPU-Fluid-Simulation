import { device, canvasSize, canvasFormat } from '../../controller';
import { bindGroupFactory } from '../../common/base';
import { vertexShader } from './shader/screenVertexShader';
import { fragmentShader } from './shader/renderPassShader';
import computeNormalShader from './shader/computeNormal.wgsl?raw';

class ScreenSpaceRenderer {

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;
  protected renderPipeline: GPURenderPipeline;
  protected renderBundle: GPURenderBundle;

  protected normalBindGroupLayout: GPUBindGroupLayout;
  protected normalBindGroup: GPUBindGroup;
  protected normalPipeline: GPUComputePipeline;

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
        'camera', 'directionalLight', 'renderingOptions', 'linearSampler',
        'fluidDepthMap', 'fluidVolumeMap', 'envMap', 'fluidNormalMap'
      ],
      resource
    );
    this.bindGroupLayout = layout_group.layout;
    this.bindGroup = layout_group.group;

    this.normalBindGroupLayout = device.createBindGroupLayout({
      label: 'Fluid Normal Compute Bind Group Layout',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' }
      }, {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float' }
      }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }]
    });

    this.normalBindGroup = device.createBindGroup({
      label: 'Fluid Normal Compute Bind Group',
      layout: this.normalBindGroupLayout,
      entries: [{
        binding: 0,
        resource: (resource.fluidDepthMap as GPUTexture).createView()
      }, {
        binding: 1,
        resource: (resource.fluidNormalMap as GPUTexture).createView()
      }, {
        binding: 2,
        resource: { buffer: resource.camera as GPUBuffer }
      }]
    });

  }

  private async initPipeline() {

    this.normalPipeline = await device.createComputePipelineAsync({
      label: 'Fluid Normal Compute Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.normalBindGroupLayout]
      }),
      compute: {
        module: device.createShaderModule({ code: computeNormalShader }),
        entryPoint: 'main'
      }
    });

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
      colorFormats: [ canvasFormat ]
    });
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.draw(4);
    this.renderBundle = bundleEncoder.finish();

  }

  public execute(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    const computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(this.normalPipeline);
    computePassEncoder.setBindGroup(0, this.normalBindGroup);
    computePassEncoder.dispatchWorkgroups(
      Math.ceil(canvasSize.width / 8),
      Math.ceil(canvasSize.height / 8)
    );
    computePassEncoder.end();

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
