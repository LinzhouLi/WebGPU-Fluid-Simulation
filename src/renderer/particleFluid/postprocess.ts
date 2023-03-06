import { device, canvasSize, canvasFormat } from '../../controller';
import { vertexShader } from './shader/screenVertexShader';
import { fragmentShader } from './shader/renderPassShader';

class Postprocess {

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

  public initBindGroup(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // light
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // sampler
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' }
      }, { // fluid depth map
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'unfilterable-float' }
      }, { // fluid volume map
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' }
      }]
    });

    this.bindGroup = device.createBindGroup({
      label: 'Particle Rendering Pipeline Bind Group',
      layout: this.bindGroupLayout,
      entries: [{ // camera
        binding: 0,
        resource: { buffer: resource.camera as GPUBuffer },
      }, { // light
        binding: 1,
        resource: { buffer: resource.directionalLight as GPUBuffer }
      }, { // sampler
        binding: 2,
        resource: resource.linearSampler as GPUSampler
      }, { // fluid depth map
        binding: 3,
        resource: (resource.fluidDepthStorageMap as GPUTexture).createView()
      }, { // fluid volume map
        binding: 4,
        resource: (resource.fluidVolumeMap as GPUTexture).createView()
      }]
    });

  }

  public async initPipeline() {

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

  public initRenderBundle() {

    // depth pass
    let bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ canvasFormat ]}
    );
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.draw(4);
    this.renderBundle = bundleEncoder.finish();

  }

  public render(
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

export { Postprocess };