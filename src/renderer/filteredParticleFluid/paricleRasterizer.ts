import { device } from '../../controller';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { depthPassVertexShader, depthPassfragmentShader } from './shader/depthPassShader';
import { volumePassVertexShader, volumePassfragmentShader } from './shader/volumePassShader';


class ParicleRasterizer {

  protected simulator: LagrangianSimulator;

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;

  protected depthRenderPipeline: GPURenderPipeline;
  protected volumeRenderPipeline: GPURenderPipeline;

  protected depthRenderBundle: GPURenderBundle;
  protected volumeRenderBundle: GPURenderBundle;

  protected depthStencilView: GPUTextureView;

  constructor(simulator: LagrangianSimulator) {

    this.simulator = simulator;

  }

  public async initResource(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.depthStencilView = (resource.renderDepthMap as GPUTexture).createView();

    this.initBindGroup(resource);
    await this.initPipeline()
    this.initRenderBundle();

  }

  protected initBindGroup(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {
    
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // Rendering Options
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
        resource: { buffer: resource.camera as GPUBuffer },
      }, { // Rendering Options
        binding: 1,
        resource: { buffer: resource.renderingOptions as GPUBuffer }
      }, { // instance positions
        binding: 2,
        resource: { 
          buffer: this.simulator.position, 
          size: 4 * this.simulator.particleCount * Float32Array.BYTES_PER_ELEMENT
        }
      }, { // light
        binding: 3,
        resource: { buffer: resource.directionalLight as GPUBuffer }
      }]
    })
  }

  protected async initPipeline() {

    // depth pass
    this.depthRenderPipeline = await device.createRenderPipelineAsync({
      label: 'Sprite Particle Depth Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: depthPassVertexShader }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: depthPassfragmentShader }),
        entryPoint: 'main',
        targets: [{ format: 'r32float' }]
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      }, 
      depthStencil: {
        depthWriteEnabled: true, // enable depth test
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

    // volume pass
    this.volumeRenderPipeline = await device.createRenderPipelineAsync({
      label: 'Sprite Particle Depth Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: volumePassVertexShader }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: volumePassfragmentShader }),
        entryPoint: 'main',
        targets: [{ // volume blend
          format: 'r16float',
          writeMask: GPUColorWrite.RED,
          blend: {
            color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
          }
        }]
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      },
    });

  }

  private setRenderCommands(
    bundleEncoder: GPURenderBundleEncoder,
    pipeline: GPURenderPipeline
  ) {

    bundleEncoder.setPipeline(pipeline);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.draw(4, this.simulator.particleCount);

  }

  protected initRenderBundle() {

    // depth pass
    let bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ 'r32float' ],
      depthStencilFormat: 'depth32float' // format of depthMap
    });
    this.setRenderCommands(bundleEncoder, this.depthRenderPipeline);
    this.depthRenderBundle = bundleEncoder.finish();

    // volume pass
    bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ 'r16float' ]
    });
    this.setRenderCommands(bundleEncoder, this.volumeRenderPipeline);
    this.volumeRenderBundle = bundleEncoder.finish();

  }

  public execute(
    commandEncoder: GPUCommandEncoder,
    depthMap: GPUTexture,
    volumeMap: GPUTexture
  ) {

    const renderPassEncoder1 = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: depthMap.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: this.depthStencilView,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      }
    });
    renderPassEncoder1.executeBundles([this.depthRenderBundle]);
    renderPassEncoder1.end();

    const renderPassEncoder2 = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: volumeMap.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    renderPassEncoder2.executeBundles([this.volumeRenderBundle]);
    renderPassEncoder2.end();

  }

}

export { ParicleRasterizer };