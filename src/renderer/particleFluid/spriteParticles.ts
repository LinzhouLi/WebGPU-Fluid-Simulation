import { device, canvasSize } from '../../controller';
import { SpriteParticles } from '../spriteParticles/particles';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { depthPassfragmentShader } from './shader/depthPassShader';
import { volumePassfragmentShader } from './shader/volumePassShader';


class FluidParicles extends SpriteParticles {

  protected depthRenderPipeline: GPURenderPipeline;
  protected volumeRenderPipeline: GPURenderPipeline;

  protected depthRenderBundle: GPURenderBundle;
  protected volumeRenderBundle: GPURenderBundle;

  protected depthStencilMap: GPUTexture;

  constructor(simulator: LagrangianSimulator) {

    super(simulator);

  }

  public override async initPipeline() {

    // depth pass
    this.depthRenderPipeline = await device.createRenderPipelineAsync({
      label: 'Sprite Particle Depth Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        entryPoint: 'main',
        buffers: this.vertexBufferLayout
      },
      fragment: {
        module: device.createShaderModule({ code: depthPassfragmentShader }),
        entryPoint: 'main',
        targets: [{ format: 'r32float' }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back'
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
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        entryPoint: 'main',
        buffers: this.vertexBufferLayout
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
        topology: 'triangle-list',
        cullMode: 'back'
      },
    });

  }

  private setRenderCommands(
    bundleEncoder: GPURenderBundleEncoder,
    pipeline: GPURenderPipeline
  ) {

    bundleEncoder.setPipeline(pipeline);
    let loction = 0, indexed = false;
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
    bundleEncoder.setBindGroup(0, this.bindGroup);
    if (indexed) bundleEncoder.drawIndexed(this.vertexCount, this.simulator.particleCount);
    else bundleEncoder.draw(this.vertexCount, this.simulator.particleCount);

  }

  public initRenderBundle() {

    // depth stencil map
    this.depthStencilMap = device.createTexture({
      size: [ canvasSize.width, canvasSize.height ],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

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

  public render(
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
        view: this.depthStencilMap.createView(),
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
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

export { FluidParicles };