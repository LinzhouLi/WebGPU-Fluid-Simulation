import * as THREE from 'three';
import { device, canvasFormat, canvasSize } from '../../controller';
import type { SPH } from '../../simulator/SPH';
import type { GlobalResource } from '../globalResource';
import { DepthShader, ThicknessShader } from './particleShader';
import { FilterShader1D, FilterShaderCleanup } from './filterShader';
import { CompositeShader } from './compositeShader';

// Screen space fluid renderer following
//   Truong & Yuksel, "A Narrow-Range Filter for Screen-Space Fluid Rendering", i3D 2018.
//
// Per frame, after the scene (mesh + skybox) has been rendered:
//   1. thickness pass  particle imposters accumulated with additive blending
//   2. depth pass      particle imposters writing the eye space depth of the surface
//   3. filter passes   narrow-range filter, separable approximation + clean-up
//   4. composite pass  surface shading blended over the scene
//
// Both imposter passes use the scene depth buffer as a depth attachment, so
// particles hidden behind scene geometry are discarded during rasterization and
// never reach the depth map (filtering them would drag the surface across the
// occluder boundary). The thickness pass must run first, since it needs to
// accumulate every particle behind the surface and therefore cannot see the
// fluid depth that the depth pass writes.

interface FluidRenderOptions {
  particleScale: number,
  filterSigma: number,
  filterDelta: number,
  filterMu: number,
  maxFilterSigma: number,
  iteration: number,
  cleanUp: boolean,
  ior: number,
  absorption: number,
  opacity: number,
  fluidColor: string
}

const OPTIONS_BUFFER_SIZE = 64;
const WORKGROUP_SIZE = 8;
const MAX_KERNEL_RADIUS = 48;

class FluidRenderer {

  private simulator: SPH;
  private globalResource: GlobalResource;

  // depth map ping-pong buffers, index of the one holding the latest result
  private depthTextures: GPUTexture[];
  private depthViews: GPUTextureView[];
  private resultIndex: number;

  private thicknessView: GPUTextureView;
  private sceneDepthView: GPUTextureView;

  private optionsArray: ArrayBuffer;
  private optionsView: DataView;
  private optionsBuffer: GPUBuffer;

  private particleBindGroup: GPUBindGroup;
  private filterBindGroups: GPUBindGroup[];    // [i] reads depthTextures[i], writes the other
  private compositeBindGroups: GPUBindGroup[]; // [i] reads depthTextures[i]

  private depthPipeline: GPURenderPipeline;
  private thicknessPipeline: GPURenderPipeline;
  private filterPipelines: GPUComputePipeline[]; // horizontal, vertical
  private cleanupPipeline: GPUComputePipeline;
  private compositePipeline: GPURenderPipeline;

  private iterationCount: number;
  private cleanUp: boolean;

  constructor(simulator: SPH, globalResource: GlobalResource) {

    this.simulator = simulator;
    this.globalResource = globalResource;
    this.iterationCount = 2;
    this.cleanUp = true;
    this.resultIndex = 0;

  }

  public async initResource(camera: THREE.PerspectiveCamera) {

    this.createTextures();
    this.createOptionsBuffer(camera);
    await this.initPipeline();

  }

  private createTextures() {

    const size = [canvasSize.width, canvasSize.height];

    // r32float: renderable and storable, but not filterable, so every shader
    // reading it uses textureLoad() and an 'unfilterable-float' binding.
    this.depthTextures = [0, 1].map(i => device.createTexture({
      label: `Fluid Depth Map ${i}`,
      size, format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    }));
    this.depthViews = this.depthTextures.map(texture => texture.createView());

    // r16float: blendable and filterable (but not a storage format, so the
    // thickness map is smoothed with a few bilinear taps while compositing).
    this.thicknessView = device.createTexture({
      label: 'Fluid Thickness Map',
      size, format: 'r16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    }).createView();

    this.sceneDepthView = (this.globalResource.resource.renderDepthMap as GPUTexture).createView();

  }

  private createOptionsBuffer(camera: THREE.PerspectiveCamera) {

    this.optionsArray = new ArrayBuffer(OPTIONS_BUFFER_SIZE);
    this.optionsView = new DataView(this.optionsArray);
    this.optionsBuffer = device.createBuffer({
      label: 'Fluid Render Options',
      size: OPTIONS_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Constant terms of the screen space kernel size (Eq.5).
    this.optionsView.setFloat32(20, canvasSize.height, true);
    this.optionsView.setFloat32(24, Math.tan(0.5 * camera.fov * Math.PI / 180), true);

  }

  public setConfig(config: FluidRenderOptions) {

    this.iterationCount = config.iteration;
    this.cleanUp = config.cleanUp;

    // The filter parameters are expressed in multiples of the imposter radius,
    // matching the paper's delta = 10r / mu = r convention.
    const radius = config.particleScale * this.simulator.particleRadius;
    // THREE.ColorManagement is enabled, so the sRGB input is already converted
    // to the linear working space that the shading expects
    const color = new THREE.Color(config.fluidColor);

    this.optionsView.setFloat32(0, radius, true);
    this.optionsView.setFloat32(4, config.filterSigma * radius, true);
    this.optionsView.setFloat32(8, config.filterDelta * radius, true);
    this.optionsView.setFloat32(12, config.filterMu * radius, true);
    this.optionsView.setFloat32(16, config.maxFilterSigma, true);
    this.optionsView.setFloat32(28, config.ior, true);
    this.optionsView.setFloat32(32, color.r, true);
    this.optionsView.setFloat32(36, color.g, true);
    this.optionsView.setFloat32(40, color.b, true);
    this.optionsView.setFloat32(44, config.absorption, true);
    this.optionsView.setFloat32(48, config.opacity, true);

    device.queue.writeBuffer(this.optionsBuffer, 0, this.optionsArray);

  }

  public optionsChange(e: any) {
    this.setConfig(e.object);
  }

  private async initPipeline() {

    const globalLayout = this.globalResource.bindgroupLayout;

    // imposter passes
    const particleLayout = device.createBindGroupLayout({
      label: 'Fluid Particle Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
      ]
    });

    this.particleBindGroup = device.createBindGroup({
      label: 'Fluid Particle Bind Group',
      layout: particleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.simulator.position } },
        { binding: 1, resource: { buffer: this.optionsBuffer } }
      ]
    });

    const imposterPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [globalLayout, particleLayout]
    });
    const imposterPrimitive = {
      topology: 'triangle-strip' as GPUPrimitiveTopology,
      cullMode: 'none' as GPUCullMode
    };

    const depthModule = device.createShaderModule({ code: DepthShader });
    this.depthPipeline = await device.createRenderPipelineAsync({
      label: 'Fluid Depth Pipeline',
      layout: imposterPipelineLayout,
      vertex: { module: depthModule, entryPoint: 'main' },
      fragment: {
        module: depthModule, entryPoint: 'depth',
        targets: [{ format: 'r32float' }]
      },
      primitive: imposterPrimitive,
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater', // reverse Z
        format: 'depth32float'
      }
    });

    const thicknessModule = device.createShaderModule({ code: ThicknessShader });
    const additiveBlend = {
      color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation },
      alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }
    };
    this.thicknessPipeline = await device.createRenderPipelineAsync({
      label: 'Fluid Thickness Pipeline',
      layout: imposterPipelineLayout,
      vertex: { module: thicknessModule, entryPoint: 'main' },
      fragment: {
        module: thicknessModule, entryPoint: 'thickness',
        targets: [{ format: 'r16float', blend: additiveBlend }]
      },
      primitive: imposterPrimitive,
      depthStencil: {
        depthWriteEnabled: false, // every particle behind the surface contributes
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

    // filter passes
    const filterLayout = device.createBindGroupLayout({
      label: 'Fluid Filter Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
      ]
    });

    this.filterBindGroups = [0, 1].map(i => device.createBindGroup({
      label: `Fluid Filter Bind Group ${i}`,
      layout: filterLayout,
      entries: [
        { binding: 0, resource: this.depthViews[i] },
        { binding: 1, resource: this.depthViews[1 - i] },
        { binding: 2, resource: { buffer: this.optionsBuffer } }
      ]
    }));

    const filterPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [filterLayout] });
    const filter1DModule = device.createShaderModule({ code: FilterShader1D });
    this.filterPipelines = await Promise.all([[1, 0], [0, 1]].map(
      ([x, y]) => device.createComputePipelineAsync({
        label: `Fluid Narrow Range Filter Pipeline (${x == 1 ? 'horizontal' : 'vertical'})`,
        layout: filterPipelineLayout,
        compute: {
          module: filter1DModule, entryPoint: 'main',
          constants: { FilterDirectionX: x, FilterDirectionY: y, MaxKernelRadius: MAX_KERNEL_RADIUS }
        }
      })
    ));

    this.cleanupPipeline = await device.createComputePipelineAsync({
      label: 'Fluid Filter Clean-up Pipeline',
      layout: filterPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: FilterShaderCleanup }),
        entryPoint: 'main'
      }
    });

    // composite pass
    const compositeLayout = device.createBindGroupLayout({
      label: 'Fluid Composite Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
      ]
    });

    this.compositeBindGroups = [0, 1].map(i => device.createBindGroup({
      label: `Fluid Composite Bind Group ${i}`,
      layout: compositeLayout,
      entries: [
        { binding: 0, resource: this.depthViews[i] },
        { binding: 1, resource: this.thicknessView },
        { binding: 2, resource: { buffer: this.optionsBuffer } }
      ]
    }));

    const compositeModule = device.createShaderModule({ code: CompositeShader });
    this.compositePipeline = await device.createRenderPipelineAsync({
      label: 'Fluid Composite Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [globalLayout, compositeLayout] }),
      vertex: { module: compositeModule, entryPoint: 'vertex' },
      fragment: {
        module: compositeModule, entryPoint: 'fragment',
        targets: [{
          format: canvasFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
          }
        }]
      },
      primitive: { topology: 'triangle-list' }
    });

  }

  public render(commandEncoder: GPUCommandEncoder, canvasView: GPUTextureView) {

    if (this.simulator.particleCount === 0) return;

    this.renderThickness(commandEncoder);
    this.renderDepth(commandEncoder);
    this.filter(commandEncoder);
    this.composite(commandEncoder, canvasView);

  }

  private renderThickness(commandEncoder: GPUCommandEncoder) {

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.thicknessView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      // read only: the scene depth culls particles behind geometry without
      // letting the imposters occlude each other
      depthStencilAttachment: { view: this.sceneDepthView, depthReadOnly: true }
    });

    this.globalResource.setResource(passEncoder);
    passEncoder.setPipeline(this.thicknessPipeline);
    passEncoder.setBindGroup(1, this.particleBindGroup);
    passEncoder.draw(4, this.simulator.particleCount);
    passEncoder.end();

  }

  private renderDepth(commandEncoder: GPUCommandEncoder) {

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.depthViews[0],
        clearValue: { r: 0, g: 0, b: 0, a: 0 }, // 0 marks a pixel without fluid
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: this.sceneDepthView,
        depthLoadOp: 'load',
        depthStoreOp: 'store'
      }
    });

    this.globalResource.setResource(passEncoder);
    passEncoder.setPipeline(this.depthPipeline);
    passEncoder.setBindGroup(1, this.particleBindGroup);
    passEncoder.draw(4, this.simulator.particleCount);
    passEncoder.end();

  }

  private filter(commandEncoder: GPUCommandEncoder) {

    const workgroupCountX = Math.ceil(canvasSize.width / WORKGROUP_SIZE);
    const workgroupCountY = Math.ceil(canvasSize.height / WORKGROUP_SIZE);

    const passEncoder = commandEncoder.beginComputePass();

    let source = 0;
    for (let i = 0; i < this.iterationCount; i++) {
      for (const pipeline of this.filterPipelines) { // alternating directions
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, this.filterBindGroups[source]);
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        source = 1 - source;
      }
    }

    if (this.cleanUp) {
      passEncoder.setPipeline(this.cleanupPipeline);
      passEncoder.setBindGroup(0, this.filterBindGroups[source]);
      passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
      source = 1 - source;
    }

    passEncoder.end();
    this.resultIndex = source;

  }

  private composite(commandEncoder: GPUCommandEncoder, canvasView: GPUTextureView) {

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: canvasView,
        loadOp: 'load', // blend over the scene
        storeOp: 'store'
      }]
    });

    this.globalResource.setResource(passEncoder);
    passEncoder.setPipeline(this.compositePipeline);
    passEncoder.setBindGroup(1, this.compositeBindGroups[this.resultIndex]);
    passEncoder.draw(3);
    passEncoder.end();

  }

}

export { FluidRenderer };
