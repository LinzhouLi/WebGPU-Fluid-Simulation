import { device, canvasFormat, canvasSize } from '../../controller';
import { SPH } from '../../simulator/SPH';
import { vertexShader as billboardVS, fragmentShader as billboardFS } from './billboardShader';
import { fullScreenVertexShader, filterFragmentShader } from './filterShader';
import { vertexShader as surfaceVS, fragmentShader as surfaceFS } from './surfaceShader';

class FluidRenderer {

  private simulator: SPH;
  private particleRadiusBuffer: GPUBuffer;

  // Intermediate textures
  private billboardDepth: GPUTexture;          // r32float — stores billboard depth values as color
  private billboardDepthView: GPUTextureView;
  private fluidDepthFilteredH: GPUTexture;
  private fluidDepthFilteredHView: GPUTextureView;
  private fluidDepthFiltered: GPUTexture;
  private fluidDepthFilteredView: GPUTextureView;
  private sceneColor: GPUTexture;
  private sceneColorView: GPUTextureView;
  private sceneDepth: GPUTexture;
  private sceneDepthView: GPUTextureView;

  // Sampler
  private pointSampler: GPUSampler;

  // Pipelines
  private billboardPipeline: GPURenderPipeline;
  private filterHPipeline: GPURenderPipeline;
  private filterVPipeline: GPURenderPipeline;
  private surfacePipeline: GPURenderPipeline;

  // Bind groups
  private billboardBindGroup1: GPUBindGroup;
  private filterHBindGroup: GPUBindGroup;
  private filterVBindGroup: GPUBindGroup;
  private surfaceBindGroup1: GPUBindGroup;

  // Separate uniform buffers for H and V filter passes (avoids writeBuffer race)
  private filterHParamsBuffer: GPUBuffer;
  private filterVParamsBuffer: GPUBuffer;

  private invProjBuffer: GPUBuffer;
  private invProjArray: Float32Array;
  private surfaceParamsBuffer: GPUBuffer;
  private surfaceParamsArray: Float32Array;

  constructor(simulator: SPH, particleRadius: number = 0.006) {
    this.simulator = simulator;
    this.particleRadiusBuffer = this.createUniformBuffer(
      new Float32Array([particleRadius]),
      'Particle Radius'
    );
  }

  private createUniformBuffer(data: Float32Array, label: string): GPUBuffer {
    const b = device.createBuffer({
      label,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(b, 0, data);
    return b;
  }

  private createTex(
    format: GPUTextureFormat,
    usage: number,
    label: string,
  ): [GPUTexture, GPUTextureView] {
    const t = device.createTexture({
      label,
      size: [canvasSize.width, canvasSize.height, 1],
      format,
      usage,
    });
    return [t, t.createView()];
  }

  async initResource(globalBindGroupLayout: GPUBindGroupLayout) {
    const w = canvasSize.width;
    const h = canvasSize.height;

    // Billboard depth — written as color to r32float, sampled by filter
    [this.billboardDepth, this.billboardDepthView] = this.createTex(
      'r32float',
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      'Billboard Depth'
    );

    // Filter output textures
    [this.fluidDepthFilteredH, this.fluidDepthFilteredHView] = this.createTex(
      'r32float',
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      'Fluid Depth Filtered H'
    );
    [this.fluidDepthFiltered, this.fluidDepthFilteredView] = this.createTex(
      'r32float',
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      'Fluid Depth Filtered'
    );

    // Scene render targets
    [this.sceneColor, this.sceneColorView] = this.createTex(
      canvasFormat,
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      'Scene Color'
    );
    [this.sceneDepth, this.sceneDepthView] = this.createTex(
      'depth32float',
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      'Scene Depth'
    );

    // Sampler
    this.pointSampler = device.createSampler({
      label: 'Point Sampler',
      magFilter: 'nearest',
      minFilter: 'nearest',
    });

    // Filter params — 6 floats (24 bytes) matching padded WGSL struct
    // Layout: texelSize(2), isHorizontal(1), sigmaRange(1), kernelRadius(1), _pad(1)
    const filterHData = new Float32Array([1.0 / w, 1.0 / h, 1.0, 0.001, 4.0, 0.0]);
    const filterVData = new Float32Array([1.0 / w, 1.0 / h, 0.0, 0.001, 4.0, 0.0]);
    this.filterHParamsBuffer = this.createUniformBuffer(filterHData, 'Filter H Params');
    this.filterVParamsBuffer = this.createUniformBuffer(filterVData, 'Filter V Params');

    this.surfaceParamsArray = new Float32Array([1.0 / w, 1.0 / h, 0.752, 0.0]); // texelSize, eta
    this.surfaceParamsBuffer = this.createUniformBuffer(this.surfaceParamsArray, 'Surface Params');

    this.invProjArray = new Float32Array(16);
    this.invProjBuffer = device.createBuffer({
      label: 'Inv Projection',
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    await this.initPipelines(globalBindGroupLayout);
    this.initBindGroups();
  }

  private async initPipelines(globalBindGroupLayout: GPUBindGroupLayout) {

    // Billboard pipeline
    const billboardBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.billboardPipeline = await device.createRenderPipelineAsync({
      label: 'Billboard Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [globalBindGroupLayout, billboardBGL],
      }),
      vertex: {
        module: device.createShaderModule({ code: billboardVS }),
        entryPoint: 'main',
      },
      fragment: {
        module: device.createShaderModule({ code: billboardFS }),
        entryPoint: 'main',
        targets: [{ format: 'r32float' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater',
        format: 'depth32float',
      },
    });

    // Filter pipeline
    const filterBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const filterLayout = device.createPipelineLayout({ bindGroupLayouts: [filterBGL] });

    const filterDesc: GPURenderPipelineDescriptor = {
      label: 'Filter Pipeline',
      layout: filterLayout,
      vertex: {
        module: device.createShaderModule({ code: fullScreenVertexShader }),
        entryPoint: 'main',
      },
      fragment: {
        module: device.createShaderModule({ code: filterFragmentShader }),
        entryPoint: 'main',
        targets: [{ format: 'r32float' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    };

    this.filterHPipeline = await device.createRenderPipelineAsync(filterDesc);
    this.filterVPipeline = await device.createRenderPipelineAsync({ ...filterDesc, label: 'Filter V Pipeline' });

    // Surface pipeline — binding 3 uses sampleType 'depth' for texture_depth_2d
    const surfaceBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }, // fluidDepth (r32float)
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },              // sceneColor
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },              // sceneDepth (texture_depth_2d)
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },                    // surfaceParams
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },                    // invProj
      ],
    });

    this.surfacePipeline = await device.createRenderPipelineAsync({
      label: 'Surface Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [globalBindGroupLayout, surfaceBGL],
      }),
      vertex: {
        module: device.createShaderModule({ code: surfaceVS }),
        entryPoint: 'main',
      },
      fragment: {
        module: device.createShaderModule({ code: surfaceFS }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  private initBindGroups() {
    this.billboardBindGroup1 = device.createBindGroup({
      label: 'Billboard BG 1',
      layout: this.billboardPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.simulator.position } },
        { binding: 1, resource: { buffer: this.particleRadiusBuffer } },
      ],
    });

    // Filter H and V use separate params buffers with baked-in direction
    this.filterHBindGroup = device.createBindGroup({
      label: 'Filter H BG',
      layout: this.filterHPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.billboardDepthView },
        { binding: 1, resource: this.pointSampler },
        { binding: 2, resource: { buffer: this.filterHParamsBuffer } },
      ],
    });

    this.filterVBindGroup = device.createBindGroup({
      label: 'Filter V BG',
      layout: this.filterVPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.fluidDepthFilteredHView },
        { binding: 1, resource: this.pointSampler },
        { binding: 2, resource: { buffer: this.filterVParamsBuffer } },
      ],
    });

    this.surfaceBindGroup1 = device.createBindGroup({
      label: 'Surface BG 1',
      layout: this.surfacePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: this.fluidDepthFilteredView },
        { binding: 1, resource: this.pointSampler },
        { binding: 2, resource: this.sceneColorView },
        { binding: 3, resource: this.sceneDepthView },
        { binding: 4, resource: { buffer: this.surfaceParamsBuffer } },
        { binding: 5, resource: { buffer: this.invProjBuffer } },
      ],
    });
  }

  updateInvProjection(projectionMatrixInverse: Float32Array | number[]) {
    this.invProjArray.set(projectionMatrixInverse);
    device.queue.writeBuffer(this.invProjBuffer, 0, this.invProjArray);
  }

  // === Render passes ===

  beginScenePass(commandEncoder: GPUCommandEncoder): GPURenderPassEncoder {
    return commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.sceneColorView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.sceneDepthView,
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
  }

  // Billboard pass reuses scene depth for occlusion: particles behind scene
  // geometry are culled by the depth test. Particle depths are written to
  // sceneDepth so the surface shader later compares against the closest surface.
  beginBillboardPass(commandEncoder: GPUCommandEncoder): GPURenderPassEncoder {
    return commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.billboardDepthView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.sceneDepthView,
        depthClearValue: 0.0,
        depthLoadOp: 'load',   // preserve scene depth for occlusion
        depthStoreOp: 'store',
      },
    });
  }

  beginFilterHPass(commandEncoder: GPUCommandEncoder): GPURenderPassEncoder {
    return commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.fluidDepthFilteredHView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
  }

  beginFilterVPass(commandEncoder: GPUCommandEncoder): GPURenderPassEncoder {
    return commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.fluidDepthFilteredView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
  }

  // Creates a render pass that renders directly to canvas with depth attachment.
  // Used as fallback when no particles are present.
  beginCanvasPass(commandEncoder: GPUCommandEncoder, ctxView: GPUTextureView): GPURenderPassEncoder {
    return commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: ctxView,
        clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.sceneDepthView,
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
  }

  renderBillboards(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.billboardPipeline);
    pass.setBindGroup(1, this.billboardBindGroup1);
    pass.draw(6, this.simulator.particleCount, 0, 0);
  }

  renderFilterH(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.filterHPipeline);
    pass.setBindGroup(0, this.filterHBindGroup);
    pass.draw(3, 1, 0, 0);
  }

  renderFilterV(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.filterVPipeline);
    pass.setBindGroup(0, this.filterVBindGroup);
    pass.draw(3, 1, 0, 0);
  }

  renderSurface(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.surfacePipeline);
    pass.setBindGroup(1, this.surfaceBindGroup1);
    pass.draw(3, 1, 0, 0);
  }
}

export { FluidRenderer };
