import { canvasFormat, canvasSize, device } from '../../controller';
import { SPH } from '../../simulator/SPH';
import {
  billboardVertexShader,
  particleDepthFragmentShader,
  particleThicknessFragmentShader,
  narrowRangeFilterShader,
  cleanupFilterShader,
  compositeVertexShader,
  compositeFragmentShader,
} from './shader';

/** Billboard depth/thickness generation, narrow-range filtering, and fluid composition. */
class RawParticles {
  private simulator: SPH;
  private width: number;
  private height: number;
  private rawDepthTexture: GPUTexture;
  private filteredDepthA: GPUTexture;
  private filteredDepthB: GPUTexture;
  private particleDepthTexture: GPUTexture;
  private thicknessTexture: GPUTexture;
  private billboardBindGroup: GPUBindGroup;
  private depthPipeline: GPURenderPipeline;
  private thicknessPipeline: GPURenderPipeline;
  private filterBindGroups: GPUBindGroup[];
  private horizontalFilterPipeline: GPUComputePipeline;
  private verticalFilterPipeline: GPUComputePipeline;
  private cleanupFilterPipeline: GPUComputePipeline;
  private compositeBindGroup: GPUBindGroup;
  private compositePipeline: GPURenderPipeline;

  constructor(simulator: SPH) {
    this.simulator = simulator;
    this.width = Math.max(1, Math.floor(canvasSize.width));
    this.height = Math.max(1, Math.floor(canvasSize.height));
  }

  public static RegisterResourceFormats() { }

  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {
    const screenSize: GPUExtent3D = [this.width, this.height, 1];
    const depthTextureDescriptor: GPUTextureDescriptor = {
      size: screenSize,
      format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
        | GPUTextureUsage.TEXTURE_BINDING
        | GPUTextureUsage.STORAGE_BINDING,
    };
    this.rawDepthTexture = device.createTexture({ ...depthTextureDescriptor, label: 'Raw Particle Eye Depth' });
    this.filteredDepthA = device.createTexture({ ...depthTextureDescriptor, label: 'Filtered Fluid Depth A' });
    this.filteredDepthB = device.createTexture({ ...depthTextureDescriptor, label: 'Filtered Fluid Depth B' });
    this.particleDepthTexture = device.createTexture({
      label: 'Particle Depth Test', size: screenSize, format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.thicknessTexture = device.createTexture({
      label: 'Fluid Thickness', size: screenSize, format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    const billboardParams = device.createBuffer({
      label: 'Fluid Billboard Parameters', size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    // radius is enlarged slightly to close gaps; sigma/delta/mu use the
    // paper's 0.7r, 10r and r values for the simulation's r=0.006.
    new Float32Array(billboardParams.getMappedRange()).set([0.009, 0.0042, 0.06, 0.006]);
    billboardParams.unmap();
    const filterParams = device.createBuffer({
      label: 'Narrow Range Filter Parameters', size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Float32Array(filterParams.getMappedRange()).set([0.0042, 0.06, 0.006, 0.0]);
    filterParams.unmap();

    const billboardLayout = device.createBindGroupLayout({
      label: 'Fluid Billboard Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.billboardBindGroup = device.createBindGroup({
      label: 'Fluid Billboard Bind Group', layout: billboardLayout,
      entries: [
        { binding: 0, resource: { buffer: globalResource.camera as GPUBuffer } },
        { binding: 1, resource: { buffer: this.simulator.position } },
        { binding: 2, resource: { buffer: billboardParams } },
      ],
    });

    const billboardModule = device.createShaderModule({ code: billboardVertexShader });
    this.depthPipeline = await device.createRenderPipelineAsync({
      label: 'Fluid Billboard Depth Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [billboardLayout] }),
      vertex: { module: billboardModule, entryPoint: 'main' },
      fragment: {
        module: device.createShaderModule({ code: particleDepthFragmentShader }),
        entryPoint: 'main', targets: [{ format: 'r32float' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    this.thicknessPipeline = await device.createRenderPipelineAsync({
      label: 'Fluid Billboard Thickness Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [billboardLayout] }),
      vertex: { module: billboardModule, entryPoint: 'main' },
      fragment: {
        module: device.createShaderModule({ code: particleThicknessFragmentShader }), entryPoint: 'main',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const filterLayout = device.createBindGroupLayout({
      label: 'Narrow Range Filter Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const makeFilterGroup = (input: GPUTexture, output: GPUTexture, label: string) => device.createBindGroup({
      label, layout: filterLayout,
      entries: [
        { binding: 0, resource: { buffer: globalResource.camera as GPUBuffer } },
        { binding: 1, resource: input.createView() },
        { binding: 2, resource: output.createView() },
        { binding: 3, resource: { buffer: filterParams } },
      ],
    });
    this.filterBindGroups = [
      makeFilterGroup(this.rawDepthTexture, this.filteredDepthA, 'Raw to Filter A'),
      makeFilterGroup(this.filteredDepthA, this.filteredDepthB, 'Filter A to B'),
      makeFilterGroup(this.filteredDepthB, this.filteredDepthA, 'Filter B to A'),
    ];
    const filterModule = device.createShaderModule({ code: narrowRangeFilterShader });
    const filterPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [filterLayout] });
    this.horizontalFilterPipeline = await device.createComputePipelineAsync({
      label: 'Narrow Range Horizontal Filter', layout: filterPipelineLayout,
      compute: { module: filterModule, entryPoint: 'horizontal' },
    });
    this.verticalFilterPipeline = await device.createComputePipelineAsync({
      label: 'Narrow Range Vertical Filter', layout: filterPipelineLayout,
      compute: { module: filterModule, entryPoint: 'vertical' },
    });
    this.cleanupFilterPipeline = await device.createComputePipelineAsync({
      label: 'Narrow Range 5x5 Cleanup Filter', layout: filterPipelineLayout,
      compute: { module: device.createShaderModule({ code: cleanupFilterShader }), entryPoint: 'main' },
    });

    const compositeLayout = device.createBindGroupLayout({
      label: 'Screen Space Fluid Composite Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: 'cube' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.compositeBindGroup = device.createBindGroup({
      label: 'Screen Space Fluid Composite Bind Group', layout: compositeLayout,
      entries: [
        { binding: 0, resource: { buffer: globalResource.camera as GPUBuffer } },
        { binding: 1, resource: this.filteredDepthA.createView() },
        { binding: 2, resource: this.thicknessTexture.createView() },
        { binding: 3, resource: globalResource.linearSampler as GPUSampler },
        { binding: 4, resource: (globalResource.envMap as GPUTexture).createView({ format: 'rgba8unorm-srgb', dimension: 'cube' }) },
        { binding: 5, resource: { buffer: globalResource.directionalLight as GPUBuffer } },
      ],
    });
    this.compositePipeline = await device.createRenderPipelineAsync({
      label: 'Screen Space Fluid Composite Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [compositeLayout] }),
      vertex: { module: device.createShaderModule({ code: compositeVertexShader }), entryPoint: 'main' },
      fragment: {
        module: device.createShaderModule({ code: compositeFragmentShader }), entryPoint: 'main',
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
  }

  private encodeFilterPass(
    commandEncoder: GPUCommandEncoder, pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup, label: string,
  ) {
    const pass = commandEncoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
  }

  public prepare(commandEncoder: GPUCommandEncoder) {
    const depthPass = commandEncoder.beginRenderPass({
      label: 'Fluid Particle Depth Pass',
      colorAttachments: [{
        view: this.rawDepthTexture.createView(),
        clearValue: { r: -1000000.0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.particleDepthTexture.createView(),
        depthClearValue: 0.0, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });
    depthPass.setPipeline(this.depthPipeline);
    depthPass.setBindGroup(0, this.billboardBindGroup);
    depthPass.draw(6, this.simulator.particleCount);
    depthPass.end();

    const thicknessPass = commandEncoder.beginRenderPass({
      label: 'Fluid Thickness Pass',
      colorAttachments: [{
        view: this.thicknessTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store',
      }],
    });
    thicknessPass.setPipeline(this.thicknessPipeline);
    thicknessPass.setBindGroup(0, this.billboardBindGroup);
    thicknessPass.draw(6, this.simulator.particleCount);
    thicknessPass.end();

    this.encodeFilterPass(commandEncoder, this.horizontalFilterPipeline, this.filterBindGroups[0], 'Narrow Range H1');
    this.encodeFilterPass(commandEncoder, this.verticalFilterPipeline, this.filterBindGroups[1], 'Narrow Range V1');
    this.encodeFilterPass(commandEncoder, this.horizontalFilterPipeline, this.filterBindGroups[2], 'Narrow Range H2');
    this.encodeFilterPass(commandEncoder, this.verticalFilterPipeline, this.filterBindGroups[1], 'Narrow Range V2');
    this.encodeFilterPass(commandEncoder, this.cleanupFilterPipeline, this.filterBindGroups[2], 'Narrow Range Cleanup');
  }

  public render(renderPassEncoder: GPURenderPassEncoder) {
    renderPassEncoder.setPipeline(this.compositePipeline);
    renderPassEncoder.setBindGroup(0, this.compositeBindGroup);
    renderPassEncoder.draw(3);
  }
}

export { RawParticles };
