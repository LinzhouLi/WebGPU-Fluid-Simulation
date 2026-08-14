import { device, canvasSize, canvasFormat } from '../../controller';
import { SPH } from '../../simulator/SPH';
import { ResourceFactory } from '../../common/resourceFactory';
import type { ResourceType } from '../../common/resourceFactory';
import {
  PrepassVertexShader,
  PrepassFragmentShader,
  ConvertShader,
  FilterHShader,
  FilterVShader,
  CleanUpShader,
  CompositeVertexShader,
  CompositeFragmentShader
} from './shader';

/**
 * Screen-space fluid renderer (Narrow-Range Filter, i3D 2018).
 *
 * Frame graph (called from Controller.run):
 *   renderPrepass(encoder) : billboard splat -> particleDepthMap (depth-only pass)
 *   runFilter(encoder)     : convert + separable narrow-range filter + clean-up
 *   composite(encoder)     : screen-space shading into the scene pass
 */
class FluidRenderer {

  private simulator: SPH;
  private particleRadius: number;

  // textures
  public particleDepthMap: GPUTexture;
  private smoothA: GPUTexture;
  private smoothB: GPUTexture;

  // buffers
  private fluidParamsBuffer: GPUBuffer;
  private filterParamsBuffer: GPUBuffer;
  private filterParamsArray: Float32Array;

  // pipelines
  private prepassPipeline: GPURenderPipeline;
  private convertPipeline: GPUComputePipeline;
  private filterHPipeline: GPUComputePipeline;
  private filterVPipeline: GPUComputePipeline;
  private cleanUpPipeline: GPUComputePipeline;
  private compositePipeline: GPURenderPipeline;

  // bind groups
  private prepassGroup1: GPUBindGroup;
  private convertGroup: GPUBindGroup;
  private filterGroups: { [key: string]: GPUBindGroup };
  private compositeGroupA: GPUBindGroup;
  private compositeGroupB: GPUBindGroup;

  // filter settings
  private iterations = 2;
  private sigmaScale = 0.7;
  private deltaScale = 10;
  private muScale = 1;

  constructor(simulator: SPH, particleRadius: number = 0.006) {
    this.simulator = simulator;
    this.particleRadius = particleRadius;
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats({

      // particle billboard depth (sampled by the convert pass)
      fluidParticleDepthMap: {
        type: 'texture' as ResourceType,
        label: 'Fluid Particle Depth Map',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        size: [canvasSize.width, canvasSize.height],
        dimension: '2d' as GPUTextureDimension,
        format: 'depth32float' as GPUTextureFormat,
      },

      // ping-pong smoothed eye-depth textures
      fluidSmoothA: {
        type: 'texture' as ResourceType,
        label: 'Fluid Smoothed Depth A',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        size: [canvasSize.width, canvasSize.height],
        dimension: '2d' as GPUTextureDimension,
        format: 'r32float' as GPUTextureFormat,
      },
      fluidSmoothB: {
        type: 'texture' as ResourceType,
        label: 'Fluid Smoothed Depth B',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        size: [canvasSize.width, canvasSize.height],
        dimension: '2d' as GPUTextureDimension,
        format: 'r32float' as GPUTextureFormat,
      },

    });
  }

  public async initResource(globalResource: {
    resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>,
    bindgroupLayout: GPUBindGroupLayout
  }) {

    const resourceFactory = new ResourceFactory();
    const textures = await resourceFactory.createResource(
      ['fluidParticleDepthMap', 'fluidSmoothA', 'fluidSmoothB'], {}
    );
    this.particleDepthMap = textures.fluidParticleDepthMap as GPUTexture;
    this.smoothA = textures.fluidSmoothA as GPUTexture;
    this.smoothB = textures.fluidSmoothB as GPUTexture;

    // uniform buffers
    this.fluidParamsBuffer = device.createBuffer({
      label: 'Fluid Params',
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(
      this.fluidParamsBuffer, 0,
      new Float32Array([this.particleRadius, 0, 0, 0])
    );

    this.filterParamsArray = new Float32Array(4);
    this.filterParamsBuffer = device.createBuffer({
      label: 'Filter Params',
      size: this.filterParamsArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.updateFilterParams();

    this.initPipelines(globalResource.bindgroupLayout);
    this.initBindGroups(globalResource);

  }

  private updateFilterParams() {
    this.filterParamsArray.set([
      this.sigmaScale * this.particleRadius,
      this.deltaScale * this.particleRadius,
      this.muScale * this.particleRadius,
      0
    ]);
    device.queue.writeBuffer(this.filterParamsBuffer, 0, this.filterParamsArray);
  }

  public setFilterParams(config: {
    sigma: number,
    delta: number,
    mu: number,
    iterations: number
  }) {
    this.sigmaScale = config.sigma;
    this.deltaScale = config.delta;
    this.muScale = config.mu;
    this.iterations = Math.max(1, Math.floor(config.iterations));
    this.updateFilterParams();
  }

  private initPipelines(globalBindGroupLayout: GPUBindGroupLayout) {

    // prepass: billboard particles -> depth only
    const prepassGroup1Layout = device.createBindGroupLayout({
      label: 'Fluid Prepass Group 1 Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
      ]
    });
    this.prepassPipeline = device.createRenderPipeline({
      label: 'Fluid Prepass Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [globalBindGroupLayout, prepassGroup1Layout]
      }),
      vertex: {
        module: device.createShaderModule({ code: PrepassVertexShader }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: PrepassFragmentShader }),
        entryPoint: 'main',
        targets: []
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

    // convert: reverse-Z depth -> linear eye depth
    const convertLayout = device.createBindGroupLayout({
      label: 'Fluid Convert Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } }
      ]
    });
    this.convertPipeline = device.createComputePipeline({
      label: 'Fluid Depth Convert Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [convertLayout] }),
      compute: {
        module: device.createShaderModule({ code: ConvertShader }),
        entryPoint: 'main'
      }
    });

    // narrow-range filter / clean-up
    const filterLayout = device.createBindGroupLayout({
      label: 'Fluid Filter Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } }
      ]
    });
    const makeComputePipeline = (label: string, code: string) => device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [filterLayout] }),
      compute: {
        module: device.createShaderModule({ code }),
        entryPoint: 'main'
      }
    });
    this.filterHPipeline = makeComputePipeline('Fluid Filter H Pipeline', FilterHShader);
    this.filterVPipeline = makeComputePipeline('Fluid Filter V Pipeline', FilterVShader);
    this.cleanUpPipeline = makeComputePipeline('Fluid Clean-Up Pipeline', CleanUpShader);

    // composite: screen-space shading
    const compositeGroup1Layout = device.createBindGroupLayout({
      label: 'Fluid Composite Group 1 Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }
      ]
    });
    this.compositePipeline = device.createRenderPipeline({
      label: 'Fluid Composite Pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [globalBindGroupLayout, compositeGroup1Layout]
      }),
      vertex: {
        module: device.createShaderModule({ code: CompositeVertexShader }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ code: CompositeFragmentShader }),
        entryPoint: 'main',
        targets: [{ format: canvasFormat }]
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

    // stash layouts for bind group creation
    this.prepassGroup1Layout = prepassGroup1Layout;
    this.filterLayout = filterLayout;
    this.compositeGroup1Layout = compositeGroup1Layout;

  }

  private prepassGroup1Layout: GPUBindGroupLayout;
  private filterLayout: GPUBindGroupLayout;
  private compositeGroup1Layout: GPUBindGroupLayout;

  private initBindGroups(globalResource: {
    resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>,
    bindgroupLayout: GPUBindGroupLayout
  }) {

    const camera = globalResource.resource.camera as GPUBuffer;

    this.prepassGroup1 = device.createBindGroup({
      layout: this.prepassGroup1Layout,
      entries: [
        { binding: 0, resource: { buffer: this.simulator.position } },
        { binding: 1, resource: { buffer: this.fluidParamsBuffer } }
      ]
    });

    const writeView = (tex: GPUTexture) => tex.createView();
    const readView = (tex: GPUTexture) => tex.createView();

    this.convertGroup = device.createBindGroup({
      layout: this.convertPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.filterParamsBuffer } },
        { binding: 1, resource: { buffer: camera } },
        { binding: 2, resource: this.particleDepthMap.createView() },
        { binding: 3, resource: writeView(this.smoothA) }
      ]
    });

    this.filterGroups = {};
    const makeFilterGroup = (inTex: GPUTexture, outTex: GPUTexture) => device.createBindGroup({
      layout: this.filterLayout,
      entries: [
        { binding: 0, resource: { buffer: this.filterParamsBuffer } },
        { binding: 1, resource: { buffer: camera } },
        { binding: 2, resource: readView(inTex) },
        { binding: 3, resource: writeView(outTex) }
      ]
    });
    this.filterGroups['AB'] = makeFilterGroup(this.smoothA, this.smoothB);
    this.filterGroups['BA'] = makeFilterGroup(this.smoothB, this.smoothA);

    this.compositeGroupA = device.createBindGroup({
      layout: this.compositeGroup1Layout,
      entries: [{ binding: 0, resource: readView(this.smoothA) }]
    });
    this.compositeGroupB = device.createBindGroup({
      layout: this.compositeGroup1Layout,
      entries: [{ binding: 0, resource: readView(this.smoothB) }]
    });

  }

  /* ---------------- frame graph entry points ---------------- */

  public renderPrepass(renderPassEncoder: GPURenderPassEncoder) {
    renderPassEncoder.setPipeline(this.prepassPipeline);
    renderPassEncoder.setBindGroup(1, this.prepassGroup1);
    renderPassEncoder.draw(6, this.simulator.particleCount);
  }

  public runFilter(commandEncoder: GPUCommandEncoder): boolean {

    const pass = commandEncoder.beginComputePass();
    const workgroups = [
      Math.ceil(canvasSize.width / 16),
      Math.ceil(canvasSize.height / 16)
    ];

    // convert particle depth -> eye depth in A
    pass.setPipeline(this.convertPipeline);
    pass.setBindGroup(0, this.convertGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1]);

    // separable narrow-range filter iterations
    let srcIsA = true;
    for (let i = 0; i < 2 * this.iterations; i++) {
      const group = srcIsA ? this.filterGroups['AB'] : this.filterGroups['BA'];
      pass.setPipeline(i % 2 == 0 ? this.filterHPipeline : this.filterVPipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(workgroups[0], workgroups[1]);
      srcIsA = !srcIsA;
    }

    // final 5x5 clean-up
    const cleanGroup = srcIsA ? this.filterGroups['AB'] : this.filterGroups['BA'];
    pass.setPipeline(this.cleanUpPipeline);
    pass.setBindGroup(0, cleanGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1]);
    srcIsA = !srcIsA;

    pass.end();
    return srcIsA; // which texture holds the final result

  }

  public composite(renderPassEncoder: GPURenderPassEncoder, finalIsA: boolean) {
    renderPassEncoder.setPipeline(this.compositePipeline);
    renderPassEncoder.setBindGroup(1, finalIsA ? this.compositeGroupA : this.compositeGroupB);
    renderPassEncoder.draw(3);
  }

}

export { FluidRenderer };
