import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { device, canvasFormat, canvasSize } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { FluidParicles } from './spriteParticles';
import { Postprocess } from './postprocess';

class ParticleFluid {

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats({

      fluidDepthMap: {
        type: 'texture' as ResourceType,
          label: 'Fluid Depth Map',
          visibility: GPUShaderStage.FRAGMENT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
          size: [canvasSize.width, canvasSize.height],
          dimension: '2d' as GPUTextureDimension,
          format: 'depth32float' as GPUTextureFormat,
          layout: { // for post process
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '2d' as GPUTextureViewDimension,
          } as GPUTextureBindingLayout
      },
  
      fluidDepthMapTemp: {
        type: 'texture' as ResourceType,
          label: 'Fluid Depth Map for Filtering',
          visibility: GPUShaderStage.FRAGMENT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
          size: [canvasSize.width, canvasSize.height],
          dimension: '2d' as GPUTextureDimension,
          format: 'depth32float' as GPUTextureFormat,
          layout: { // for post process
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '2d' as GPUTextureViewDimension,
          } as GPUTextureBindingLayout
      },
  
      fluidVolumeMap: {
        type: 'texture' as ResourceType,
          label: 'Fluid Volume Map',
          visibility: GPUShaderStage.FRAGMENT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
          size: [canvasSize.width / 2, canvasSize.height / 2],
          dimension: '2d' as GPUTextureDimension,
          format: 'r16float' as GPUTextureFormat,
          layout: { // for post process
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '2d' as GPUTextureViewDimension,
          } as GPUTextureBindingLayout
      },
  
    });
  }

  protected fluidParticles: FluidParicles;
  protected postprocess: Postprocess;

  protected resourceAttributes: string[]; // resource name
  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU
  protected renderDepthMap: GPUTexture;

  protected particleRenderBundle: GPURenderBundle;
  protected postprocessBundle: GPURenderBundle;

  constructor(simulator: LagrangianSimulator) {

    this.fluidParticles = new FluidParicles(simulator);
    this.postprocess = new Postprocess();
    
  }


  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.renderDepthMap = globalResource.renderDepthMap as GPUTexture;
    this.resourceAttributes = [ 'fluidDepthMap', 'fluidDepthMapTemp', 'fluidVolumeMap' ];
    this.resource = await resourceFactory.createResource(this.resourceAttributes, { });

    // point sprite render
    this.fluidParticles.initVertexBuffer();
    await this.fluidParticles.initGroupResource(globalResource);
    await this.fluidParticles.initPipeline();
    this.fluidParticles.initRenderBundle();

    // post process render
    this.postprocess.initBindGroup({ ...globalResource, ...this.resource });
    await this.postprocess.initPipeline();
    this.postprocess.initRenderBundle();

  }

  public render(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    // render point sprite
    this.fluidParticles.render(
      commandEncoder, 
      this.resource.fluidDepthMap as GPUTexture, 
      this.resource.fluidVolumeMap as GPUTexture
    );

    this.postprocess.render(
      commandEncoder,
      ctxTextureView
    );

  }

}

export { ParticleFluid }