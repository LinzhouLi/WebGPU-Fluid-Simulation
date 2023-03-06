import type { ResourceType, BufferData } from '../../common/resourceFactory';
import { canvasSize } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { FluidParicles } from './spriteParticles';
import { TextureCopy } from './textureCopy';
import { TextureFilter } from './textureFilter';
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
          layout: {
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '2d' as GPUTextureViewDimension,
          } as GPUTextureBindingLayout
      },
  
      fluidDepthStorageMap: {
        type: 'texture' as ResourceType,
          label: 'Fluid Depth Storage Map for Filtering',
          visibility: GPUShaderStage.FRAGMENT,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
          size: [canvasSize.width, canvasSize.height],
          dimension: '2d' as GPUTextureDimension,
          format: 'r32float' as GPUTextureFormat,
          layout: {
            sampleType: 'unfilterable-float' as GPUTextureSampleType,
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
          layout: {
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '2d' as GPUTextureViewDimension,
          } as GPUTextureBindingLayout
      },
  
    });
  }

  protected fluidParticles: FluidParicles;
  protected textureCopy: TextureCopy;
  protected textureFilter: TextureFilter;
  protected postprocess: Postprocess;

  protected resourceAttributes: string[]; // resource name
  protected resourceCPUData: Record<string, BufferData>; // resource in CPU
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU
  protected renderDepthMap: GPUTexture;

  protected particleRenderBundle: GPURenderBundle;
  protected postprocessBundle: GPURenderBundle;

  constructor(simulator: LagrangianSimulator) {

    this.fluidParticles = new FluidParicles(simulator);
    this.textureCopy = new TextureCopy();
    this.textureFilter = new TextureFilter();
    this.postprocess = new Postprocess();
    
  }


  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.renderDepthMap = globalResource.renderDepthMap as GPUTexture;
    this.resourceAttributes = [ 'fluidDepthMap', 'fluidVolumeMap', 'fluidDepthStorageMap' ];
    this.resource = await resourceFactory.createResource(this.resourceAttributes, { });

    // point sprite render
    this.fluidParticles.initVertexBuffer();
    await this.fluidParticles.initGroupResource(globalResource);
    await this.fluidParticles.initPipeline();
    this.fluidParticles.initRenderBundle();

    // texture copy
    this.textureCopy.setTexture(
      this.resource.fluidDepthMap as GPUTexture,
      this.resource.fluidDepthStorageMap as GPUTexture,
      [canvasSize.width, canvasSize.height] // texture size
    );
    await this.textureCopy.initResource();

    // texture filter
    this.textureFilter.setTexture(
      this.resource.fluidDepthStorageMap as GPUTexture,
      [canvasSize.width, canvasSize.height] // texture size
    );
    this.textureFilter.initBindGroup();
    await this.textureFilter.initPipeline();

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

    // texture copy
    this.textureCopy.execute( commandEncoder );

    // texture filter
    this.textureFilter.execute( commandEncoder );

    // screen space rendering
    this.postprocess.render( commandEncoder, ctxTextureView );

  }

}

export { ParticleFluid }