import * as THREE from 'three';
import type { ResourceType } from '../../common/resourceFactory';
import { canvasSize } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { ParicleRasterizer } from './paricleRasterizer';
import { TextureFilter } from './textureFilter';
import { ScreenSpaceRenderer } from './ScreenSpaceRenderer';

class FilteredParticleFluid {

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats({
  
      fluidDepthMap: {
        type: 'texture' as ResourceType,
          label: 'Fluid Depth Storage Map for Filtering',
          visibility: GPUShaderStage.FRAGMENT,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
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

  protected camera: THREE.PerspectiveCamera;

  protected paricleRasterizer: ParicleRasterizer;
  protected textureFilter: TextureFilter;
  protected screenSpaceRenderer: ScreenSpaceRenderer;

  protected resourceAttributes: string[]; // resource name
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU
  protected renderDepthMap: GPUTexture;

  protected particleRenderBundle: GPURenderBundle;
  protected postprocessBundle: GPURenderBundle;

  constructor(
    simulator: LagrangianSimulator,
    camera: THREE.PerspectiveCamera
  ) {

    this.camera = camera;
    this.paricleRasterizer = new ParicleRasterizer(simulator);
    this.textureFilter = new TextureFilter();
    this.screenSpaceRenderer = new ScreenSpaceRenderer();
    
  }


  public async initResource(
    globalResource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.renderDepthMap = globalResource.renderDepthMap as GPUTexture;
    this.resourceAttributes = [ 'fluidDepthMap', 'fluidVolumeMap' ];
    this.resource = await resourceFactory.createResource(this.resourceAttributes, { });

    // particle rasterizer
    await this.paricleRasterizer.initResource(globalResource);

    // texture filter
    await this.textureFilter.initResource(
      this.resource.fluidDepthMap as GPUTexture,
      [canvasSize.width, canvasSize.height] // texture size
    );

    // post process render
    await this.screenSpaceRenderer.initResource({ ...globalResource, ...this.resource });

  }

  public render(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    this.textureFilter.setFilterSize(10);

    // render point sprite
    this.paricleRasterizer.execute(
      commandEncoder, 
      this.resource.fluidDepthMap as GPUTexture, 
      this.resource.fluidVolumeMap as GPUTexture
    );

    // texture filter
    this.textureFilter.execute( commandEncoder );

    // screen space rendering
    this.screenSpaceRenderer.execute( commandEncoder, ctxTextureView );

  }

}

export { FilteredParticleFluid }