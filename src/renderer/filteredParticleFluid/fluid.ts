import * as THREE from 'three';
import type { ResourceType } from '../../common/resourceFactory';
import { canvasSize, device, timeStampQuerySet } from '../../controller';
import { resourceFactory } from '../../common/base';
import { ResourceFactory } from '../../common/resourceFactory';
import { SPH } from '../../simulator/SPH';
import { ParicleRasterizer } from './paricleRasterizer';
import { TextureFilter } from './textureFilter';
import { ScreenSpaceRenderer } from './screenSpaceRenderer';

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

      renderingOptions: {
        type: 'buffer' as ResourceType,
        label: 'Rendering Options', 
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        layout: {
          type: 'uniform' as GPUBufferBindingType
        } as GPUBufferBindingLayout
      }
  
    });
  }

  protected camera: THREE.PerspectiveCamera;
  protected filter: boolean;

  protected paricleRasterizer: ParicleRasterizer;
  protected textureFilter: TextureFilter;
  protected screenSpaceRenderer: ScreenSpaceRenderer;

  protected resourceAttributes: string[]; // resource name
  protected resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU
  protected renderDepthMap: GPUTexture;
  protected optionsArray: ArrayBuffer;
  protected optionsBufferView: DataView;

  protected particleRenderBundle: GPURenderBundle;
  protected postprocessBundle: GPURenderBundle;

  constructor(
    simulator: SPH,
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

    this.optionsArray = new ArrayBuffer(8 * Float32Array.BYTES_PER_ELEMENT);
    this.optionsBufferView = new DataView(this.optionsArray);
    this.renderDepthMap = globalResource.renderDepthMap as GPUTexture;
    this.resourceAttributes = [ 'fluidDepthMap', 'fluidVolumeMap', 'renderingOptions' ];
    this.resource = await resourceFactory.createResource(
      this.resourceAttributes, 
      { renderingOptions: { value: this.optionsArray } }
    );

    // particle rasterizer
    await this.paricleRasterizer.initResource({ ...globalResource, ...this.resource });

    // texture filter
    await this.textureFilter.initResource(
      this.resource.renderingOptions as GPUBuffer,
      this.resource.fluidDepthMap as GPUTexture,
      [canvasSize.width, canvasSize.height] // texture size
    );

    // post process render
    await this.screenSpaceRenderer.initResource({ ...globalResource, ...this.resource });

  }

  public setConfig(config: {
    mode: number,
    filterSize: number,
    particleRadius: number,
    particleTickness: number,
    tintColor: { r: number, g: number, b: number }
  }) {

    this.filter = config.filterSize < 2 ? false : true;
    this.optionsBufferView.setUint32(0, config.mode, true);
    this.optionsBufferView.setUint32(4, config.filterSize, true);
    this.optionsBufferView.setFloat32(8, config.particleRadius, true);
    this.optionsBufferView.setFloat32(12, config.particleTickness, true);
    this.optionsBufferView.setFloat32(16, config.tintColor.r / 255.0, true);
    this.optionsBufferView.setFloat32(20, config.tintColor.g / 255.0, true);
    this.optionsBufferView.setFloat32(24, config.tintColor.b / 255.0, true);
    device.queue.writeBuffer(this.resource.renderingOptions as GPUBuffer, 0, this.optionsArray, 0 );

  }

  public render(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    // render point sprite
    this.paricleRasterizer.execute(
      commandEncoder, 
      this.resource.fluidDepthMap as GPUTexture, 
      this.resource.fluidVolumeMap as GPUTexture
    );

    // texture filter
    if (this.filter) this.textureFilter.execute( commandEncoder );

    // screen space rendering
    this.screenSpaceRenderer.execute( commandEncoder, ctxTextureView );

  }

  public renderTimestamp(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    // render point sprite
    this.paricleRasterizer.execute(
      commandEncoder, 
      this.resource.fluidDepthMap as GPUTexture, 
      this.resource.fluidVolumeMap as GPUTexture
    );

    commandEncoder.writeTimestamp(timeStampQuerySet, 6);

    // texture filter
    if (this.filter) this.textureFilter.execute( commandEncoder );

    commandEncoder.writeTimestamp(timeStampQuerySet, 7);

    // screen space rendering
    this.screenSpaceRenderer.execute( commandEncoder, ctxTextureView );

    commandEncoder.writeTimestamp(timeStampQuerySet, 8);

  }

  public optionsChange(msg) {
    this.setConfig(msg.object);
  }

}

export { FilteredParticleFluid }