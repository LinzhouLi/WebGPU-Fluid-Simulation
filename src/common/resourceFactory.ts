import { device } from '../controller';
import type { TypedArray } from './base';

type ResourceType = 'buffer' | 'sampler' | 'texture' | 'cube-texture' | 'texture-array';

interface BufferData {
  value?: TypedArray
  size?: number
};

interface TextureData {
  value?: ImageBitmap
  size?: GPUExtent3DStrict
  flipY?: boolean
};

interface TextureArrayData {
  value?: Array<ImageBitmap>
  size?: GPUExtent3DStrict
  flipY?: boolean[]
};


class ResourceFactory {

  public static Formats: { [x: string]: any } = { };
  private resources: WeakMap<
    TypedArray | ImageBitmap | object,
    GPUBuffer | GPUTexture | GPUSampler
  >;

  constructor() {

    this.resources = new WeakMap();

  }

  public static RegisterFormats(formats: { [x: string]: any }) {

    for (const key in formats) {
      if (!!ResourceFactory.Formats[key])
        throw new Error(`Resource Format ${key} has already Registered!`);
      else
        ResourceFactory.Formats[key] = formats[key];
    }

  }

  public async toBitmap(data: ImageBitmapSource | ImageBitmap) {
    if (data instanceof ImageBitmap) return data;
    else return await createImageBitmap(data);
  }

  public toBitmaps(dataArray: (ImageBitmapSource | ImageBitmap)[]) {
    let results: Promise<ImageBitmap>[] = [];
    dataArray.forEach(data => {
      results.push(this.toBitmap(data));
    });
    return Promise.all(results);
  }

  public async createResource(
    attributes: string[], 
    data: Record<string, BufferData | TextureData | TextureArrayData>
  ): Promise<Record<string, GPUBuffer | GPUTexture | GPUSampler>> {

    let result: Record<string, GPUBuffer | GPUTexture | GPUSampler> = {  };

    for (const attribute of attributes) {
      const format = ResourceFactory.Formats[attribute];
      if (!format) throw new Error(`Resource Attribute Not Exist: ${attribute}`);

      switch (format.type) {

        case 'buffer': { // GPU buffer
          const bufferData = data[attribute] as BufferData;

          let buffer = this.resources.get(bufferData?.value);
          if (buffer) result[attribute] = buffer;
          else {
            buffer = device.createBuffer({
              label: format.label,
              size: bufferData?.value?.byteLength || bufferData?.size || format.size,
              usage: format.usage
            });
            if (bufferData?.value) {
              device.queue.writeBuffer(buffer, 0, bufferData?.value);
              this.resources.set(bufferData?.value, buffer);
            }
            result[attribute] = buffer;
          }
          break;
        }

        case 'sampler': { // GPU sampler
          let sampler = this.resources.get(format);
          if (sampler) result[attribute] = sampler;
          else {
            sampler = device.createSampler({
              label: format.label,
              magFilter: format.magFilter || 'linear',
              minFilter: format.minFilter || 'linear',
              mipmapFilter: format.mipmapFilter || 'linear',
              compare: format.compare || undefined // The provided value 'null' is not a valid enum value of type GPUCompareFunction.
            });
            this.resources.set(format, sampler);
            result[attribute] = sampler;
          }
          break;
        }

        case 'texture': { // GPU texture
          const textureData = data[attribute] as TextureData;
          
          let texture = this.resources.get(textureData?.value);
          if (texture) result[attribute] = texture;
          else {
            const bitmap = textureData?.value;
            const textureSize = bitmap ? [bitmap.width, bitmap.height] : textureData?.size || format.size;
            let textureDescriptor = {
              label: format.label,
              size: textureSize,
              mipLevelCount: format.mipLevelCount || 1,
              dimension: format.dimension || '2d',
              format: format.format,
              usage: format.usage
            } as GPUTextureDescriptor;
            if (format.viewFormat) textureDescriptor.viewFormats = [format.viewFormat];
            texture = device.createTexture(textureDescriptor);
            if (bitmap) {
              device.queue.copyExternalImageToTexture(
                { source: bitmap, flipY: textureData?.flipY || false },
                { texture: texture },
                textureSize
              );
              this.resources.set(bitmap, texture);
            }
            result[attribute] = texture;
          }
          break;
        }

        case 'cube-texture': {// GPU cube texture\
          const textureArrayData = data[attribute] as TextureArrayData;

          if (textureArrayData?.value) {
            if (textureArrayData.value.length != 6)
              throw new Error('Array Length of cube-texture is Not 6');
          }
          else if (textureArrayData?.size && textureArrayData.size[2] != 6)
            throw new Error('Array Length of cube-texture is Not 6');
        }
        case 'texture-array': { // GPU texture array
          const textureArrayData = data[attribute] as TextureArrayData;

          let textureArray = this.resources.get(textureArrayData?.value);
          if (textureArray) result[attribute] = textureArray;
          else {
            const bitmaps = textureArrayData?.value;
            const textureSize = bitmaps ? 
              [bitmaps[0].width, bitmaps[0].height, bitmaps.length] : 
              textureArrayData?.size || format.size;
            let textureDescriptor = {
              label: format.label,
              size: textureSize,
              mipLevelCount: format.mipLevelCount || 1,
              dimension: format.dimension || '2d',
              format: format.format,
              usage: format.usage
            } as GPUTextureDescriptor;
            if (format.viewFormat) textureDescriptor.viewFormats = [format.viewFormat];
            textureArray = device.createTexture(textureDescriptor);
            if (bitmaps) {
              for (let i = 0; i < bitmaps.length; i++) {
                device.queue.copyExternalImageToTexture(
                  { 
                    source: bitmaps[i], 
                    flipY: textureArrayData?.flipY[i] || false 
                  }, { 
                    texture: textureArray,  // Defines the origin of the copy - the minimum corner of the texture sub-region to copy to/from.
                    origin: [0, 0, i]       // Together with `copySize`, defines the full copy sub-region.
                  },
                  [ textureSize[0], textureSize[1], 1]
                )
              }
              this.resources.set(bitmaps, textureArray);
            }
            result[attribute] = textureArray;
          }
          break;
        }

        default: {
          throw new Error('Resource Type Not Support');
        }
      }
      
    }

    return result;

  }

}

export type { ResourceType, BufferData, TextureData, TextureArrayData };
export { ResourceFactory };
