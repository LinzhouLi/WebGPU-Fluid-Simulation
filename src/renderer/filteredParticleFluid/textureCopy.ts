import { device } from '../../controller';

const computeShader = /* wgsl */`

override width: u32;
override height: u32;

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var destTexture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) global_index : vec3<u32>) {

  if (global_index.x >= width || global_index.y >= height) { return; }
  let val = textureLoad(srcTexture, global_index.xy, 0);
  textureStore(destTexture, global_index.xy, val);
  return;

}

`;

class TextureCopy {

  computeShaderCode: string;

  srcTexture: GPUTexture;
  destTexture: GPUTexture;
  size: number[];

  bindGroup: GPUBindGroup;
  pipeline: GPUComputePipeline;

  constructor( ) {

    this.computeShaderCode = computeShader;
    
  }

  public setTexture(
    srcTexture: GPUTexture,
    destTexture: GPUTexture,
    textureSize: number[]
  ) {

    this.srcTexture = srcTexture;
    this.destTexture = destTexture;
    this.size = textureSize;

  }

  public async initResource() {

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0, visibility: GPUShaderStage.COMPUTE, 
        texture: { sampleType: 'unfilterable-float' }
      }, {
        binding: 1, visibility: GPUShaderStage.COMPUTE, 
        storageTexture: { format: 'r32float', access: 'write-only' }
      }]
    });

    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: this.srcTexture.createView() },
        { binding: 1, resource: this.destTexture.createView() },
      ]
    });

    this.pipeline = await device.createComputePipelineAsync({
      label: 'Texture Copy Compute Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: this.computeShaderCode }),
        entryPoint: 'main',
        constants: {
          width: this.size[0],
          height: this.size[1]
        }
      }
    })

  }

  public execute( commandEncoder: GPUCommandEncoder ) {

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.dispatchWorkgroups( this.size[0] / 4, this.size[1] / 4 );
    passEncoder.end();

  }

}

export { TextureCopy }