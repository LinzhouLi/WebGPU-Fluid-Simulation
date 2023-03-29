import { device } from '../../controller';
import { computeShader } from './shader/filterPassShader';

class TextureFilter {

  protected computeShaderCode: string;

  protected textureSize: number[];
  protected filterTexture: GPUTexture;
  protected tempTexture: GPUTexture;

  protected filterSize: Int32Array;
  protected filterSizeBuffer: GPUBuffer;

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroupX: GPUBindGroup;
  protected bindGroupY: GPUBindGroup;
  protected pipeline: GPUComputePipeline;

  constructor() {

    this.computeShaderCode = computeShader;

  }

  public setTexture(
    filterTexture: GPUTexture,
    textureSize: number[]
  ) {

    this.filterTexture = filterTexture;
    this.textureSize = textureSize;

    this.tempTexture = device.createTexture({
      size: this.textureSize,
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });

    this.filterSize = new Int32Array(1);
    this.filterSizeBuffer = device.createBuffer({
      size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
    })

  }

  private createFilterDirBuffer() {

    const filterDirBufferDescriptor = {
      size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
    };

    const filterDirBufferX = device.createBuffer(filterDirBufferDescriptor);
    const filterDirBufferY = device.createBuffer(filterDirBufferDescriptor);

    const filterDirDataX = new Int32Array(2);
    filterDirDataX.set([1, 0]);
    device.queue.writeBuffer(
      filterDirBufferX, 0,
      filterDirDataX, 0
    );

    const filterDirDataY = new Int32Array(2);
    filterDirDataY.set([0, 1]);
    device.queue.writeBuffer(
      filterDirBufferY, 0,
      filterDirDataY, 0
    );

    return { filterDirBufferX, filterDirBufferY };

  }

  public initBindGroup() {

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'Filter Pipeline Bind Group Layout',
      entries: [{ // filter src map
        binding: 0, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' }
      }, { // filter dest map
        binding: 1, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: 'r32float', access: 'write-only' }
      }, { // filter direction
        binding: 2, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }, { // filter size
        binding: 3, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
      }]
    });

    const filterTextureView = this.filterTexture.createView();
    const tempTextureView = this.tempTexture.createView();
    const { filterDirBufferX, filterDirBufferY } = this.createFilterDirBuffer();

    this.bindGroupX = device.createBindGroup({
      label: 'Filter Pipeline Bind Group (X-axis)',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: filterTextureView }, 
        { binding: 1, resource: tempTextureView },
        { binding: 2, resource: { buffer: filterDirBufferX } },
        { binding: 3, resource: { buffer: this.filterSizeBuffer } }
      ]
    });

    this.bindGroupY = device.createBindGroup({
      label: 'Filter Pipeline Bind Group (Y-axis)',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: tempTextureView }, 
        { binding: 1, resource: filterTextureView },
        { binding: 2, resource: { buffer: filterDirBufferY } },
        { binding: 3, resource: { buffer: this.filterSizeBuffer } }
      ]
    });

  }

  public async initPipeline() {

    this.pipeline = await device.createComputePipelineAsync({
      label: 'Filter Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: this.computeShaderCode }),
        entryPoint: 'main',
        // constants: {
        //   width: this.textureSize[0],
        //   height: this.textureSize[1]
        // }
      }
    });

  }

  public setFilterSize(filterSize: number) {

    this.filterSize.set([filterSize]);
    device.queue.writeBuffer(
      this.filterSizeBuffer, 0, 
      this.filterSize, 0
    );

  }

  public execute( commandEncoder: GPUCommandEncoder ) {

    // clear temp texture
    const renderPassEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.tempTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    renderPassEncoder.end();

    // filter
    const computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(this.pipeline);

    // x-axis
    computePassEncoder.setBindGroup(0, this.bindGroupX);
    computePassEncoder.dispatchWorkgroups(
      Math.ceil(this.textureSize[0] / 32), 
      Math.ceil(this.textureSize[1] / 1)
    );

    // y-axis
    computePassEncoder.setBindGroup(0, this.bindGroupY);
    computePassEncoder.dispatchWorkgroups(
      Math.ceil(this.textureSize[1] / 32), 
      Math.ceil(this.textureSize[0] / 1)
    );

    computePassEncoder.end();

  }

}

export { TextureFilter };