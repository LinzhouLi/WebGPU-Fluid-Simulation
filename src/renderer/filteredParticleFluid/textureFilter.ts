import { device } from '../../controller';
import { computeShader } from './shader/filterPassShader';

class TextureFilter {

  protected computeShaderCode: string;

  protected textureSize: number[];
  protected filterTexture: GPUTexture;
  protected tempTexture: GPUTexture;

  protected filterSize: Int32Array;
  protected filterSizeBuffer: GPUBuffer;
  protected filterDirBuffer: {
    X: GPUBuffer;
    Y: GPUBuffer;
  }

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroupX: GPUBindGroup;
  protected bindGroupY: GPUBindGroup;
  protected pipeline: GPUComputePipeline;

  constructor() {

    this.computeShaderCode = computeShader;

  }

  public async initResource(
    filterTexture: GPUTexture,
    textureSize: number[]
  ) {

    this.filterTexture = filterTexture;
    this.textureSize = textureSize;
    
    this.initGroupResource();
    this.initBindGroup();
    await this.initPipeline();

  }

  protected initGroupResource() {

    // temp texture for storing intermediate filtering result
    this.tempTexture = device.createTexture({
      size: this.textureSize,
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });

    // filter size buffer
    this.filterSize = new Int32Array(1);
    this.filterSizeBuffer = device.createBuffer({
      size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
    });

    // filter direction buffer [1, 0] & [0, 1]
    const filterDirBufferDescriptor = {
      size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
    };
    this.filterDirBuffer = {
      X: device.createBuffer(filterDirBufferDescriptor),
      Y: device.createBuffer(filterDirBufferDescriptor)
    };

    const filterDirData = new Int32Array(2);
    filterDirData.set([1, 0]);
    device.queue.writeBuffer( this.filterDirBuffer.X, 0, filterDirData, 0 );
    filterDirData.set([0, 1]);
    device.queue.writeBuffer( this.filterDirBuffer.Y, 0, filterDirData, 0 );

  }

  protected initBindGroup() {

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

    this.bindGroupX = device.createBindGroup({
      label: 'Filter Pipeline Bind Group (X-axis)',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: filterTextureView }, 
        { binding: 1, resource: tempTextureView },
        { binding: 2, resource: { buffer: this.filterDirBuffer.X } },
        { binding: 3, resource: { buffer: this.filterSizeBuffer } }
      ]
    });

    this.bindGroupY = device.createBindGroup({
      label: 'Filter Pipeline Bind Group (Y-axis)',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: tempTextureView }, 
        { binding: 1, resource: filterTextureView },
        { binding: 2, resource: { buffer: this.filterDirBuffer.Y } },
        { binding: 3, resource: { buffer: this.filterSizeBuffer } }
      ]
    });

  }

  protected async initPipeline() {

    this.pipeline = await device.createComputePipelineAsync({
      label: 'Filter Pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: this.computeShaderCode }),
        entryPoint: 'main',
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