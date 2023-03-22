import { device } from '../../controller';
import { ScanShaderCode, CopyShaderCode, GathreShaderCode } from './scanShader';

const THREAD_COUNT = 256;

class ExclusiveScan {

  public static ARRAY_ALIGNMENT = THREAD_COUNT * 2;
  public static MAX_ARRAY_LENGTH = ExclusiveScan.ARRAY_ALIGNMENT * ExclusiveScan.ARRAY_ALIGNMENT;

  private srcArrayBuffer: GPUBuffer;
  private destArrayBuffer: GPUBuffer;
  private tempSrcArrayBuffer: GPUBuffer;
  private tempDestArrayBuffer: GPUBuffer;
  private arrayLength: number;

  private scan1BindGroup: GPUBindGroup;
  private scan2BindGroup: GPUBindGroup;

  private static scanPipeline: GPUComputePipeline;
  private static copyPipeline: GPUComputePipeline;
  private static gatherPipeline: GPUComputePipeline;

  // private debugBuffer: GPUBuffer;

  constructor(
    srcArrayBuffer: GPUBuffer,
    destArrayBuffer: GPUBuffer,
    arrayLength: number
  ) {

    this.srcArrayBuffer = srcArrayBuffer;
    this.destArrayBuffer = destArrayBuffer;
    this.arrayLength = arrayLength;

    if (this.arrayLength % ExclusiveScan.ARRAY_ALIGNMENT != 0)
      throw new Error(`ExclusiveScan Array Length should be a power of ${ExclusiveScan.ARRAY_ALIGNMENT}!`);
    if (this.arrayLength > ExclusiveScan.MAX_ARRAY_LENGTH)
      throw new Error(`ExclusiveScan Array Length should less than ${ExclusiveScan.MAX_ARRAY_LENGTH}!`);

  }

  public async initResource() {

    // temp array buffer
    const tempBufferDesp = {
      size: ExclusiveScan.ARRAY_ALIGNMENT * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    } as GPUBufferDescriptor;
    this.tempSrcArrayBuffer = device.createBuffer(tempBufferDesp);
    this.tempDestArrayBuffer = device.createBuffer(tempBufferDesp);

    // this.debugBuffer = device.createBuffer({
    //   size: this.arrayLength * Uint32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    // });

    // bind group
    const bindGroupLayout = device.createBindGroupLayout({
      label: '123',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, 
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });
    this.scan1BindGroup = device.createBindGroup({
      label: 'scan 1',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.srcArrayBuffer } },
        { binding: 1, resource: { buffer: this.destArrayBuffer } },
      ]
    });
    this.scan2BindGroup = device.createBindGroup({
      label: 'scan 2',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.tempSrcArrayBuffer } },
        { binding: 1, resource: { buffer: this.tempDestArrayBuffer } }
      ]
    });

    // pipeline
    if (!ExclusiveScan.scanPipeline) {
      ExclusiveScan.scanPipeline = await device.createComputePipelineAsync({
        label: 'Scan Pipeline (ExclusiveScan)',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: {
          module: device.createShaderModule({ code: ScanShaderCode }),
          entryPoint: 'main'
        }
      });
    }

    if (!ExclusiveScan.copyPipeline) {
      ExclusiveScan.copyPipeline = await device.createComputePipelineAsync({
        label: 'Copy Pipeline (ExclusiveScan)',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout, bindGroupLayout] }),
        compute: {
          module: device.createShaderModule({ code: CopyShaderCode }),
          entryPoint: 'main'
        }
      });
    }

    if (!ExclusiveScan.gatherPipeline) {
      ExclusiveScan.gatherPipeline = await device.createComputePipelineAsync({
        label: 'Gather Pipeline (ExclusiveScan)',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout, bindGroupLayout] }),
        compute: {
          module: device.createShaderModule({ code: GathreShaderCode }),
          entryPoint: 'main'
        }
      });
    }

  }

  public execute(passEncoder: GPUComputePassEncoder) {

    passEncoder.setBindGroup(0, this.scan1BindGroup);
    passEncoder.setBindGroup(1, this.scan2BindGroup);

    // scan source array block by block (first scan)
    passEncoder.setPipeline(ExclusiveScan.scanPipeline);
    const blockCount = Math.ceil(this.arrayLength / ExclusiveScan.ARRAY_ALIGNMENT);
    passEncoder.dispatchWorkgroups( blockCount );

    // copy block result
    passEncoder.setPipeline(ExclusiveScan.copyPipeline);
    passEncoder.dispatchWorkgroups( Math.ceil(blockCount / THREAD_COUNT) );

    passEncoder.setBindGroup(0, this.scan2BindGroup);
    passEncoder.setBindGroup(1, this.scan1BindGroup);

    // scan block result in one block (second scan)
    passEncoder.setPipeline(ExclusiveScan.scanPipeline);
    passEncoder.dispatchWorkgroups(1);

    // gather result
    passEncoder.setPipeline(ExclusiveScan.gatherPipeline);
    passEncoder.dispatchWorkgroups( Math.ceil(this.arrayLength / THREAD_COUNT) );

  }

  public async debug() {

    // const ce = device.createCommandEncoder();
    // ce.copyBufferToBuffer(
    //   this.destArrayBuffer, 0,
    //   this.debugBuffer, 0,
    //   this.arrayLength * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce.finish() ]);
    // await this.debugBuffer.mapAsync(GPUMapMode.READ);
    // const buffer = this.debugBuffer.getMappedRange(0, this.arrayLength * Uint32Array.BYTES_PER_ELEMENT);
    // const array = new Uint32Array(buffer);
    // console.log(array);

  }

}

export { ExclusiveScan };