import { device } from '../../controller';
import { ExclusiveScan } from './exclusiveScan';

class NeighborList {

  private particleCount: number;
  private maxNeighborCount: number;

  private particlePosition: GPUBuffer;
  private neighborList: GPUBuffer;

  private scanSourceBuffer: GPUBuffer;
  private scanResultBuffer: GPUBuffer;
  private debugBuffer: GPUBuffer;

  private scan: ExclusiveScan;

  constructor(
    // particlePosition: GPUBuffer,  particleCount: number,
    // neighborList: GPUBuffer,      maxNeighborCount: number
  ) {

    // this.particleCount = particleCount;
    // this.maxNeighborCount = maxNeighborCount;
    // this.particlePosition = particlePosition;
    // this.neighborList = neighborList;

  }

  public async initResource() {

    this.scanSourceBuffer = device.createBuffer({
      size: 512 * 512 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.scanResultBuffer = device.createBuffer({
      size: 512 * 512 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    this.debugBuffer = device.createBuffer({
      size: 512 * 512 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const scanSourceArray = new Uint32Array(512 * 512);
    scanSourceArray.fill(1);
    device.queue.writeBuffer(
      this.scanSourceBuffer, 0,
      scanSourceArray, 0
    );

    this.scan = new ExclusiveScan(this.scanSourceBuffer, this.scanResultBuffer, 512 * 512);
    await this.scan.initResource();

  }

  public execute(passEncoder: GPUComputePassEncoder) {

    this.scan.execute(passEncoder);

  }

  public async debug() {

    // const ce = device.createCommandEncoder();
    // ce.copyBufferToBuffer(
    //   this.scanResultBuffer, 0,
    //   this.debugBuffer, 0,
    //   256*256 * Uint32Array.BYTES_PER_ELEMENT
    // );
    // device.queue.submit([ ce.finish() ]);
    // await this.debugBuffer.mapAsync(GPUMapMode.READ);
    // const buffer = this.debugBuffer.getMappedRange();
    // const array = new Uint32Array(buffer);
    // console.log(array);

    await this.scan.debug()

  }

}

export { NeighborList };