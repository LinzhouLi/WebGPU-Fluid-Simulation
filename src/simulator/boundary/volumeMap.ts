import * as THREE from 'three';
import { device } from '../../controller';

class BoundaryModel {

  private filePath: string;

  public resolution: number[];
  public sdf: GPUBuffer;
  public volumeMap: GPUBuffer;

  constructor(filePath: string) {

    this.filePath = filePath;

  }

  async initResource() {

    const loader = new THREE.FileLoader();
    const data = await loader.loadAsync(this.filePath) as string;
    const data_split = data.split('\n');

    this.resolution = data_split[0].split(' ').map(parseFloat);

    const sdf_data = new Float32Array( data_split[2].split(' ').map(parseFloat).slice(0, -1) );

    const volumeMap_data = new Float32Array( data_split[4].split(' ').map(parseFloat).slice(0, -1) );
    
    this.sdf = device.createBuffer({
      size: sdf_data.length * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer( this.sdf, 0, sdf_data, 0 );

    this.volumeMap = device.createBuffer({
      size: volumeMap_data.length * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer( this.volumeMap, 0, volumeMap_data, 0 );

  }

}

export { BoundaryModel }