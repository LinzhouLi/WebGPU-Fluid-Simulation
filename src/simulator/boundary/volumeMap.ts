import * as THREE from 'three';
import { device } from '../../controller';
import { DebugShader } from './discreteFieldShader'

class BoundaryModel {

  private filePath: string;

  public resolution: number[];
  public field: GPUBuffer;
  public fieldByteLength: number;

  constructor(filePath: string) {

    this.filePath = filePath;

  }

  public async initResource() {

    const loader = new THREE.FileLoader();
    const data = await loader.loadAsync(this.filePath) as string;
    const data_split = data.split('\n');

    this.resolution = data_split[0].split(' ').map(parseFloat);

    const sdf_data = new Float32Array( data_split[2].split(' ').map(parseFloat).slice(0, -1) );
    const volumeMap_data = new Float32Array( data_split[4].split(' ').map(parseFloat).slice(0, -1) );

    if (sdf_data.length != volumeMap_data.length) {
      throw new Error('Invalid Boundary Model File!');
    }
    this.fieldByteLength = sdf_data.length * Float32Array.BYTES_PER_ELEMENT;
    
    this.field = device.createBuffer({
      size: 2 * this.fieldByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer( this.field, 0, sdf_data, 0 );
    device.queue.writeBuffer( this.field, this.fieldByteLength, volumeMap_data, 0 );

  }

  public async debug() {

    // const field_size = [1, 1, 1];
    // const field_data_size = (
    //   (field_size[0] + 1) * (field_size[1] + 1) * (field_size[2] + 1) + 
    //   (field_size[0] * 2) * (field_size[1] + 1) * (field_size[2] + 1) + 
    //   (field_size[0] + 1) * (field_size[1] * 2) * (field_size[2] + 1) + 
    //   (field_size[0] + 1) * (field_size[1] + 1) * (field_size[2] * 2)
    // );
    const result_data_size = 4 * 8;
    // console.log(field_data_size)

    // const field_data = new Float32Array(field_data_size);
    // field_data.forEach((_, i) => field_data[i] = i);
    // console.log(field_data)

    // const field_data_buffer = device.createBuffer({
    //   size: field_data_size * Float32Array.BYTES_PER_ELEMENT,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    const result_buffer = device.createBuffer({
      size: result_data_size * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
    });
    const map_buffer = device.createBuffer({
      size: result_data_size * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // device.queue.writeBuffer(field_data_buffer, 0, field_data, 0);

    const bindgroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, buffer: { type: 'storage' }, visibility: GPUShaderStage.COMPUTE },
        { binding: 1, buffer: { type: 'storage' }, visibility: GPUShaderStage.COMPUTE }
      ]
    });

    const bindgroup = device.createBindGroup({
      layout: bindgroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.field } },
        { binding: 1, resource: { buffer: result_buffer } }
      ]
    });

    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [ bindgroupLayout ] }),
      compute: {
        module: device.createShaderModule({ code: DebugShader }),
        entryPoint: 'main'
      }
    });

    const ce = device.createCommandEncoder();

    const pe = ce.beginComputePass();
    pe.setBindGroup(0, bindgroup);
    pe.setPipeline(pipeline);
    pe.dispatchWorkgroups(1);
    pe.end();

    ce.copyBufferToBuffer(
      result_buffer, 0, 
      map_buffer, 0, 
      result_data_size * Float32Array.BYTES_PER_ELEMENT
    );

    device.queue.submit([ ce.finish() ]);

    await map_buffer.mapAsync(GPUMapMode.READ);
    const buffer = map_buffer.getMappedRange(0, result_data_size * Float32Array.BYTES_PER_ELEMENT);
    const array = new Float32Array(buffer);
    console.log(array);

  }

}

export { BoundaryModel }