import { device } from '../controller';


const VertexBufferFormat = {
  index: {
    label: 'Index Buffer',
    cpuFormat: '[object Uint16Array]',
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  },
  position: {
    label: 'Position Vertex Buffer',
    cpuFormat: '[object Float32Array]',
    gpuFormat: 'float32x3' as GPUVertexFormat,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    stride: 3 * 4
  },
  normal: {
    label: 'Normal Vertex Buffer',
    cpuFormat: '[object Float32Array]',
    gpuFormat: 'float32x3' as GPUVertexFormat,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    stride: 3 * 4
  },
  color: {
    label: 'Color Vertex Buffer',
    cpuFormat: '[object Float32Array]',
    gpuFormat: 'float32x4' as GPUVertexFormat,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    stride: 4 * 4
  },
  uv: {
    label: 'UV Vertex Buffer',
    cpuFormat: '[object Float32Array]',
    gpuFormat: 'float32x2' as GPUVertexFormat,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    stride: 2 * 4
  },
};



type TypedArray = Float64Array | Float32Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;

class VertexBufferFactory {

  constructor() {

  }

  createLayout(attributes: string[]) {

    let layouts: GPUVertexBufferLayout[] = [];

    let location = 0;
    for (let attribute of attributes) {

      if (attribute === 'index') continue;

      if (!VertexBufferFormat[attribute])
        throw new Error(`Vertex Buffer Attribute Not Exist: ${attribute}`);

      layouts.push({ // GPUVertexBufferLayout
        arrayStride: VertexBufferFormat[attribute].stride,
        attributes: [{
          shaderLocation: location,
          offset: 0,
          format: VertexBufferFormat[attribute].gpuFormat
        }]
      });
      location++;

    }

    return layouts;

  }

  createResource(attributes: string[], data: { [x: string]: TypedArray }) {

    let result: { [x: string]: GPUBuffer } = {  };

    for (const attribute of attributes) {

      if (!VertexBufferFormat[attribute])
        throw new Error(`Vertex Buffer Attribute Not Exist: ${attribute}`);
      if (!data[attribute])
        throw new Error(`${attribute} Needs Copy Data`);
      if (Object.prototype.toString.call(data[attribute]) != VertexBufferFormat[attribute].cpuFormat)
        throw new Error(`Invalide Type of Vertex Buffer Attribute '${attribute}'. Should Be ${VertexBufferFormat[attribute].cpuFormat}, But Got ${Object.prototype.toString.call(data[attribute])}.`)

      if (data[attribute].byteLength % 4 != 0) {
        // Number of bytes to write must be a multiple of 4
        data[attribute] = new Uint16Array([...data[attribute], 0]);
      }
      const buffer = device.createBuffer({
        label: VertexBufferFormat[attribute].label,
        usage: VertexBufferFormat[attribute].usage,
        size: data[attribute].byteLength
      });
      device.queue.writeBuffer(buffer, 0, data[attribute]);
      result[attribute] = buffer;

    }

    return result;

  }

}

export { VertexBufferFactory };