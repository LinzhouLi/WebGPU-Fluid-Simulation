import { ResourceFactory } from './resourceFactory';
import { device } from '../controller';


class BindGroupFactory {

  constructor() {

  }

  createLayout(attributes: string[], groupLabel: string = undefined) {

    let entries: GPUBindGroupLayoutEntry[] = [];

    let bindIndex = 0;
    for (const attribute of attributes) {

      if (!ResourceFactory.Formats[attribute])
        throw new Error(`Resource Attribute Not Exist: ${attribute}`);

      switch(ResourceFactory.Formats[attribute].type) {
        case 'buffer': { // GPU buffer
          entries.push({
            binding: bindIndex,
            visibility: ResourceFactory.Formats[attribute].visibility,
            buffer: ResourceFactory.Formats[attribute].layout
          });
          break;
        }
        case 'sampler': { // GPU sampler
          entries.push({
            binding: bindIndex,
            visibility: ResourceFactory.Formats[attribute].visibility,
            sampler: ResourceFactory.Formats[attribute].layout
          });
          break;
        }
        case 'texture': // GPU texture
        case 'texture-array': // GPU texture array
        case 'cube-texture': { // GPU cube texture
          entries.push({
            binding: bindIndex,
            visibility: ResourceFactory.Formats[attribute].visibility,
            texture: ResourceFactory.Formats[attribute].layout
          });
          break;
        }
        default: {
          throw new Error('Resource Type Not Support');
        }
      }

      bindIndex++;

    }
    
    return device.createBindGroupLayout({ entries, label: groupLabel });
    
  }

  create(
    attributes: string[], 
    data: { [x: string]: GPUBuffer | GPUTexture | GPUSampler },
    groupLayout: GPUBindGroupLayout | null = null,
    groupLabel: string = undefined
  ) {

    let layout: GPUBindGroupLayout;
    if (groupLayout) layout = groupLayout;
    else layout = this.createLayout(attributes, groupLabel);

    let entries: GPUBindGroupEntry[] = [];

    let bindIndex = 0;
    for (const attribute of attributes) {

      const format = ResourceFactory.Formats[attribute];

      if (!format)
        throw new Error(`Resource Attribute Not Exist: ${attribute}`);
      if (!data[attribute])
        throw new Error(`Resource '${attribute}' Not Exist`);


      switch(format.type) {
        case 'buffer': { // GPU buffer
          entries.push({
            binding: bindIndex,
            resource: { buffer: data[attribute] as GPUBuffer }
          });
          break;
        }
        case 'sampler': { // GPU sampler
          entries.push({
            binding: bindIndex,
            resource: data[attribute] as GPUSampler
          });
          break;
        }
        case 'texture': // GPU texture
        case 'texture-array': // GPU texture array
        case 'cube-texture': { // GPU cube texture
          entries.push({
            binding: bindIndex,
            resource: (data[attribute] as GPUTexture).createView({
              format: format.viewFormat || format.format,
              dimension: format.layout.viewDimension || '2d'
            })
          });
          break;
        }
        default: {
          throw new Error('Resource Type Not Support');
        }
      }
      bindIndex++;

    }

    let bindGroupDescriptor: GPUBindGroupDescriptor;
    if (groupLabel) bindGroupDescriptor = { label: groupLabel, layout, entries };
    else bindGroupDescriptor = { layout, entries };
    let group = device.createBindGroup(bindGroupDescriptor);
    
    return { layout, group };

  }

}

export { BindGroupFactory }