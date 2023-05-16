import { device } from '../../controller';
import { SPH } from '../SPH';
import { PBFConfig } from './PBFConfig';
import { NeighborSearch } from '../neighbor/neighborSearch';
import { BoundaryModel } from '../boundary/volumeMap';
import { loader } from '../../common/loader';

// shaders
import { TimeIntegrationShader } from './shader/timeIntegration';
import { VorticityConfinementShader } from './shader/vorticityConfinement';
import { XSPHShader } from './shader/XSPH';
import { BoundaryVolumeShader } from './shader/boundaryVolume';
import { LambdaCalculationShader, ConstrainSolveShader, ConstrainApplyShader } from './shader/pressureConstrain';
import { AttributeUpdateShader } from './shader/attributeUpdate';


function kernalPoly6(r_len: number) {
  if (r_len <= PBF.KERNEL_RADIUS) {
    const coef = 315.0 / (64.0 * Math.PI * Math.pow(PBF.KERNEL_RADIUS, 9));
    const x = PBF.KERNEL_RADIUS * PBF.KERNEL_RADIUS - r_len * r_len;
    return coef * Math.pow(x, 3);
  }
  else {
    return 0;
  }
}

class PBF extends PBFConfig {

  private position2: GPUBuffer;
  private lambda: GPUBuffer;
  private boundaryData: GPUBuffer;
  private normal: GPUBuffer;

  // reused buffers
  private deltaPosition: GPUBuffer;
  private angularVelocity: GPUBuffer;

  private configBindGroupLayout: GPUBindGroupLayout;
  private neighborBindGroupLayout: GPUBindGroupLayout;
  private integrationBindGroupLayout: GPUBindGroupLayout;
  private constrainBindGroupLayout: GPUBindGroupLayout;
  private nonPressureBindGroupLayout: GPUBindGroupLayout;

  private configBindGroup: GPUBindGroup;
  private neighborBindGroup: GPUBindGroup;
  private integrationBindGroup: GPUBindGroup;
  private constrainBindGroup: GPUBindGroup;
  private nonPressureBindGroup: GPUBindGroup;

  private forceApplyPipeline: GPUComputePipeline;
  private boundaryVolumePipeline: GPUComputePipeline;
  private lambdaCalculationPipeline: GPUComputePipeline;
  private constrainSolvePipeline: GPUComputePipeline;
  private constrainApplyPipeline: GPUComputePipeline;
  private attributeUpdatePipeline: GPUComputePipeline;
  private vortcityConfinementPipeline: GPUComputePipeline;
  private XSPHPipeline: GPUComputePipeline;

  private boundaryModel: BoundaryModel;
  private neighborSearch: NeighborSearch;

  private static debug = true;
  private tempBuffer: GPUBuffer;
  private debugBuffer1: GPUBuffer;
  private debugBuffer2: GPUBuffer;

  constructor() {
    super();
  }

  public reset() {

    const ce = device.createCommandEncoder();

    this.baseReset(ce);

    ce.clearBuffer(this.position2);
    ce.clearBuffer(this.lambda);
    ce.clearBuffer(this.boundaryData);
    ce.clearBuffer(this.normal);

    this.neighborSearch.reset(ce);
    this.boundaryModel.reset(ce);

    device.queue.submit([ ce.finish() ]);

    this.ifBoundary = false;
    
  }

  public setBoundaryData(data: string) {
    this.boundaryModel.setData(data);
    this.ifBoundary = true;
  }

  public computeParticleDensity(particleIndex: number) {

    let density = 0;
    let pos = [
      this.particlePositionArray[4*particleIndex], 
      this.particlePositionArray[4*particleIndex + 1], 
      this.particlePositionArray[4*particleIndex + 2]
    ];
    for (let i = 0; i < this.particleCount; i++) {
      let npos = [
        this.particlePositionArray[4 * i] - pos[0], 
        this.particlePositionArray[4 * i + 1] - pos[1], 
        this.particlePositionArray[4 * i + 2] - pos[2]
      ];
      let len = Math.sqrt(npos[0]*npos[0] + npos[1]*npos[1] + npos[2]*npos[2]);
      density += kernalPoly6(len);
    }
    density *= this.particleWeight;
    return density;

  }

  private createStorageData() {

    // create GPU Buffers
    // vec3/vec4 particle attribute buffer
    let attributeBufferDesp = {
      size: 4 * SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    } as GPUBufferDescriptor;
    this.position2 = device.createBuffer(attributeBufferDesp);
    this.boundaryData = device.createBuffer(attributeBufferDesp);
    this.normal = device.createBuffer(attributeBufferDesp);

    this.deltaPosition = this.velocity;
    this.angularVelocity = this.position2;

    // f32 particle attribute buffer
    attributeBufferDesp = {
      size: SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    };
    this.lambda = device.createBuffer(attributeBufferDesp);

    if (PBF.debug) {
      this.tempBuffer = device.createBuffer({
        size: SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      this.debugBuffer1 = device.createBuffer({
        size: 4 * SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      this.debugBuffer2 = device.createBuffer({
        size: 4 * SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
    }

  }

  private createBindGroupLayout() {

    // config
    this.configBindGroupLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }]
    });

    // neighbor
    this.neighborBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
      ]
    });

    // time integration
    this.integrationBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    // pressure constrain
    this.constrainBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
      ]
    });

    // non pressure force
    this.nonPressureBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

  }

  private createBindGroup() {

    // config
    this.configBindGroup = device.createBindGroup({
      layout: this.configBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.optionsBuffer } }]
    })

    // neighbor
    this.neighborBindGroup = device.createBindGroup({
      layout: this.neighborBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.neighborSearch.neighborOffset } },
        { binding: 1, resource: { buffer: this.neighborSearch.neighborList } }
      ]
    });

    // time integration
    this.integrationBindGroup = device.createBindGroup({
      layout: this.integrationBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.position } },
        { binding: 1, resource: { buffer: this.position2 } },
        { binding: 2, resource: { buffer: this.velocity } },
        { binding: 3, resource: { buffer: this.acceleration } }
      ]
    });

    // pressure constrain
    this.constrainBindGroup = device.createBindGroup({
      layout: this.constrainBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.position2 } },
        { binding: 1, resource: { buffer: this.deltaPosition } },
        { binding: 2, resource: { buffer: this.lambda } },
        { binding: 3, resource: { buffer: this.boundaryData } },
        { binding: 4, resource: { buffer: this.boundaryModel.field } }
      ]
    });

    // non pressure force
    this.nonPressureBindGroup = device.createBindGroup({
      layout: this.nonPressureBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.position } },
        { binding: 1, resource: { buffer: this.angularVelocity } }, // position2
        { binding: 2, resource: { buffer: this.velocity } },
        { binding: 3, resource: { buffer: this.acceleration } },
        { binding: 4, resource: { buffer: this.normal } },
        { binding: 5, resource: { buffer: this.boundaryData } }
      ]
    });

  }

  public async initResource() {

    this.createBindGroupLayout();
    this.createBaseStorageData();
    this.createStorageData();

    // boundary model
    this.boundaryModel = new BoundaryModel();
    await this.boundaryModel.initResource();

    // neighbor search
    this.neighborSearch = new NeighborSearch(this);
    await this.neighborSearch.initResource(this.configBindGroupLayout, this.position2);

    this.createBindGroup();
    await this.initComputePipeline();

  }

  private getScorrCoefficient() {

    let poly6 = kernalPoly6(this.scorrCoefDq * PBF.KERNEL_RADIUS);
    return -this.scorrCoefK / Math.pow(poly6, this.scorrCoefN);

  }

  public async initComputePipeline() {

    // integration
    const integrationPipelineLayout = device.createPipelineLayout({ 
      bindGroupLayouts: [this.configBindGroupLayout, this.integrationBindGroupLayout] 
    });
    this.forceApplyPipeline = await device.createComputePipelineAsync({
      label: 'Force Apply Pipeline (PBF)',
      layout: integrationPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: TimeIntegrationShader }),
        entryPoint: 'main'
      }
    });

    // constrain
    const constrainPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.configBindGroupLayout, this.neighborBindGroupLayout, this.constrainBindGroupLayout]
    });

    this.boundaryVolumePipeline = await device.createComputePipelineAsync({
      label: 'Boundary Volume Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: BoundaryVolumeShader }),
        entryPoint: 'main',
        constants: {
          ParticleRadius: this.particleRadius
        }
      }
    });

    this.lambdaCalculationPipeline = await device.createComputePipelineAsync({
      label: 'Lambda Calculation Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: LambdaCalculationShader }),
        entryPoint: 'main',
        constants: {
          ParticleVolume: this.particleVolume,
          ParticleVolume2: this.particleVolume * this.particleVolume,
          LambdaEPS: this.lambdaEPS
        }
      }
    });

    this.constrainSolvePipeline = await device.createComputePipelineAsync({
      label: 'Constrain Solve Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: ConstrainSolveShader }),
        entryPoint: 'main',
        constants: {
          ParticleVolume: this.particleVolume
        }
      }
    });

    this.constrainApplyPipeline = await device.createComputePipelineAsync({
      label: 'Constrain Apply Pipeline (PBF)',
      layout: constrainPipelineLayout,
      compute: {
        module: device.createShaderModule({ code: ConstrainApplyShader }),
        entryPoint: 'main'
      }
    });

    // non pressure force
    const nonPressurePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.configBindGroupLayout, this.neighborBindGroupLayout, this.nonPressureBindGroupLayout]
    });
    this.attributeUpdatePipeline = await device.createComputePipelineAsync({
      label: 'Attribute Update Pipeline (PBF)',
      layout: nonPressurePipelineLayout,
      compute: {
        module: device.createShaderModule({ code: AttributeUpdateShader }),
        entryPoint: 'main',
        constants: {
          ParticleWeight: this.particleWeight
        }
      }
    });

    this.vortcityConfinementPipeline = await device.createComputePipelineAsync({
      label: 'Vorticity Confinement Pipeline (PBF)',
      layout: nonPressurePipelineLayout,
      compute: {
        module: device.createShaderModule({ code: VorticityConfinementShader }),
        entryPoint: 'main',
        constants: {
          ParticleWeight: this.particleWeight
        }
      }
    });

    this.XSPHPipeline = await device.createComputePipelineAsync({
      label: 'XSPH Pipeline (PBF)',
      layout: nonPressurePipelineLayout,
      compute: {
        module: device.createShaderModule({ code: XSPHShader }),
        entryPoint: 'main',
        constants: {
          DoubleDensity0: 2 * this.restDensity,
          ParticleWeight: this.particleWeight
        }
      }
    });

  }

  public run(commandEncoder: GPUCommandEncoder) {
    
    if (this.pause) return;
    
    this.neighborSearch.clearBuffer(commandEncoder);
    
    const passEncoder = commandEncoder.beginComputePass();
    const workgroupCount = Math.ceil(this.particleCount / 256);
    
    passEncoder.setBindGroup(0, this.configBindGroup);

    passEncoder.setBindGroup(1, this.integrationBindGroup);
    passEncoder.setPipeline(this.forceApplyPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    this.neighborSearch.execute(passEncoder);

    passEncoder.setBindGroup(1, this.neighborBindGroup);
    passEncoder.setBindGroup(2, this.constrainBindGroup);
    for (let i = 0; i < this.constrainIterationCount; i++) {
      if (this.ifBoundary) {
        passEncoder.setPipeline(this.boundaryVolumePipeline);
        passEncoder.dispatchWorkgroups(workgroupCount);
      }

      passEncoder.setPipeline(this.lambdaCalculationPipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);

      passEncoder.setPipeline(this.constrainSolvePipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);

      passEncoder.setPipeline(this.constrainApplyPipeline);
      passEncoder.dispatchWorkgroups(workgroupCount);
    }

    // passEncoder.setPipeline(this.boundaryVolumePipeline);
    // passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.setBindGroup(2, this.nonPressureBindGroup);
    passEncoder.setPipeline(this.attributeUpdatePipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.setPipeline(this.vortcityConfinementPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.setPipeline(this.XSPHPipeline);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.end();
  
  }

  public update() { }

  public async debug() {

    // await this.boundaryModel.debug();

    if (PBF.debug) {
      const ce = device.createCommandEncoder();

      this.run(ce);

      ce.copyBufferToBuffer(
        this.acceleration, 0,
        this.debugBuffer1, 0,
        4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT
      );
      // ce.copyBufferToBuffer(
      //   this.tempBuffer, 0,
      //   this.debugBuffer2, 0,
      //   this.particleCount * Float32Array.BYTES_PER_ELEMENT
      // );
      device.queue.submit([ ce.finish() ]);
      
      await device.queue.onSubmittedWorkDone();

      await this.debugBuffer1.mapAsync(GPUMapMode.READ);
      const buffer1 = this.debugBuffer1.getMappedRange(0, 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT);
      const array1 = new Float32Array(buffer1);

      // console.log(array1);
      this.debugBuffer1.unmap();

      // await this.debugBuffer1.mapAsync(GPUMapMode.READ);
      // const buffer1 = this.debugBuffer1.getMappedRange(0, this.particleCount * Float32Array.BYTES_PER_ELEMENT);
      // const array1 = new Float32Array(buffer1);

      // await this.debugBuffer2.mapAsync(GPUMapMode.READ);
      // const buffer2 = this.debugBuffer2.getMappedRange(0, this.particleCount * Float32Array.BYTES_PER_ELEMENT);
      // const array2 = new Float32Array(buffer2);

      // let min = array1[0]; let max = array1[0];
      // let min_index = 0; let max_index = 0;
      // array1.forEach((val, i) => {
      //   if (val >= max) { max = val; max_index = i; }
      //   if (val <= min) { min = val; min_index = i; }
      // });
      // console.log(min_index, min);
      // console.log(max_index, max);

      // min = array2[0]; max = array2[0];
      // min_index = 0; max_index = 0;
      // array2.forEach((val, i) => {
      //   if (val >= max) { max = val; max_index = i; }
      //   if (val <= min) { min = val; min_index = i; }
      // });
      // console.log(min_index, min);
      // console.log(max_index, max);

      await this.neighborSearch.debug();
    }

  }

}

export { PBF };