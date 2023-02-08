import type { ResourceType } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';
import { device } from '../controller';

abstract class BaseSimulator {

  private static ResourceFormats = {
    particlePosition: {
      type: 'buffer' as ResourceType,
      label: 'Particle Positions',
      visibility: GPUShaderStage.COMPUTE,
      layout: { 
        type: 'storage' as GPUBufferBindingType
      } as GPUBufferBindingLayout
    },
  };

  public particleCount: number;
  public stepCount: number;

  public particlePositionBuffer: GPUBuffer;

  constructor(particleCount: number = 1, stepCount: number = 25) {
    this.stepCount = stepCount;
    this.particleCount = particleCount;
    this.particlePositionBuffer = device.createBuffer({
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(BaseSimulator.ResourceFormats);
  }

  public abstract initResource(): Promise<void>;
  public abstract enableInteraction(): void;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { BaseSimulator };