import { SPH } from '../SPH';
import { device } from '../../controller';

abstract class PBFConfig extends SPH {

  protected ifBoundary: boolean;
  protected constrainIterationCount;
  protected lambdaEPS = 1e-6;
  protected scorrCoefK = 5e-5;
  protected scorrCoefDq = 0.1; // [0.1, 0.3]
  protected scorrCoefN = 4; // fixed

  protected boundaryFilePath = "model/torus.cdm";

  protected optionsArray: ArrayBuffer;
  protected optionsBuffer: GPUBuffer;
  protected optionsBufferView: DataView;

  constructor() {

    super(0.007, 1); // particle radius = 0.006, sub step count = 1

    this.ifBoundary = false;
    const bufferSize = 8 * Float32Array.BYTES_PER_ELEMENT;
    this.optionsArray = new ArrayBuffer(bufferSize);
    this.optionsBufferView = new DataView(this.optionsArray);
    this.optionsBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  }

  public setConfig(config: {
    iteration: number,
    XSPH: number,
    vorticity: number,
    surfaceTension: number,
    gravity: number
  }) {

    this.constrainIterationCount = config.iteration;
    this.optionsBufferView.setFloat32(4, config.XSPH, true);
    this.optionsBufferView.setFloat32(8, config.vorticity, true);
    this.optionsBufferView.setFloat32(12, config.surfaceTension, true);
    this.optionsBufferView.setFloat32(20, -config.gravity, true);
    device.queue.writeBuffer(this.optionsBuffer, 4, this.optionsArray, 4);

  }

  public override setParticlePosition() {

    if (this.particlePositionArray.length != this.particleCount * 4) {
      throw new Error('Illegal length of Particle Position Buffer!');
    }
    if (this.particleCount > SPH.MAX_PARTICLE_NUM) {
      throw new Error('To Many Particles!');
    }

    const bufferArray = new Float32Array(this.particlePositionArray);
    device.queue.writeBuffer( this.position, 0, bufferArray, 0 );
    this.optionsBufferView.setUint32(0, this.particleCount, true);
    device.queue.writeBuffer(this.optionsBuffer, 0, this.optionsArray, 0, 4);

  }

  public optionsChange(e) {
    this.setConfig(e.object)
  }

  public abstract initResource(): Promise<void>;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract runTimestamp(commandEncoder: GPUCommandEncoder): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { PBFConfig };