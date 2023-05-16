import { SPH } from '../SPH';
import { device } from '../../controller';

abstract class PBFConfig extends SPH {

  static BOUNDARY_GRID = [20, 20, 20];

  protected optionsArray: ArrayBuffer;
  protected optionsBuffer: GPUBuffer;
  protected optionsBufferView: DataView;

  protected constrainIterationCount = 5;
  protected lambdaEPS = 1e-6;
  protected scorrCoefK = 5e-5;
  protected scorrCoefDq = 0.1; // [0.1, 0.3]
  protected scorrCoefN = 4; // fixed

  protected XSPHCoef = 0.01;
  protected VorticityCoef = 0.1;
  protected SurfaceTensionCoef = 0.5;

  protected boundaryFilePath = "model/torus.cdm";

  constructor() {

    super(0.007, 1); // particle radius = 0.006, sub step count = 1

    const bufferSize = 4 * Float32Array.BYTES_PER_ELEMENT;
    this.optionsArray = new ArrayBuffer(bufferSize);
    this.optionsBufferView = new DataView(this.optionsArray);
    this.optionsBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  }

  public setConfig(config: {
    XSPH: number,
    vorticity: number,
    surfaceTension: number
  }) {

    this.optionsBufferView.setInt32(0, this.particleCount, true);
    this.optionsBufferView.setFloat32(4, config.XSPH, true);
    this.optionsBufferView.setFloat32(8, config.vorticity, true);
    this.optionsBufferView.setFloat32(12, config.surfaceTension, true);
    device.queue.writeBuffer(this.optionsBuffer, 0, this.optionsArray, 0);

  }

  public optionsChange(e) {
    this.setConfig(e)
  }

  public abstract initResource(): Promise<void>;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { PBFConfig };