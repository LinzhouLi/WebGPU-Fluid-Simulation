import { LagrangianSimulator } from '../LagrangianSimulator';

abstract class PBFConfig extends LagrangianSimulator {

  static MAX_NEIGHBOR_COUNT = 50;
  static KERNEL_RADIUS = 0.025;
  static BOUNDARY_GRID = [20, 20, 20];

  protected constrainIterationCount = 5;
  protected timeStep = 1 / 100;
  protected lambdaEPS = 1e-6;
  protected scorrCoefK = 1e-6;
  protected scorrCoefDq = 1e-4; // [0.1, 0.3]
  protected scorrCoefN = 4; // fixed
  protected XSPHCoef = 0.80;
  protected restDensity: number = 1000.0;
  protected particleVolume: number;
  protected particleWeight: number;

  protected boundaryFilePath = "model/torus.cdm";

  constructor(particleCount: number = 1) {
    super(particleCount, 1); // sub step count = 1
  }

  public abstract initResource(): Promise<void>;
  public abstract enableInteraction(): void;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { PBFConfig };