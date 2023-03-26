import { LagrangianSimulator } from '../LagrangianSimulator';

abstract class PBFConfig extends LagrangianSimulator {

  static MAX_NEIGHBOR_COUNT = 47;
  static KERNEL_RADIUS = 0.04;

  protected constrainIterationCount = 40;
  protected timeStep = 1 / 300;
  protected lambdaEPS = 100.0;
  protected scorrCoefK = 0.001;
  protected scorrCoefDq = 0.3; // [0.1, 0.3]
  protected scorrCoefN = 4; // fixed
  protected XSPHCoef = 0.01;
  protected restDensity: number;

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