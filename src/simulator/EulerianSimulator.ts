import type { ResourceType } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';
import { device } from '../controller';
import { resourceFactory, bindGroupFactory } from '../common/base';

class EulerianSimulator {

  private static ResourceFormats = {
  };

  private timeStep: number;
  private gridSize: { x: number; y: number; z: number };

  private resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>;

  constructor(gridX: number, gridY: number, gridZ: number) {

    this.gridSize = { x: gridX, y: gridY, z: gridZ };
    this.timeStep = 0.04; // 25 fps
    
  }

  public async initResource() {

    this.resource = await resourceFactory.createResource(
      [ 
        'velocityCur', 'velocityNxt', 
        'densityCur',  'densityNxt',
        'pressureCur', 'pressureNxt',
      ],
      {
        
      }
    );
    

  }

}

export { EulerianSimulator };