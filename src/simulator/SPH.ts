import * as THREE from 'three';
import { MeshBVH  } from 'three-mesh-bvh';
import type { ResourceType } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';
import { device } from '../controller';

abstract class SPH {

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

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats(SPH.ResourceFormats);
  }

  static MAX_PARTICLE_NUM = 1 << 17 - 1;
  static MAX_NEIGHBOR_COUNT = 60;
  static KERNEL_RADIUS = 0.025;
  static TIME_STEP = 1 / 300;

  public pause: boolean;
  public particleCount: number;
  public stepCount: number;
  protected particleRadius: number;
  protected restDensity: number = 1000.0;
  protected particleVolume: number;
  protected particleWeight: number;

  protected particlePositionArray: Array<number>;

  public position: GPUBuffer;
  protected velocity: GPUBuffer;
  protected acceleration: GPUBuffer;

  constructor(particleRadius: number = 0.006, stepCount: number = 25) {

    this.pause = true;
    this.stepCount = stepCount;
    this.particleRadius = particleRadius;
    const particleDiam = 2 * this.particleRadius;
    this.particleVolume = 0.9 * particleDiam * particleDiam * particleDiam;
    this.particleWeight = this.particleVolume * this.restDensity;

    this.particlePositionArray = [];
    this.particleCount = 0;

  }

  public baseReset(commandEncoder: GPUCommandEncoder) {

    commandEncoder.clearBuffer(this.position);
    commandEncoder.clearBuffer(this.velocity);
    commandEncoder.clearBuffer(this.acceleration);

    this.particlePositionArray = [];
    this.particleCount = 0;

  }

  public setParticlePosition() {

    if (this.particlePositionArray.length != this.particleCount * 4) {
      throw new Error('Illegal length of Particle Position Buffer!');
    }
    if (this.particleCount > SPH.MAX_PARTICLE_NUM) {
      throw new Error('To Many Particles!');
    }

    const bufferArray = new Float32Array(this.particlePositionArray);
    device.queue.writeBuffer( this.position, 0, bufferArray, 0 );

  }

  public stop() { this.pause = true; }
  public start() { this.pause = this.position ? false : true; }

  public createBaseStorageData() {

    const attributeBufferDesp = {
      size: 4 * SPH.MAX_PARTICLE_NUM * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    } as GPUBufferDescriptor;

    this.position = device.createBuffer(attributeBufferDesp);
    this.velocity = device.createBuffer(attributeBufferDesp);
    this.acceleration = device.createBuffer(attributeBufferDesp);

  }

  public voxelizeCube( min: THREE.Vector3, max: THREE.Vector3 ) {

    const particleDiam = 2 * this.particleRadius;
    const voxelDimX = Math.floor((max.x - min.x) / particleDiam);
    const voxelDimY = Math.floor((max.y - min.y) / particleDiam);
    const voxelDimZ = Math.floor((max.z - min.z) / particleDiam);

    let position = new THREE.Vector3();
    let count = 0;
    for (let i = 0; i < voxelDimX; i++) {
      for (let j = 0; j < voxelDimY; j++) {
        for (let k = 0; k < voxelDimZ; k++) {
          position.set(i, j, k).multiplyScalar(particleDiam).add(min);
          this.particlePositionArray.push(position.x, position.y, position.z, 0);
          count++;
        }
      }
    }

    this.particleCount += count;
    return count;

  }

  public voxelizeSphere( center: THREE.Vector3, radius: number ) {

    const particleDiam = 2 * this.particleRadius;
    const voxelDimX = Math.floor(radius / this.particleRadius / 2);
    const voxelDimY = voxelDimX;
    const voxelDimZ = voxelDimX;
    const radius2 = radius * radius;

    let position = new THREE.Vector3();
    let count = 0;
    for (let i = -voxelDimX; i <= voxelDimX; i++) {
      for (let j = -voxelDimY; j <= voxelDimY; j++) {
        for (let k = -voxelDimZ; k <= voxelDimZ; k++) {
          position.set(i, j, k).multiplyScalar(particleDiam);
          if (position.lengthSq() <= radius2) {
            position.add(center);
            this.particlePositionArray.push(position.x, position.y, position.z, 0);
            count++;
          }
        }
      }
    }

    this.particleCount += count;
    return count;

  }

  public voxelizeMesh( mesh: THREE.Mesh ) {

    const particleDiam = 2 * this.particleRadius;
    mesh.updateMatrixWorld();
    const aabb = new THREE.Box3().expandByObject(mesh);

    const voxelDimX = Math.floor((aabb.max.x - aabb.min.x) / particleDiam);
    const voxelDimY = Math.floor((aabb.max.y - aabb.min.y) / particleDiam);
    const voxelDimZ = Math.floor((aabb.max.z - aabb.min.z) / particleDiam);

    const ray = new THREE.Ray();
    ray.direction.set( 0, 0, 1 );
    const invMat = new THREE.Matrix4().copy( mesh.matrixWorld ).invert();
    const bvh = new MeshBVH(mesh.geometry);

    let position = new THREE.Vector3();
    let count = 0;
    for (let i = 0; i < voxelDimX; i++) {
      for (let j = 0; j < voxelDimY; j++) {
        for (let k = 0; k < voxelDimZ; k++) {

          position.set(i, j, k).multiplyScalar(particleDiam).add(aabb.min);
          ray.origin.copy( position ).applyMatrix4( invMat );
          const res = bvh.raycastFirst( ray, 2 );
					if ( res && res.face.normal.dot( ray.direction ) > 0.0 ) {
            this.particlePositionArray.push(position.x, position.y, position.z, 0);
            count++;
					}

        }
      }
    }

    this.particleCount += count;
    return count;

  }

  public enableInteraction() {
    
    document.addEventListener('keydown', event => {
      if (event.key.toUpperCase() === ' ') {
        this.pause = !this.pause || !this.position;
      }
    });

  }

  public abstract initResource(): Promise<void>;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract reset(): void;
  public abstract setBoundaryData(data: string): void;
  public abstract optionsChange(e: any): void;
  public abstract setConfig(conf: any): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { SPH };