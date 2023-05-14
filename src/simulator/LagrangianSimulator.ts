import * as THREE from 'three';
import { MeshBVH  } from 'three-mesh-bvh';
import type { ResourceType } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';
import { device } from '../controller';

abstract class LagrangianSimulator {

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
    ResourceFactory.RegisterFormats(LagrangianSimulator.ResourceFormats);
  }

  static MAX_NEIGHBOR_COUNT = 50;
  static KERNEL_RADIUS = 0.025;

  public pause: boolean;
  public particleCount: number;
  public stepCount: number;
  protected particleRadius: number;
  protected gravity: number;

  protected particlePositionArray: Array<number>;
  protected gravityArray: Float32Array;

  public position: GPUBuffer;
  protected velocity: GPUBuffer;
  protected acceleration: GPUBuffer;
  protected gravityBuffer: GPUBuffer;

  constructor(particleRadius: number = 0.008, stepCount: number = 25) {

    this.pause = true;
    this.particleRadius = particleRadius;
    this.stepCount = stepCount;

    this.gravity = 9.8;
    this.gravityBuffer = device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.gravityArray = new Float32Array(4);
    this.gravityArray.set([0, -this.gravity, 0, 0]);
    device.queue.writeBuffer(
      this.gravityBuffer, 0,
      this.gravityArray, 0
    );

    this.clearParticles();

  }

  public stop() { this.pause = true; }
  public start() { this.pause = this.position ? false : true; }

  public clearParticles() {

    this.particlePositionArray = [];
    this.particleCount = 0;
    this.position = null;

  }

  public createBasicStorageData() {
    
    if (this.particlePositionArray.length != this.particleCount * 4) {
      throw new Error('Illegal length of Particle Position Buffer!');
    }

    const attributeBufferDesp = {
      size: 4 * this.particleCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    } as GPUBufferDescriptor;

    this.position = device.createBuffer(attributeBufferDesp);
    this.velocity = device.createBuffer(attributeBufferDesp);
    this.acceleration = device.createBuffer(attributeBufferDesp);

    const bufferArray = new Float32Array(this.particlePositionArray);
    device.queue.writeBuffer( this.position, 0, bufferArray, 0 );

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
      if (event.key.toUpperCase() === 'W') {
        this.gravityArray.set([-this.gravity, 0, 0, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === 'A') {
        this.gravityArray.set([0, 0, this.gravity, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === 'S') {
        this.gravityArray.set([this.gravity, 0, 0, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === 'D') {
        this.gravityArray.set([0, 0, -this.gravity, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === 'Q') {
        this.gravityArray.set([0, this.gravity, 0, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === 'E') {
        this.gravityArray.set([0, -this.gravity, 0, 0]);
        device.queue.writeBuffer( this.gravityBuffer, 0, this.gravityArray, 0 );
      }
      else if (event.key.toUpperCase() === ' ') {
        this.pause = !this.pause || !this.position;
      }
    });

  }

  public abstract initResource(): Promise<void>;
  public abstract initComputePipeline(): Promise<void>;
  public abstract run(commandEncoder: GPUCommandEncoder): void;
  public abstract update(): void;
  public abstract debug(): Promise<void>;

}

export { LagrangianSimulator };