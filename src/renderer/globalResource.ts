import * as THREE from 'three';
import { loader } from '../common/loader';
import { device, canvasSize } from '../controller';
import type { TypedArray } from '../common/base';
import { resourceFactory, EnvMapResolution } from '../common/base';
import type { ResourceType, BufferData, TextureData, TextureArrayData } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';


class GlobalResource {

  private camera: THREE.PerspectiveCamera;
  private light: THREE.PointLight | THREE.DirectionalLight;

  private resourceAttributes: string[]; // resource name
  private resourceCPUData: Record<string, BufferData | TextureData | TextureArrayData>; // resource in CPU
  
  public resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU

  constructor(camera: THREE.PerspectiveCamera, light: THREE.DirectionalLight) {

    this.camera = camera;
    this.light = light;

  }

  public static RegisterResourceFormats() {
    ResourceFactory.RegisterFormats({

      // render attachment // no binding
      renderDepthMap: {
        type: 'texture' as ResourceType,
        label: 'Render Depth Map',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        size: [canvasSize.width, canvasSize.height],
        dimension: '2d' as GPUTextureDimension,
        format: 'depth32float' as GPUTextureFormat,
      },
  
      // camera
      camera: {
        type: 'buffer' as ResourceType,
        label: 'Camera Structure', 
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        layout: { 
          type: 'uniform' as GPUBufferBindingType
        } as GPUBufferBindingLayout
      },
  
      // light
      directionalLight: {
        type: 'buffer' as ResourceType,
        label: 'Directional Light Structure', // direction(vec3<f32>), color(vec3<f32>), view projection matrix(mat4x4<f32>)
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        layout: { 
          type: 'uniform' as GPUBufferBindingType
        } as GPUBufferBindingLayout
      },

      linearSampler: {
        type: 'sampler' as ResourceType,
        label: 'Linear Sampler',
        visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        magFilter: 'linear' as GPUFilterMode,
        minFilter: 'linear' as GPUFilterMode,
        layout: { 
          type: 'filtering' as GPUSamplerBindingType 
        } as GPUSamplerBindingLayout
      }, 

      envMap: {
        type: 'cube-texture' as ResourceType,
        label: 'Skybox Map',
        visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        size: [EnvMapResolution, EnvMapResolution, 6],
        dimension: '2d' as GPUTextureDimension,
        mipLevelCount: 1,
        format: 'rgba8unorm' as GPUTextureFormat, 
        viewFormat: 'rgba8unorm-srgb', // with GPU Gamma decoding
        layout: { 
          sampleType: 'float' as GPUTextureSampleType,
          viewDimension: 'cube' as GPUTextureViewDimension
        } as GPUTextureBindingLayout
      },
  
    });
  }
  public async initResource() {

    this.resourceAttributes = [ 'renderDepthMap', 'camera', 'directionalLight', 'envMap', 'linearSampler' ];

    const light = this.light as THREE.DirectionalLight;
    // light.position.setFromMatrixPosition(light.matrixWorld);
    // light.target.position.setFromMatrixPosition(light.target.matrixWorld);
    let lightDir = light.position.clone().sub(light.target.position).normalize();
    let lightColor = new THREE.Vector3(...this.light.color.toArray()).setScalar(this.light.intensity);

    let background = await loader.loadCubeTexture([
      "skybox/right.jpg", "skybox/left.jpg", // px nx
      "skybox/top.jpg", "skybox/bottom.jpg", // py ny
      "skybox/front.jpg", "skybox/back.jpg"  // pz nz
    ]);
    
    this.resourceCPUData = {
      camera: { // update per frame
        value: new Float32Array((4 + 16 + 16) * 4)
      },
      directionalLight: { 
        value: new Float32Array([
          ...lightDir.toArray(), 0,
          ...lightColor.toArray(), 0
        ])
      },
      envMap: { 
        value: await resourceFactory.toBitmaps(background.image),
        flipY: new Array(6).fill(background.flipY)
      }
    }
    
    this.resource = await resourceFactory.createResource(this.resourceAttributes, this.resourceCPUData);

  }

  public update() {

    // camera
    this.camera.position.setFromMatrixPosition(this.camera.matrixWorld);

    const cameraBufferData = this.resourceCPUData.camera as BufferData;
    cameraBufferData.value.set([
      ...this.camera.position.toArray(), 0,
      ...this.camera.matrixWorldInverse.toArray(),
      ...this.camera.projectionMatrix.toArray()
    ]);

    device.queue.writeBuffer(
      this.resource.camera as GPUBuffer, 0, 
      cameraBufferData.value as TypedArray, 0
    );

  }

}

export { GlobalResource }