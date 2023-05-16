import * as THREE from 'three';
import { device, canvasSize, canvasFormat } from '../controller';
import type { TypedArray } from '../common/base';
import { resourceFactory, bindGroupFactory, EnvMapResolution } from '../common/base';
import type { ResourceType, BufferData, TextureData, TextureArrayData } from '../common/resourceFactory';
import { ResourceFactory } from '../common/resourceFactory';


class GlobalResource {

  private camera: THREE.PerspectiveCamera;
  private light: THREE.PointLight | THREE.DirectionalLight;

  private resourceData: Record<string, BufferData | TextureData | TextureArrayData>; // resource in CPU
  public resource: Record<string, GPUBuffer | GPUTexture | GPUSampler>; // resource in GPU
  public bindgroupLayout: GPUBindGroupLayout;
  private bindgroup: GPUBindGroup;

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
        visibility: GPUShaderStage.FRAGMENT,
        usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        layout: { 
          type: 'uniform' as GPUBufferBindingType
        } as GPUBufferBindingLayout
      },

      // sampler
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

      // env map
      envMap: {
        type: 'cube-texture' as ResourceType,
        label: 'Skybox Map',
        visibility: GPUShaderStage.FRAGMENT,
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


  private initCameraResource() {

    // camera params. used for functions: linear01Depth(), linearEyeDepth()
		let height = 2 * this.camera.near * Math.tan( Math.PI / 180 * 0.5 * this.camera.fov ) / this.camera.zoom;
    let n = this.camera.near; 
    let f = this.camera.far;

    let paramX = this.camera.aspect * height / n; // width / near
    let paramY = -height / n; // - height / near
    let paramZ = (f - n) / (n * f);
    let paramW = 1.0 / f;

    let cameraBuffer = new Float32Array(4 + 16 * 3 + 4);
    cameraBuffer.set([ paramX, paramY, paramZ, paramW ], 4 + 16 * 3);

    return cameraBuffer;

  }

  public async initResource(background: THREE.CubeTexture) {

    const light = this.light as THREE.DirectionalLight;
    let lightDir = light.position.clone().sub(light.target.position).normalize();
    let lightColor = new THREE.Vector3(...this.light.color.toArray()).setScalar(this.light.intensity);
    
    this.resourceData = {
      camera: { // update per frame
        value: this.initCameraResource()
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
    
    this.resource = await resourceFactory.createResource(
      [ 'renderDepthMap', 'camera', 'directionalLight', 'linearSampler', 'envMap' ], 
      this.resourceData
    );

    const layout_group = bindGroupFactory.create(
      [ 'camera', 'directionalLight', 'linearSampler', 'envMap' ],
      this.resource
    );
    this.bindgroupLayout = layout_group.layout;
    this.bindgroup = layout_group.group;

  }

  public async setResource(
    encoder: GPURenderBundleEncoder | GPURenderPassEncoder
  ) {

    encoder.setBindGroup(0, this.bindgroup);

  }

  public update() {

    // camera
    this.camera.position.setFromMatrixPosition(this.camera.matrixWorld);

    const cameraBufferData = this.resourceData.camera as BufferData;
    (cameraBufferData.value as TypedArray).set([
      ...this.camera.position.toArray(), 0,
      ...this.camera.matrixWorldInverse.toArray(),
      ...this.camera.matrixWorld.toArray(),
      ...this.camera.projectionMatrix.toArray()
    ]);

    device.queue.writeBuffer(
      this.resource.camera as GPUBuffer, 0, 
      cameraBufferData.value as TypedArray, 0
    );

  }

}

export { GlobalResource }