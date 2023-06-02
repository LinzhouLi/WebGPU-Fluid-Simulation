import * as THREE from 'three';
import { GlobalResource } from './renderer/globalResource';
import { Config } from './common/config';
import { Skybox } from './renderer/skybox/skybox';
import { Mesh } from './renderer/mesh/mesh';
import { SPH } from './simulator/SPH';
import { PBF } from './simulator/PBF/PBF';
import { FilteredParticleFluid } from './renderer/filteredParticleFluid/fluid';
import { loader } from './common/loader';
import { resourceFactory } from './common/base';


// console.info( 'THREE.WebGPURenderer: Modified Matrix4.makePerspective() and Matrix4.makeOrtographic() to work with WebGPU, see https://github.com/mrdoob/three.js/issues/20276.' );
// @ts-ignore
THREE.Matrix4.prototype.makePerspective = function ( left, right, top, bottom, near, far ) : THREE.Matrix4 {
  
	const te = this.elements;
	const x = 2 * near / ( right - left );
	const y = 2 * near / ( top - bottom );

	const a = ( right + left ) / ( right - left );
	const b = ( top + bottom ) / ( top - bottom );
	// const c = - far / ( far - near );
	// const d = - far * near / ( far - near );
  const c = near / ( far - near );              // Reverse Z. https://vincent-p.github.io/posts/vulkan_perspective_matrix/
	const d = far * near / ( far - near );

	te[ 0 ] = x;	te[ 4 ] = 0;	te[ 8 ] = a;	te[ 12 ] = 0;
	te[ 1 ] = 0;	te[ 5 ] = y;	te[ 9 ] = b;	te[ 13 ] = 0;
	te[ 2 ] = 0;	te[ 6 ] = 0;	te[ 10 ] = c;	te[ 14 ] = d;
	te[ 3 ] = 0;	te[ 7 ] = 0;	te[ 11 ] = -1;	te[ 15 ] = 0;

	return this;

};

THREE.Matrix4.prototype.makeOrthographic = function ( left, right, top, bottom, near, far ) {

	const te = this.elements;
	const w = 1.0 / ( right - left );
	const h = 1.0 / ( top - bottom );
	const p = 1.0 / ( far - near );

	const x = ( right + left ) * w;
	const y = ( top + bottom ) * h;
	const z = near * p;

	te[ 0 ] = 2 * w;	te[ 4 ] = 0;		te[ 8 ] = 0;		te[ 12 ] = - x;
	te[ 1 ] = 0;		te[ 5 ] = 2 * h;	te[ 9 ] = 0;		te[ 13 ] = - y;
	te[ 2 ] = 0;		te[ 6 ] = 0;		te[ 10 ] = - 1 * p;	te[ 14 ] = - z;
	te[ 3 ] = 0;		te[ 7 ] = 0;		te[ 11 ] = 0;		te[ 15 ] = 1;

	return this;

};

THREE.Frustum.prototype.setFromProjectionMatrix = function ( m ) {

	const planes = this.planes;
	const me = m.elements;
	const me0 = me[ 0 ], me1 = me[ 1 ], me2 = me[ 2 ], me3 = me[ 3 ];
	const me4 = me[ 4 ], me5 = me[ 5 ], me6 = me[ 6 ], me7 = me[ 7 ];
	const me8 = me[ 8 ], me9 = me[ 9 ], me10 = me[ 10 ], me11 = me[ 11 ];
	const me12 = me[ 12 ], me13 = me[ 13 ], me14 = me[ 14 ], me15 = me[ 15 ];

	planes[ 0 ].setComponents( me3 - me0, me7 - me4, me11 - me8, me15 - me12 ).normalize();
	planes[ 1 ].setComponents( me3 + me0, me7 + me4, me11 + me8, me15 + me12 ).normalize();
	planes[ 2 ].setComponents( me3 + me1, me7 + me5, me11 + me9, me15 + me13 ).normalize();
	planes[ 3 ].setComponents( me3 - me1, me7 - me5, me11 - me9, me15 - me13 ).normalize();
	planes[ 4 ].setComponents( me3 - me2, me7 - me6, me11 - me10, me15 - me14 ).normalize();
	planes[ 5 ].setComponents( me2, me6, me10, me14 ).normalize();

	return this;

};


let device: GPUDevice;
let canvasFormat: GPUTextureFormat;
let canvasSize: { width: number, height: number };
let timeStampQuerySet: GPUQuerySet;
let frame = 0;

class Controller {

  // basic
  private config: Config;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private renderDepthView: GPUTextureView;
  private camera: THREE.PerspectiveCamera;
  private globalResource: GlobalResource;

  private ifSkybox: boolean;
  private ifMesh: boolean;
  private ifFluid: boolean;

  private skybox: Skybox;
  private mesh: Mesh;
  private simulator: SPH;
  private fluidRender: FilteredParticleFluid;

  private background_sea: ImageBitmap[];
  private background_church: ImageBitmap[];
  private background_fall: ImageBitmap[];
  private background_mountain: ImageBitmap[];
  private bunny_mesh: THREE.Mesh;
  private torus_mesh: THREE.Mesh;
  private torus_boundary: string;
  private domain_boundary: string;

  private timeStampSize = 9;
  private timeStampBuffer: GPUBuffer;
  private timeStampReadBuffer: GPUBuffer;
  private timeStampReadArray: Array<number>;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.config = new Config();
  }

  private RegisterResourceFormats() {
    GlobalResource.RegisterResourceFormats();
    Mesh.RegisterResourceFormats();
    FilteredParticleFluid.RegisterResourceFormats();
    SPH.RegisterResourceFormats();
  }

  public async initWebGPU() {

    if(!navigator.gpu) throw new Error('Not Support WebGPU');

    // adapter
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance' // 'low-power'
    });
    if (!adapter) throw new Error('No Adapter Found');
    adapter.features.forEach(feature => console.log(`Support feature: ${feature}`));

    const adapterInfo = await adapter.requestAdapterInfo();
    console.log(adapterInfo)
    
    // device
    device = await adapter.requestDevice({ // @ts-ignore
      // requiredFeatures: ['timestamp-query', 'timestamp-query-inside-passes'] // 'float32-filterable'
      requiredFeatures: []
    }); // "shader-f16" feature is not supported on my laptop
    console.log(device)
    // context
    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('Can Not Get GPUCanvasContext');
    this.context = context;

    // size
    canvasSize = { width: this.canvas.width, height: this.canvas.height };

    // format
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: device, 
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      format: canvasFormat,
      alphaMode: 'opaque' // prevent chrome warning
    })

  }

  private setSceneConfig(config: {
    property: string,
    object: {
      scene: number,
      skybox: number,
      fluid: boolean
    }
  }) {

    this.ifSkybox = config.object.skybox != 0;
    this.ifFluid = config.object.fluid;

    if (config.property == 'skybox' || config.property == 'all') {
      switch (config.object.skybox) {
        case 1: { this.globalResource.setSkybox(this.background_sea); break; }
        case 2: { this.globalResource.setSkybox(this.background_church); break; }
        case 3: { this.globalResource.setSkybox(this.background_fall); break; }
        case 4: { this.globalResource.setSkybox(this.background_mountain); break; }
      }
    }

    if (config.property == 'scene' || config.property == 'all') {
      this.simulator.reset();
      this.simulator.stop();
      switch (config.object.scene) {
        case 0: { // Bunny Drop
          this.simulator.voxelizeMesh(this.bunny_mesh);
          this.ifMesh = false;
          break;
        }
        case 1: { // Cube Drop
          this.simulator.voxelizeCube(
            new THREE.Vector3(0.15, 0.35, 0.15),
            new THREE.Vector3(0.65, 0.85, 0.65)
          );
          this.ifMesh = false;
          break;
        }
        case 2: { // Water Droplet
          this.simulator.voxelizeCube(
            new THREE.Vector3(0.005, 0.005, 0.005),
            new THREE.Vector3(0.995, 0.08, 0.995)
          );
          this.simulator.voxelizeSphere(
            new THREE.Vector3(0.5, 0.7, 0.5),
            0.12
          );
          this.ifMesh = false;
          break;
        }
        case 3: { // Double Dam Break
          this.simulator.voxelizeCube(
            new THREE.Vector3(0.005, 0.005, 0.005),
            new THREE.Vector3(0.3, 0.6, 0.3)
          );
          this.simulator.voxelizeCube(
            new THREE.Vector3(0.7, 0.005, 0.7),
            new THREE.Vector3(0.995, 0.6, 0.995)
          );
          this.ifMesh = false;
          break;
        }
        case 4: { // Boundary
          this.simulator.voxelizeCube(
            new THREE.Vector3(0.15, 0.35, 0.15),
            new THREE.Vector3(0.65, 0.85, 0.65)
          );
          this.simulator.setBoundaryData(this.torus_boundary);
          this.mesh.setMesh(this.torus_mesh);
          this.ifMesh = true;
          break;
        }
      }
      this.simulator.setParticlePosition();
      console.log(this.simulator.particleCount);
    }

  }

  public async loadData(onProgress: (percentage: number) => void) {

    const cubetex_sea = await loader.loadCubeTexture([
      "skybox/sea/right.jpg", "skybox/sea/left.jpg", // px nx
      "skybox/sea/top.jpg", "skybox/sea/bottom.jpg", // py ny
      "skybox/sea/front.jpg", "skybox/sea/back.jpg"  // pz nz
    ]);
    this.background_sea = await resourceFactory.toBitmaps(cubetex_sea.image);
    onProgress(35);

    const cubetex_church = await loader.loadCubeTexture([
      "skybox/church/posx.jpg", "skybox/church/negx.jpg", // px nx
      "skybox/church/posy.jpg", "skybox/church/negy.jpg", // py ny
      "skybox/church/posz.jpg", "skybox/church/negz.jpg"  // pz nz
    ]);
    this.background_church = await resourceFactory.toBitmaps(cubetex_church.image);
    onProgress(50);

    const cubetex_fall = await loader.loadCubeTexture([
      "skybox/fall/posx.jpg", "skybox/fall/negx.jpg", // px nx
      "skybox/fall/posy.jpg", "skybox/fall/negy.jpg", // py ny
      "skybox/fall/posz.jpg", "skybox/fall/negz.jpg"  // pz nz
    ]);
    this.background_fall = await resourceFactory.toBitmaps(cubetex_fall.image);
    onProgress(65);

    const cubetex_mountain = await loader.loadCubeTexture([
      "skybox/mountain/posx.jpg", "skybox/mountain/negx.jpg", // px nx
      "skybox/mountain/posy.jpg", "skybox/mountain/negy.jpg", // py ny
      "skybox/mountain/posz.jpg", "skybox/mountain/negz.jpg"  // pz nz
    ]);
    this.background_mountain = await resourceFactory.toBitmaps(cubetex_mountain.image);
    onProgress(80);

    const glb = await loader.loadGLTF("model/bunny.glb", true);
    this.bunny_mesh = glb.scene.children[0] as THREE.Mesh;
    this.bunny_mesh.scale.set(0.4, 0.4, 0.4);
    this.bunny_mesh.position.set(0.5, 0.3, 0.5);
    this.bunny_mesh.updateMatrixWorld();
    onProgress(90);

    const geometry = new THREE.TorusGeometry( 1.0, 0.4, 16, 60 );
    const material = new THREE.MeshPhongMaterial( { color: 0xffff00 } );
    this.torus_mesh = new THREE.Mesh( geometry, material );
    this.torus_mesh.position.set(0.5, 0.2, 0.5);
    this.torus_mesh.scale.set(0.2, 0.2, 0.2);
    this.torus_mesh.rotation.set(Math.PI / 2, 0, 0.0);
    this.torus_mesh.updateMatrixWorld();

    this.torus_boundary =  await loader.loadFile("model/torus.cdm") as string;
    // this.domain_boundary = await loader.loadFile("model/domain_boundary.cdm") as string;

  }

  public initTimeStamp() {

    timeStampQuerySet = device.createQuerySet({ type: 'timestamp', count: this.timeStampSize });
    this.timeStampReadArray = new Array(this.timeStampSize - 1).fill(0);
    this.timeStampBuffer = device.createBuffer({
      size: this.timeStampSize * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.timeStampReadBuffer = device.createBuffer({
      size: this.timeStampSize * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

  }

  public async initScene(camera: THREE.PerspectiveCamera, light: THREE.DirectionalLight) {
    
    this.RegisterResourceFormats();

    // global resource
    this.camera = camera;
    this.camera.updateMatrixWorld();
    this.camera.updateProjectionMatrix(); 
    this.globalResource = new GlobalResource(camera, light);
    await this.globalResource.initResource();
    this.renderDepthView = (this.globalResource.resource.renderDepthMap as GPUTexture).createView();

    // sky box renderer
    this.skybox = new Skybox();
    await this.skybox.initResouce(this.globalResource.bindgroupLayout);

    // mesh
    this.mesh = new Mesh();
    await this.mesh.initPipeline(this.globalResource.bindgroupLayout);

    // PBF simulator
    this.simulator = new PBF();
    await this.simulator.initResource();
    this.simulator.enableInteraction();
    this.config.initSimulationOptions((e) => this.simulator.optionsChange(e));
    this.simulator.setConfig(this.config.simulationOptions);

    // fluid renderer
    this.fluidRender = new FilteredParticleFluid(this.simulator, this.camera);
    await this.fluidRender.initResource(this.globalResource.resource);
    this.config.initRenderingOptions((e) => this.fluidRender.optionsChange(e));
    this.fluidRender.setConfig(this.config.renderingOptions);

    this.config.initSceneOptions(
      (e) => this.setSceneConfig(e),
      () => this.simulator.switch()
    );
    this.setSceneConfig({
      object: this.config.scnenOptions,
      property: 'all'
    });

  }

  public showConfigUI() {
    this.config.show();
  }

  public run() {

		const commandEncoder = device.createCommandEncoder();

    // simulate
    for (let i = 0; i < this.simulator.stepCount; i++) // this.simulator.stepCount
      this.simulator.run(commandEncoder);

		// render
    const ctxTextureView = this.context.getCurrentTexture().createView();
    const renderPassEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: ctxTextureView,
        clearValue: { r: 1, g: 1, b: 1, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: this.renderDepthView,
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      }
    });
    this.globalResource.setResource(renderPassEncoder);
    if(this.ifMesh) this.mesh.render(renderPassEncoder);
    if(this.ifSkybox) this.skybox.render(renderPassEncoder);
    renderPassEncoder.end();

    if (this.ifFluid) this.fluidRender.render(commandEncoder, ctxTextureView);

		const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

  }

  public async runTimestamp() {

    frame++;
		const commandEncoder = device.createCommandEncoder();
    commandEncoder.writeTimestamp(timeStampQuerySet, 0);

    // simulate
    for (let i = 0; i < this.simulator.stepCount; i++) // this.simulator.stepCount
      this.simulator.runTimestamp(commandEncoder);

		// render
    const ctxTextureView = this.context.getCurrentTexture().createView();
    const renderPassEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: ctxTextureView,
        clearValue: { r: 1, g: 1, b: 1, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: this.renderDepthView,
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      }
    });
    this.globalResource.setResource(renderPassEncoder);
    if(this.ifMesh) this.mesh.render(renderPassEncoder);
    if(this.ifSkybox) this.skybox.render(renderPassEncoder);
    renderPassEncoder.end();

    commandEncoder.writeTimestamp(timeStampQuerySet, 5);

    if (this.ifFluid) this.fluidRender.renderTimestamp(commandEncoder, ctxTextureView);

    commandEncoder.resolveQuerySet(
      timeStampQuerySet, 0, this.timeStampSize,
      this.timeStampBuffer, 0
    );
    commandEncoder.copyBufferToBuffer(
      this.timeStampBuffer, 0,
      this.timeStampReadBuffer, 0,
      this.timeStampBuffer.size
    );

		const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

    await device.queue.onSubmittedWorkDone();
    await this.timeStampReadBuffer.mapAsync(GPUMapMode.READ);
    const buffer = this.timeStampReadBuffer.getMappedRange(0, this.timeStampReadBuffer.size);
    const array = new BigUint64Array(buffer);
    for (let i = 1; i < this.timeStampSize; i++) {
      this.timeStampReadArray[i-1] += Number(array[i] - array[i-1]) * 1e-3;
    }
    // console.log(this.timeStampReadArray)
    this.timeStampReadBuffer.unmap()

    if (frame == 300) {
      this.timeStampReadArray.forEach(val => console.log(val / frame));
    }

  }

  public async debug() {

    this.update();
    await this.simulator.debug();

  }

  public update() {

    this.camera.updateMatrixWorld();
    this.camera.updateProjectionMatrix();
    this.globalResource.update();
    this.simulator.update();

  }

}

export { Controller, device, canvasFormat, canvasSize, timeStampQuerySet };