import * as THREE from 'three';
import { GlobalResource } from './renderer/globalResource';
import { Config } from './common/config';
import { Skybox } from './renderer/skybox/skybox';
import { Mesh } from './renderer/mesh/mesh';
import { SPH } from './simulator/SPH';
import { PBF } from './simulator/PBF/PBF';
import { FluidRenderer } from './renderer/fluid/fluid';
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
  private ifParticles: boolean;

  private skybox: Skybox;
  private mesh: Mesh;
  private simulator: SPH;
  private fluidRenderer: FluidRenderer;

  private background_sea: ImageBitmap[];
  private bunny_mesh: THREE.Mesh;
  private torus_mesh: THREE.Mesh;
  private torus_boundary: string;
  private domain_boundary: string;

  private timeStampSize = 6;
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

    const requestAdapterInfo = (adapter as GPUAdapter & {
      requestAdapterInfo?: () => Promise<unknown>;
      info?: unknown;
    }).requestAdapterInfo;
    const adapterInfo = typeof requestAdapterInfo === 'function'
      ? await requestAdapterInfo.call(adapter)
      : (adapter as GPUAdapter & { info?: unknown }).info;
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
      particles: boolean
    }
  }) {

    this.ifSkybox = config.object.skybox != 0;
    this.ifParticles = config.object.particles;

    if (config.property == 'skybox' || config.property == 'all') {
      switch (config.object.skybox) {
        case 1: { this.globalResource.setSkybox(this.background_sea); break; }
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

    // fluid screen-space renderer (replaces RawParticles)
    this.fluidRenderer = new FluidRenderer(this.simulator);
    await this.fluidRenderer.initResource(this.globalResource.bindgroupLayout);

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

    // simulation
    for (let i = 0; i < this.simulator.stepCount; i++)
      this.simulator.run(commandEncoder);

    // === Pass 1: Scene background → sceneColor + sceneDepth ===
    const scenePass = this.fluidRenderer.beginScenePass(commandEncoder);
    this.globalResource.setResource(scenePass);
    if (this.ifMesh) this.mesh.render(scenePass);
    if (this.ifSkybox) this.skybox.render(scenePass);
    scenePass.end();

    if (this.ifParticles && this.simulator.particleCount > 0) {
      // === Pass 2: Billboard particles → fluidDepth ===
      const billboardPass = this.fluidRenderer.beginBillboardPass(commandEncoder);
      this.globalResource.setResource(billboardPass);
      this.fluidRenderer.renderBillboards(billboardPass);
      billboardPass.end();

      // === Pass 3: Narrow-range filter horizontal ===
      const filterHPass = this.fluidRenderer.beginFilterHPass(commandEncoder);
      this.fluidRenderer.renderFilterH(filterHPass);
      filterHPass.end();

      // === Pass 4: Narrow-range filter vertical ===
      const filterVPass = this.fluidRenderer.beginFilterVPass(commandEncoder);
      this.fluidRenderer.renderFilterV(filterVPass);
      filterVPass.end();

      // === Pass 5: Surface shading composite → canvas ===
      const ctxTextureView = this.context.getCurrentTexture().createView();
      const surfacePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: ctxTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      this.globalResource.setResource(surfacePass);
      this.fluidRenderer.renderSurface(surfacePass);
      surfacePass.end();
    } else {
      // No particles: render scene directly to canvas with depth attachment
      const ctxTextureView = this.context.getCurrentTexture().createView();
      const copyPass = this.fluidRenderer.beginCanvasPass(commandEncoder, ctxTextureView);
      this.globalResource.setResource(copyPass);
      if (this.ifSkybox) this.skybox.render(copyPass);
      if (this.ifMesh) this.mesh.render(copyPass);
      copyPass.end();
    }

    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

  }

  public async runTimestamp() {

    frame++;
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.writeTimestamp(timeStampQuerySet, 0);

    // simulation
    for (let i = 0; i < this.simulator.stepCount; i++)
      this.simulator.runTimestamp(commandEncoder);

    // Scene pass
    const scenePass = this.fluidRenderer.beginScenePass(commandEncoder);
    this.globalResource.setResource(scenePass);
    if (this.ifMesh) this.mesh.render(scenePass);
    if (this.ifSkybox) this.skybox.render(scenePass);
    scenePass.end();

    if (this.ifParticles && this.simulator.particleCount > 0) {
      // Billboard pass
      const billboardPass = this.fluidRenderer.beginBillboardPass(commandEncoder);
      this.globalResource.setResource(billboardPass);
      this.fluidRenderer.renderBillboards(billboardPass);
      billboardPass.end();

      // Filter passes
      const filterHPass = this.fluidRenderer.beginFilterHPass(commandEncoder);
      this.fluidRenderer.renderFilterH(filterHPass);
      filterHPass.end();

      const filterVPass = this.fluidRenderer.beginFilterVPass(commandEncoder);
      this.fluidRenderer.renderFilterV(filterVPass);
      filterVPass.end();

      // Surface composite
      const ctxTextureView = this.context.getCurrentTexture().createView();
      const surfacePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: ctxTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      this.globalResource.setResource(surfacePass);
      this.fluidRenderer.renderSurface(surfacePass);
      surfacePass.end();
    } else {
      const ctxTextureView = this.context.getCurrentTexture().createView();
      const copyPass = this.fluidRenderer.beginCanvasPass(commandEncoder, ctxTextureView);
      this.globalResource.setResource(copyPass);
      if (this.ifSkybox) this.skybox.render(copyPass);
      if (this.ifMesh) this.mesh.render(copyPass);
      copyPass.end();
    }

    commandEncoder.writeTimestamp(timeStampQuerySet, 5);

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

    // Update inverse projection for fluid surface shader
    const invProj = this.camera.projectionMatrixInverse.toArray();
    this.fluidRenderer.updateInvProjection(invProj);

  }

}

export { Controller, device, canvasFormat, canvasSize, timeStampQuerySet };
