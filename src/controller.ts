import * as THREE from 'three';
import { GlobalResource } from './renderer/globalResource';
import { SpriteParticles } from './renderer/spriteParticles/particles';
import { ParticleFluid } from './renderer/particleFluid/fluid'
import { Skybox } from './renderer/skybox';
import { LagrangianSimulator } from './simulator/LagrangianSimulator';
import { MPM } from './simulator/MPM';

import { NeighborList } from './simulator/neighbor/neighborList';


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

class Controller {

  // basic
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private renderBundle: GPURenderBundle;
  private renderDepthMap: GPUTexture;
  private camera: THREE.PerspectiveCamera;
  public globalResource: GlobalResource;

  private skybox: Skybox;
  // private particles: SpriteParticles;
  private fluidRender: ParticleFluid;
  private simulator: LagrangianSimulator;

  nl: NeighborList;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  private RegisterResourceFormats() {
    GlobalResource.RegisterResourceFormats();
    SpriteParticles.RegisterResourceFormats();
    ParticleFluid.RegisterResourceFormats();
    LagrangianSimulator.RegisterResourceFormats();
    MPM._RegisterResourceFormats();
  }

  public async initWebGPU() {

    if(!navigator.gpu) throw new Error('Not Support WebGPU');

    // adapter
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance' // 'low-power'
    });
    if (!adapter) throw new Error('No Adapter Found');
    adapter.features.forEach(feature => console.log(`Support feature: ${feature}`));
    
    // device
    device = await adapter.requestDevice({
      requiredFeatures: [] // 'float32-filterable'
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

  public async initScene(camera: THREE.PerspectiveCamera, light: THREE.DirectionalLight) {
    
    this.RegisterResourceFormats();

    // global resource
    this.camera = camera;
    this.camera.updateMatrixWorld();
    this.camera.updateProjectionMatrix(); 
    this.globalResource = new GlobalResource(camera, light);
    await this.globalResource.initResource();
    this.renderDepthMap = this.globalResource.resource.renderDepthMap as GPUTexture;

    // MPM simulator
    this.simulator = new MPM();
    await this.simulator.initResource();
    await this.simulator.initComputePipeline();
    this.simulator.enableInteraction();

    // fluid renderer
    // this.particles = new SpriteParticles(this.simulator);
    // this.particles.initVertexBuffer();
    // await this.particles.initGroupResource(this.globalResource.resource);
    // await this.particles.initPipeline();
    this.fluidRender = new ParticleFluid(this.simulator, this.camera);
    await this.fluidRender.initResource(this.globalResource.resource);

    // sky box renderer
    this.skybox = new Skybox();
    this.skybox.initVertexBuffer();
    await this.skybox.initGroupResource(this.globalResource.resource);
    await this.skybox.initPipeline();

    const renderBundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ canvasFormat ],
      depthStencilFormat: 'depth32float' // format of renderDepthMap
    });
    await this.skybox.setRenderBundle(renderBundleEncoder);
    // await this.particles.setRenderBundle(renderBundleEncoder);
    this.renderBundle = renderBundleEncoder.finish();

    // neighbor debug
    this.nl = new NeighborList();
    await this.nl.initResource()

  }

  public run() {

		const commandEncoder = device.createCommandEncoder();

    const pe = commandEncoder.beginComputePass();
    this.nl.execute(pe);
    pe.end();
    
    // simulate
    // for (let i = 0; i < this.simulator.stepCount; i++) // this.simulator.stepCount
    //   this.simulator.run(commandEncoder);

		// render
    // const ctxTextureView = this.context.getCurrentTexture().createView();
    // const renderPassEncoder = commandEncoder.beginRenderPass({
    //   colorAttachments: [{
    //     view: ctxTextureView,
    //     clearValue: { r: 0, g: 0, b: 0, a: 0.0 },
    //     loadOp: 'clear',
    //     storeOp: 'store'
    //   }],
    //   depthStencilAttachment: {
    //     view: this.renderDepthMap.createView(),
    //     depthClearValue: 0.0,
    //     depthLoadOp: 'clear',
    //     depthStoreOp: 'store',
    //   }
    // });
    // renderPassEncoder.executeBundles([this.renderBundle]);
    // renderPassEncoder.end();

    // this.fluidRender.render(commandEncoder, ctxTextureView);

		const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

    // this.simulator.debug();
    this.nl.debug();

  }

  public update() {

    this.camera.updateMatrixWorld();
    this.camera.updateProjectionMatrix();
    this.globalResource.update();
    this.simulator.update();

  }

}

export { Controller, device, canvasFormat, canvasSize };