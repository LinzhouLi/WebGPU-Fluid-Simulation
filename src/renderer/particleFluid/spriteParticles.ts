import { device } from '../../controller';
import { SpriteParticles } from '../spriteParticles/particles';
import { LagrangianSimulator } from '../../simulator/LagrangianSimulator';
import { wgsl } from '../../3rd-party/wgsl-preprocessor';

const depthPassfragmentShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};

struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>
};

struct Material {
  sphereRadius: f32,
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @builtin(frag_depth) frag_depth: f32
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  // caculate normal from uv
  var normalCam = vec4<f32>(input.vUv * 2.0 - 1.0, 0.0, 0.0);
  let radius2 = dot(normalCam.xy, normalCam.xy);
  if (radius2 > 1.0) { discard; }
  normalCam.z = sqrt(1.0 - radius2);

  // caculate depth
  let fragPosCam = input.vPositionCam + normalCam * material.sphereRadius;
  let positionClip = camera.projectionMatrix * fragPosCam;
  let depth = positionClip.z / positionClip.w;

  // caculate volume // sigma = 0.5
  // let volume = exp(-radius2 * 2.0) * 0.2;

  return FragmentOutput( depth );
}

`;

const volumePassfragmentShader = wgsl/* wgsl */`

struct Camera {
  position: vec3<f32>,
  viewMatrix: mat4x4<f32>,
  viewMatrixInverse: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  params: vec4<f32>
};

struct DirectionalLight {
  direction: vec3<f32>,
  color: vec3<f32>
};

struct Material {
  sphereRadius: f32,
  metalness: f32,
  specularIntensity: f32,
  roughness: f32,
  color: vec3<f32>
};

struct FragmentInput {
  @location(0) @interpolate(perspective, center) vPositionCam: vec4<f32>,
  @location(1) @interpolate(perspective, center) vUv: vec2<f32>,
};

struct FragmentOutput {
  @location(0) volume: f32
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(3) var<uniform> light: DirectionalLight;

@fragment
fn main(input: FragmentInput) -> FragmentOutput {

  // caculate normal from uv
  var normalCam = vec4<f32>(input.vUv * 2.0 - 1.0, 0.0, 0.0);
  let radius2 = dot(normalCam.xy, normalCam.xy);
  if (radius2 > 1.0) { discard; }
  // normalCam.z = sqrt(1.0 - radius2);

  // caculate depth
  // let fragPosCam = input.vPositionCam + normalCam * material.sphereRadius;
  // let positionClip = camera.projectionMatrix * fragPosCam;
  // let depth = positionClip.z / positionClip.w;

  // caculate volume // sigma = 0.5
  let volume = exp(-radius2 * 2.0) * 0.2;

  return FragmentOutput( volume );
}

`;


class FluidParicles extends SpriteParticles {

  protected depthRenderPipeline: GPURenderPipeline;
  protected volumeRenderPipeline: GPURenderPipeline;

  protected depthRenderBundle: GPURenderBundle;
  protected volumeRenderBundle: GPURenderBundle;

  constructor(simulator: LagrangianSimulator) {

    super(simulator);

  }

  public override async initPipeline() {

    // depth pass
    this.depthRenderPipeline = await device.createRenderPipelineAsync({
      label: 'Sprite Particle Depth Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        entryPoint: 'main',
        buffers: this.vertexBufferLayout
      },
      fragment: {
        module: device.createShaderModule({ code: depthPassfragmentShader }),
        entryPoint: 'main',
        targets: []
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back'
      }, 
      depthStencil: {
        depthWriteEnabled: true, // enable depth test
        depthCompare: 'greater',
        format: 'depth32float'
      }
    });

    // volume pass
    this.volumeRenderPipeline = await device.createRenderPipelineAsync({
      label: 'Sprite Particle Depth Render Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        entryPoint: 'main',
        buffers: this.vertexBufferLayout
      },
      fragment: {
        module: device.createShaderModule({ code: volumePassfragmentShader }),
        entryPoint: 'main',
        targets: [{ // volume blend
          format: 'r16float',
          writeMask: GPUColorWrite.RED,
          blend: {
            color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
          }
        }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back'
      },
    });

  }

  private setRenderCommands(
    bundleEncoder: GPURenderBundleEncoder,
    pipeline: GPURenderPipeline
  ) {

    bundleEncoder.setPipeline(pipeline);
    let loction = 0, indexed = false;
    for (const attribute of this.vertexBufferAttributes) {
      if (attribute === 'index') {
        bundleEncoder.setIndexBuffer(this.vertexBuffers.index, 'uint16');
        indexed = true;
      }
      else {
        bundleEncoder.setVertexBuffer(loction, this.vertexBuffers[attribute]);
        loction++;
      }
    }
    bundleEncoder.setBindGroup(0, this.bindGroup);
    if (indexed) bundleEncoder.drawIndexed(this.vertexCount, this.simulator.particleCount);
    else bundleEncoder.draw(this.vertexCount, this.simulator.particleCount);

  }

  public initRenderBundle() {

    // depth pass
    let bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [],
      depthStencilFormat: 'depth32float' // format of depthMap
    });
    this.setRenderCommands(bundleEncoder, this.depthRenderPipeline);
    this.depthRenderBundle = bundleEncoder.finish();

    // volume pass
    bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ 'r16float' ]
    });
    this.setRenderCommands(bundleEncoder, this.volumeRenderPipeline);
    this.volumeRenderBundle = bundleEncoder.finish();

  }

  public render(
    commandEncoder: GPUCommandEncoder,
    depthMap: GPUTexture,
    volumeMap: GPUTexture
  ) {

    const renderPassEncoder1 = commandEncoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: depthMap.createView(),
        depthClearValue: 0.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      }
    });
    renderPassEncoder1.executeBundles([this.depthRenderBundle]);
    renderPassEncoder1.end();

    const renderPassEncoder2 = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: volumeMap.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    renderPassEncoder2.executeBundles([this.volumeRenderBundle]);
    renderPassEncoder2.end();

  }

}

export { FluidParicles };