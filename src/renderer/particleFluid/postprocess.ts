import { device, canvasSize, canvasFormat } from '../../controller';

// VertOutput.coord.xy range from (0, 0) to (1, 1). Used as uv sampling coord
// VertOutput.coord.zw range from (0, 0) to (screenWidth, screenHeight). Used as texture coord for textureLoad()

const vertexShader = /* wgsl */`

override screenWidth: f32;
override screenHeight: f32;

struct VertOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(linear, center) coord: vec4<f32>,
};

const coords = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0), // Bottom Left
  vec2<f32>( 1.0, -1.0), // Bottom Right
  vec2<f32>(-1.0,  1.0), // Top Left
  vec2<f32>( 1.0,  1.0)  // Top Right
);

@vertex
fn main(@builtin(vertex_index) index: u32) -> VertOutput {
  let coord = coords[index];
  let position = vec4<f32>(coord, 0.0, 1.0);
  let uv = coord * vec2<f32>(0.5, -0.5) + 0.5; // https://www.w3.org/TR/webgpu/#coordinate-systems
  let gbufferCoord = vec4<f32>(uv, uv * vec2<f32>(screenWidth, screenHeight));
  return VertOutput(
    position, gbufferCoord
  );
}

`;


const fragmentShader = /* wgsl */`

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

struct FragInput {
  @location(0) @interpolate(linear, center) coord: vec4<f32>,
};

struct FragOutput {
  @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: DirectionalLight;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var fluidDepthMap: texture_depth_2d;
@group(0) @binding(4) var fluidVolumeMap: texture_2d<f32>;

fn sRGBGammaEncode(color: vec3<f32>) -> vec3<f32> {
  return mix(
    color.rgb * 12.92,                                    // x <= 0.0031308
    pow(color.rgb, vec3<f32>(0.41666)) * 1.055 - 0.055,   // x >  0.0031308
    saturate(sign(color.rgb - 0.0031308))
  );
}

fn linearEyeDepth(z: f32) -> f32 {
  return 1.0 / dot(camera.params.zw, vec2<f32>(z, 1.0));
}

fn uvToEye(uv: vec2<f32>, z: f32) -> vec3<f32> {
  let depthEye = linearEyeDepth(z); // Z coordinate of eye space is negtive
  return vec3<f32>(
    (uv - 0.5) * camera.params.xy * depthEye,
    -depthEye
  );
}

fn getNormal(uv: vec2<f32>, z: f32) -> vec4<f32> {

  let posEye = uvToEye(uv, z);
  let ddx = dpdx(posEye);
  let ddy = dpdy(posEye);

  let normalEye = normalize(cross(ddy, ddx));
  return camera.viewMatrixInverse * vec4<f32>(normalEye, 0.0);

}

@fragment
fn main(input: FragInput) -> FragOutput {

  let frameCoord = vec2<u32>(floor(input.coord.zw));
  let z = textureLoad(fluidDepthMap, frameCoord, 0);
  if (z < 1e-4) { discard; }
  let fluidVolume = textureSample(fluidVolumeMap, linearSampler, input.coord.xy).r;
  let normalWorld = getNormal(input.coord.xy, z);

  // simple diffuse shading
  let NoL = saturate(dot(normalWorld.xyz, light.direction));
  let irradiance = NoL * light.color;
  let diffuse = (irradiance + 0.02) * 0.3183098861837907 *vec3<f32>(4,142,219)/256.0; // RECIPROCAL_PI

  return FragOutput(
    vec4<f32>(sRGBGammaEncode(diffuse), 1.0)
  );

}

`;


class Postprocess {

  protected vertexShaderCode: string;
  protected fragmentShaderCode: string;

  protected bindGroupLayout: GPUBindGroupLayout;
  protected bindGroup: GPUBindGroup;
  protected renderPipeline: GPURenderPipeline;
  protected renderBundle: GPURenderBundle;

  constructor() {

    this.vertexShaderCode = vertexShader;
    this.fragmentShaderCode = fragmentShader;

  }

  public initBindGroup(
    resource: { [x: string]: GPUBuffer | GPUTexture | GPUSampler }
  ) {

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'Particle Rendering Pipeline Bind Group Layout',
      entries: [{ // camera
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // light
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }, { // sampler
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' }
      }, { // fluid depth map
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'depth' }
      }, { // fluid volume map
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' }
      }]
    });

    this.bindGroup = device.createBindGroup({
      label: 'Particle Rendering Pipeline Bind Group',
      layout: this.bindGroupLayout,
      entries: [{ // camera
        binding: 0,
        resource: { buffer: resource.camera as GPUBuffer },
      }, { // light
        binding: 1,
        resource: { buffer: resource.directionalLight as GPUBuffer }
      }, { // sampler
        binding: 2,
        resource: resource.linearSampler as GPUSampler
      }, { // fluid depth map
        binding: 3,
        resource: (resource.fluidDepthMap as GPUTexture).createView()
      }, { // fluid volume map
        binding: 4,
        resource: (resource.fluidVolumeMap as GPUTexture).createView()
      }]
    });

  }

  public async initPipeline() {

    this.renderPipeline = await device.createRenderPipelineAsync({
      label: 'Postprocess Pipeline',
      layout: device.createPipelineLayout({ 
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      vertex: {
        module: device.createShaderModule({ code: this.vertexShaderCode }),
        constants: {
          screenWidth: canvasSize.width,
          screenHeight: canvasSize.height
        },
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({ label: '1', code: this.fragmentShaderCode }),
        entryPoint: 'main',
        targets: [{ 
          format: canvasFormat,
          writeMask: GPUColorWrite.RED | GPUColorWrite.GREEN | GPUColorWrite.BLUE,
          blend: {
            color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }
          }
        }]
      },
      primitive: {
        topology: 'triangle-strip',
        cullMode: 'none'
      }
    });

  }

  public initRenderBundle() {

    // depth pass
    let bundleEncoder = device.createRenderBundleEncoder({
      colorFormats: [ canvasFormat ]}
    );
    bundleEncoder.setPipeline(this.renderPipeline);
    bundleEncoder.setBindGroup(0, this.bindGroup);
    bundleEncoder.draw(4);
    this.renderBundle = bundleEncoder.finish();

  }

  public render(
    commandEncoder: GPUCommandEncoder,
    ctxTextureView: GPUTextureView
  ) {

    const renderPassEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: ctxTextureView,
        loadOp: 'load',
        storeOp: 'store'
      }]
    });
    renderPassEncoder.executeBundles([this.renderBundle]);
    renderPassEncoder.end();

  }

}

export { Postprocess };