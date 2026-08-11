# WebGPU Particle Fluid Simulation Baseline

## Introduction

1. Simulate fluid on WebGPU with Position Based Fluid (PBF).
2. Use implicit boundary condition (Volume Map) to handle the boundary.
3. Improving details with Surface Tension, Vorticity Confinement and XSPH.
4. Fully parallel neighbor search (Hash Grid, Exclusive Scan).
5. Visualize particles as instanced low-poly spheres with simple Lambert shading.

## Screen-Space Fluid Rendering

The renderer reconstructs a continuous-looking fluid surface entirely on the GPU:

1. Each particle is rasterized as a procedural six-vertex camera-facing billboard. The billboard is clipped to a circle and its fragment depth is evaluated analytically as a sphere.
2. A front-surface eye-space depth map and an additive thickness map are generated in separate passes.
3. The depth map is smoothed with the narrow-range filter from Truong and Yuksel, including dynamic screen-space kernel sizing, depth clamping, dynamic range adjustment, symmetric bias correction, two separable iterations, and a final 5x5 cleanup pass.
4. Surface normals are reconstructed from the filtered depth. The final full-screen pass uses environment-map reflection and refraction, Schlick Fresnel, thickness-dependent Beer-Lambert absorption, and directional-light highlights.

The main implementation is in `src/renderer/rawParticles/particles.ts` and its WGSL shaders are in `src/renderer/rawParticles/shader.ts`.

## Development

```bash
npm install
npm run dev
```

## References

MACKLIN M, MÜLLER M. Position based fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(4): 1-12.

KOSCHIER D, BENDER J, SOLENTHALER B, et al. Smoothed particle hydrodynamics techniques for the physics based simulation of fluids and solids[C/OL]//JAKOB W, PUPPO E. Eurographics 2019 - Tutorials. The Eurographics Association, 2019. DOI: 10.2312/egt.20191035.

BENDER J, KUGELSTADT T, WEILER M, et al. Volume maps: An implicit boundary representation for sph[C]//Proceedings of the 12th ACM SIGGRAPH Conference on Motion, Interaction and Games. 2019: 1-10.

AKINCI N, AKINCI G, TESCHNER M. Versatile surface tension and adhesion for sph fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(6): 1-8.

HARRIS M, SENGUPTA S, OWENS J D. Parallel prefix sum (scan) with cuda[J]. GPU gems, 2007, 3(39): 851-876.
