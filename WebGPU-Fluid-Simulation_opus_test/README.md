# WebGPU Particle Fluid Simulation

## Introduction

1. Simulate fluid on WebGPU with Position Based Fluid (PBF).
2. Use implicit boundary condition (Volume Map) to handle the boundary.
3. Improving details with Surface Tension, Vorticity Confinement and XSPH.
4. Fully parallel neighbor search (Hash Grid, Exclusive Scan).
5. Screen-space fluid rendering with a narrow-range depth filter, environment reflection and refraction.

## Screen-Space Fluid Rendering

The fluid surface is reconstructed in screen space, without generating a mesh. Each frame, after the scene has been rendered:

1. **Thickness pass** — the particles are rasterized as sphere imposters and their chord lengths are accumulated with additive blending.
2. **Depth pass** — the same imposters write the eye space depth of the sphere surface, producing the raw depth map.
3. **Filter passes** — the depth map is smoothed by the narrow-range filter of Truong & Yuksel, applied as a separable approximation (1D passes with alternating directions) followed by a small 2D clean-up pass. Unlike a bilateral filter, depth values outside the narrow range are clamped rather than ignored, which keeps discontinuities such as isolated droplets sharp while still producing a smooth, curved surface. The filter also uses bias correction and a dynamic depth range so that flat surfaces stay smooth at grazing angles.
4. **Composite pass** — the surface position and normal are rebuilt from the filtered depth map and shaded with Fresnel-weighted environment reflection, refraction, and Beer-Lambert absorption driven by the thickness map.

Particles are drawn as billboards rather than sphere meshes: the perspective projection of a sphere is approximated by a screen space circle, and the sphere surface is reconstructed per fragment. This costs 4 vertices per particle instead of a full sphere mesh.

Both imposter passes use the scene depth buffer as their depth attachment, so particles hidden behind scene geometry are discarded before they can reach the depth map and be smeared across the occluder boundary by the filter.

The `Fluid Render Options` panel exposes the filter parameters (in multiples of the imposter radius, following the paper's `delta = 10r`, `mu = r`, `sigma = 0.7r`), the shading parameters, and a switch back to the raw particle view.

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

TRUONG N, YUKSEL C. A narrow-range filter for screen-space fluid rendering[C]//Proceedings of the ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games (i3D). 2018.

HARRIS M, SENGUPTA S, OWENS J D. Parallel prefix sum (scan) with cuda[J]. GPU gems, 2007, 3(39): 851-876.
