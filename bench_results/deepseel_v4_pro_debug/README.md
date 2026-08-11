# WebGPU Particle Fluid Simulation Baseline

## Introduction

1. Simulate fluid on WebGPU with Position Based Fluid (PBF).
2. Use implicit boundary condition (Volume Map) to handle the boundary.
3. Improving details with Surface Tension, Vorticity Confinement and XSPH.
4. Fully parallel neighbor search (Hash Grid, Exclusive Scan).
5. Visualize particles as instanced low-poly spheres with simple Lambert shading.

## Rendering Baseline

The current renderer draws one sphere mesh instance per simulated particle. It does not reconstruct a continuous fluid surface and does not use screen-space depth filtering, thickness estimation, reflection, or refraction.

Each particle shares the same sphere geometry and reads its position directly from the simulator's GPU storage buffer. This provides a straightforward reference implementation for evaluating rendering quality and performance improvements.

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
