# WebGPU Fluid Simulation

Try: https://linzhouli.github.io/WebGPU-Fluid-Simulation/

Thesis: [LinzhouLi/UndergraduateThesis (github.com)](https://github.com/LinzhouLi/UndergraduateThesis)

## Introduction

1. Simulate fluid on WebGPU with Position Based Fluid (PBF).
2. Use implicit boundary condition (Volume Map) to handle the boundary.
3. Improving details with Surface Tension, Vorticity Confinement and XSPH.
4. Fully parallel neighbor search (Hash Grid, Exclusive Scan).
5. Real-time rendering fluids in screen space, smooth the depth map with Narrow-Range Filter.

## Real-Time Demo

![result](img/result.jpg)

## Simulation Results

![demo](img/demo.jpg)

## References

MACKLIN M, MÜLLER M. Position based fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(4): 1-12.

KOSCHIER D, BENDER J, SOLENTHALER B, et al. Smoothed particle hydrodynamics techniques for the physics based simulation of fluids and solids[C/OL]//JAKOB W, PUPPO E. Eurographics 2019 - Tutorials. The Eurographics Association, 2019. DOI: 10.2312/egt.20191035.

BENDER J, KUGELSTADT T, WEILER M, et al. Volume maps: An implicit boundary representation for sph[C]//Proceedings of the 12th ACM SIGGRAPH Conference on Motion, Interaction and Games. 2019: 1-10.

AKINCI N, AKINCI G, TESCHNER M. Versatile surface tension and adhesion for sph fluids[J]. ACM Transactions on Graphics (TOG), 2013, 32(6): 1-8.

TRUONG N, YUKSEL C. A narrow-range filter for screen-space fluid rendering[J]. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 2018, 1(1): 1-15.

HARRIS M, SENGUPTA S, OWENS J D. Parallel prefix sum (scan) with cuda[J]. GPU gems, 2007, 3(39): 851-876.
