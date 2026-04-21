import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  base: command === 'serve' ? '/' : '/WebGPU-Fluid-Simulation/',
  build: {
    target: 'esnext',
    outDir: 'docs'
  }
}));
