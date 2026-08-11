import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  base: './',
  build: {
    target: 'esnext',
    outDir: 'build'
  }
}));
