import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: '.',  // Ensure Vite looks in the current folder
  build: {
    outDir: 'dist',
    emptyOutDir: true,  // Ensure the output directory is cleared
  },
  plugins: [react()],
});
