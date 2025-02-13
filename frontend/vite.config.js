import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: 'frontend', // ✅ Explicitly set frontend as the root
  plugins: [react()],
  build: {
    outDir: '../dist', // ✅ Ensures the build output is placed correctly
  },
});
