import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: process.env.NODE_ENV === 'development' ? {
      '/auth': {
        target: 'http://localhost:5000', // Only for local dev
        changeOrigin: true,
        secure: false,
      },
    } : undefined,  // Proxy only in dev mode
  },
  build: {
    outDir: 'dist', // Ensure output directory is correct
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
