/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: proxy /api to the Spring Boot API so the SPA is same-origin (no CORS, SSE works). Prod: nginx does the same.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: 'http://localhost:8080', changeOrigin: true } } },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.ts'], globals: true, css: false },
})
