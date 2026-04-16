import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      devOptions: { enabled: true },
      manifest: {
        name: 'AgriBot Dashboard',
        short_name: 'AgriBot',
        description: 'AI-Powered Farming Assistant',
        theme_color: '#050505',
        background_color: '#050505',
        icons: [
          {
            src: 'https://cdn-icons-png.flaticon.com/512/188/188333.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ],
  server: {
    port: 3000,
  }
})
