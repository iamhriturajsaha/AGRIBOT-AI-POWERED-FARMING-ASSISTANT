/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Outfit', 'sans-serif'],
      },
      colors: {
        background: '#0A0A0B',
        panel: 'rgba(255, 255, 255, 0.03)',
        panelBorder: 'rgba(255, 255, 255, 0.08)',
        neon: {
          green: '#00ffff', // Cyan mapping
          blue: '#f42b8e', // Pink mapping
          pink: '#ffb703', // Orange mapping
          purple: '#8e2de2'
        }
      },
      animation: {
        'glow-pulse': 'glow 3s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 10px rgba(0,230,118, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(0,230,118, 0.6)' }
        }
      }
    },
  },
  plugins: [],
}
