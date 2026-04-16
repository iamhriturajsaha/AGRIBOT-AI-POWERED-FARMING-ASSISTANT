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
        background: '#f8fafc', // slate-50
        panel: '#ffffff',
        panelBorder: '#e2e8f0', // slate-200
        agri: {
          green: '#166534', // green-800
          lightGreen: '#22c55e', // green-500
          earth: '#78350f', // amber-900
          leaf: '#4ade80', // green-400
        }
      },
    },
  },
  plugins: [],
}
