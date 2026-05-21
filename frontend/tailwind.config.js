/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
        display: ['Syne', 'sans-serif'],
      },
      colors: {
        surface: {
          0: '#0a0d12',
          1: '#0f1318',
          2: '#141920',
          3: '#1a2130',
        },
        accent: {
          DEFAULT: '#00e5ff',
          dim: '#00b8cc',
          glow: 'rgba(0,229,255,0.15)',
        },
        energy: '#f97316',
        success: '#22c55e',
        warn: '#eab308',
        danger: '#ef4444',
      },
      boxShadow: {
        glow: '0 0 20px rgba(0,229,255,0.2)',
        'glow-lg': '0 0 40px rgba(0,229,255,0.15)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slideUp 0.4s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
      },
      keyframes: {
        slideUp: {
          '0%': { transform: 'translateY(16px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}
