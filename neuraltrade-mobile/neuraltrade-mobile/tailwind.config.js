/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,jsx,ts,tsx}',
    './components/**/*.{js,jsx,ts,tsx}',
  ],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        // Neural-Dark Theme Colors
        neural: {
          black: '#000000',
          dark: '#0A0A0A',
          gray: '#1A1A1A',
          border: '#2A2A2A',
        },
        matrix: {
          green: '#00FF41',
          'green-dim': '#00CC34',
          'green-bright': '#33FF66',
          'green-glow': 'rgba(0, 255, 65, 0.3)',
        },
        status: {
          bullish: '#00FF41',
          bearish: '#FF3B30',
          neutral: '#FFD60A',
          info: '#0A84FF',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
