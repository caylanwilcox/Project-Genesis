/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx,ts,tsx,mdx}",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0B0E11',
        'dark-secondary': '#181A20',
        'dark-tertiary': '#2B3139',
        'green-500': '#0ECB81',
        'red-500': '#F6465D',
        'yellow-500': '#FCD535',
      }
    },
  },
  plugins: [],
}

