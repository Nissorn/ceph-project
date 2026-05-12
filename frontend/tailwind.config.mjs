/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
	theme: {
		extend: {
      colors: {
        singapodent: {
          primary: '#0c2340',
          accent: '#f28c28',
        }
      }
    },
	},
	plugins: [],
}
