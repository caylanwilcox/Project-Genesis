/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: true, // Temporarily ignore TS errors in scripts during Vercel build
  },
  eslint: {
    ignoreDuringBuilds: true, // Temporarily ignore ESLint during build
  },
}

module.exports = nextConfig