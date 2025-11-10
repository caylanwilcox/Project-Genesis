/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: true, // Ignore type errors in backend services/scripts during Vercel build
  },
  eslint: {
    ignoreDuringBuilds: true, // Skip linting during build for faster deployment
  },
}

module.exports = nextConfig