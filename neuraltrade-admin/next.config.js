const path = require('path');
const nextBuildId = require('next-build-id');

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

module.exports = withBundleAnalyzer({
  generateBuildId: () => nextBuildId({ dir: __dirname }),
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.alias['@'] = path.join(__dirname, '');
    }
    return config;
  },
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
    NEXT_PUBLIC_URL: process.env.NEXT_PUBLIC_URL,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  experimental: {
    esmExternals: false,
    optimizePackageImports: ['@mantine/core', '@mantine/hooks'],
  },
  sassOptions: {
    prependData: `@import "./_mantine.scss";`,
  },
});
