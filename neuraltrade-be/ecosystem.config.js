module.exports = {
  apps: [
    {
      name: 'neuraltrade-be',
      script: 'dist/src/main.js',
      exec_mode: 'cluster',
      instances: 2,
      env_test: {
        NODE_ENV: 'test',
      },
    },
  ],

  deploy: {
    test: {
    //   user: 'worker',
    //   host: ['18.159.13.159'],
    //   key: '~/.ssh/id_ed25519',
      ref: 'origin/dev',
      repo: '',
      path: '',
      'pre-deploy-local': '',
      'pre-deploy': 'cp ../.env ./.env',
      'post-deploy':
        'npm install && npx prisma migrate deploy && npx prisma generate && npm run build && pm2 reload ecosystem.config.js --env test',
      'pre-setup': '',
      env: {
        DATABASE_URL: process.env.DATABASE_URL,
        DATA: process.env.DATA,
        NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
        PORT: process.env.PORT,

        AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
        AWS_REGION: process.env.AWS_REGION,
        AWS_S3_BUCKET_NAME: process.env.AWS_S3_BUCKET_NAME,

        REDIS_HOST: process.env.REDIS_HOST,
        REDIS_PORT: process.env.REDIS_PORT,
        REDIS_PASSWORD: process.env.REDIS_PASSWORD,
        REDIS_DB: process.env.REDIS_DB,
        REDIS_URL: process.env.REDIS_URL,
        REDIS_EXPIRES_IN_SECONDS: process.env.REDIS_EXPIRES_IN_SECONDS,
      },
    },
  },
};
