// eslint-disable-next-line import/no-unresolved
require('dotenv').config();
const { nextStart } = require('./node_modules/next/dist/bin/next');

nextStart({
    port: process.env.PORT || 3001,
}).catch((err) => {
    console.error('Error starting the application:', err);
    process.exit(1);
});
