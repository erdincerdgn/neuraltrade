module.exports = {
    apps: [
        {
            name: 'neuraltrade-admin',
            script: "node_modules/next/dist/bin/next",
            args: "start",
            cwd: "./",
            instances: "max",
            exec_mode: "cluster",
            env_test: {
                NODE_ENV: 'test',
            },
        },
    ],

    
};
