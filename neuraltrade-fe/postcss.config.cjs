module.exports = {
  plugins: {
    'postcss-preset-mantine': {},
    'postcss-nested': {},
    'postcss-simple-vars': {
      variables: {
        // Breakpoint Variables
        'mantine-breakpoint-xs': '36em',
        'mantine-breakpoint-sm': '48em',
        'mantine-breakpoint-md': '62em',
        'mantine-breakpoint-lg': '75em',
        'mantine-breakpoint-xl': '88em',
      },
    },
  },
};
