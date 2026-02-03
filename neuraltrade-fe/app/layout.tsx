// import '@mantine/core/styles.css';
// import '@/styles/globals.css';
// import '@mantine/dropzone/styles.css';
// import '@mantine/notifications/styles.css';
// import '@mantine/tiptap/styles.css';
// import '@mantine/carousel/styles.css';

// import React from 'react';
// import { ColorSchemeScript, MantineProvider } from '@mantine/core';
// import { Notifications } from '@mantine/notifications';

// import * as Sentry from '@sentry/nextjs';

// import AlertComponent from '@/components/common/alert';
// import { theme } from '@/theme';
// import { Metadata } from 'next';
// import Modals from '@/components/common/modals';
// import Provider from '@/components/ui/provider/provider';

// export default function RootLayout({ children }: { children: React.ReactNode }) {

//   return (
//     <html lang="en">
//       <head>
//         <ColorSchemeScript />
//         <link rel="icon" href="/images/icon.png" sizes="any" />
//         <meta
//           name="viewport"
//           content="minimum-scale=1, initial-scale=1, width=device-width, user-scalable=no"
//         />
//       </head>
//       <body>
//         <MantineProvider theme={theme}>
//           <Notifications />
//           <Provider>
//             {children}
//             <Modals />
//           </Provider>
//           <AlertComponent />
//         </MantineProvider>
//       </body>
//     </html>
//   );
// }

// export function generateMetadata(): Metadata {
//   return {
//     title: "BuildLink Human Resource Solutions",
//     description: "BuildLink - human resource solutions for construction companies.",
//     other: {
//       ...Sentry.getTraceData()
//     }
//   };
// }