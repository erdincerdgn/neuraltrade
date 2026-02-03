// app/layout.tsx
import { ColorSchemeScript } from '@mantine/core';
import '../globals.scss';
import { ClientLayout } from '@/components/layouts/client-layout';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr" style={{ height: '100%' }}>
      <head>
        <ColorSchemeScript defaultColorScheme="auto"/>
        <link rel="shortcut icon" href="/icons/favicon.ico" />
        <meta
          name="viewport"
          content="minimum-scale=1, initial-scale=1, width=device-width, user-scalable=no"
        />
        <title>NeuralTrade</title>
      </head>
      <body style={{ height: '100%' }}>
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
