'use client';

import { MantineProvider } from '@mantine/core';
import GlobalProvider from '@/components/common/global-provider';
import SessionLayout from './session-layout';

export function ClientLayout({ children }: { children: React.ReactNode }) {
  return (
    <MantineProvider>
      <GlobalProvider>
        <SessionLayout>{children}</SessionLayout>
      </GlobalProvider>
    </MantineProvider>
  );
}
