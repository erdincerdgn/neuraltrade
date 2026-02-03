'use client';

import { ReactNode } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { AuthStatus } from '@/types/common';

interface Props {
  children: ReactNode;
}

export default function LoginLayout({ children }: Props) {
  const { status } = useSession();
  const route = useRouter();

  // If already authenticated, redirect to dashboard
  if (status === AuthStatus.Authenticated) {
    route.replace('/dashboard');
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed', 
        inset: 0,         
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 0,
        margin: 0,
        boxSizing: 'border-box',
      }}
    >
      {children}
    </div>
  );
}
