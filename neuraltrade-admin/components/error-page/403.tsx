'use client';

import React from 'react';
import { signOut } from 'next-auth/react';

export default function UnauthorizedAccess() {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        flexDirection: 'column',
      }}
    >
      Click to Login
      <br />
      <br />
      <button
        type="button"
        onClick={() => {
          signOut({
            callbackUrl: `${process.env.NEXT_PUBLIC_URL}/login`,
          });
        }}
      >
        Login
      </button>
    </div>
  );
}
