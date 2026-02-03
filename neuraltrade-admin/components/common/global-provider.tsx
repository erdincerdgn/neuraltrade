'use client';

import React, { ReactNode } from 'react';
import { ColorSchemeScript, createTheme, localStorageColorSchemeManager, MantineProvider } from '@mantine/core';
import { SessionProvider } from 'next-auth/react';
// eslint-disable-next-line import/extensions
import 'mantine-datatable/styles.layer.css';
import '@mantine/core/styles.css';

const theme = createTheme({
  primaryColor: 'blue',
  colors: {
    dark: [
      '#f6f7f8',
      '#bae0ff',
      '#91caff',
      '#69b1ff',
      '#4096ff',
      '#1677ff', 
      '#138aec', // 6 - your brand color
      '#0958d9', 
      '#003eb3', 
      '#101a22',
    ], // 10 shades required
  },
  fontFamily: 'Inter, sans-serif',
  defaultRadius: 'md',
  headings: { fontFamily: 'Inter, sans-serif' },

  other: {
    auth: {
      center: {
        mih: { base: '100dvh' },
        w: '100%',
        px: { base: 'md', sm: 0 },
      }
    },
    paper: {
      p: { base: 'lg', sm: 'xl' },
      shadow: 'md',
      w: { base: '100%', sm: 'auto' },
      maw: {
        base: '100%',
        xs: '22.5rem',  // 360
        sm: '30rem',    // 480
        md: '45rem',    // 720
        lg: '60rem',    // 960
      },
      miw: { xs: '20rem' },
    },
  },
},

);

const csManager = localStorageColorSchemeManager({ key: 'mantine-color-scheme' });

export default function GlobalProvider({ children }: { children: ReactNode }) {
  return (
    <SessionProvider>
      <ColorSchemeScript defaultColorScheme="auto" />
      <MantineProvider
        defaultColorScheme='auto'
        colorSchemeManager={csManager}
        
      >
        
        {children}
      </MantineProvider>
    </SessionProvider>
  );
}
