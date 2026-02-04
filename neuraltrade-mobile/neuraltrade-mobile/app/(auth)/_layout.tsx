/**
 * Auth Layout
 * ============
 * Route group layout for authentication screens
 * Provides unauthenticated-only access (redirects if logged in)
 */

import { Stack, useRouter } from 'expo-router';
import { useEffect } from 'react';
import { useAuthStore } from '@/store/auth.store';

export default function AuthLayout() {
  const router = useRouter();
  const { isAuthenticated, isInitialized } = useAuthStore();

  useEffect(() => {
    // If user is already authenticated, redirect to main app
    if (isInitialized && isAuthenticated) {
      router.replace('/(tabs)');
    }
  }, [isAuthenticated, isInitialized, router]);

  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: '#000000' },
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen name="login" />
      <Stack.Screen name="register" />
    </Stack>
  );
}
