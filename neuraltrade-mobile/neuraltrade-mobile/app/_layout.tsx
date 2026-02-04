/**
 * Root Layout
 * ============
 * NeuralTrade Mobile App - Layer 1: Presentation & Client Layer
 * 
 * Configures:
 * - SafeAreaProvider for notch/system bar handling
 * - React Query for data fetching
 * - Dark theme as default
 * - Authentication protection
 * - Global socket connection
 */

import { useEffect, useState, useCallback } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { DarkTheme, ThemeProvider } from '@react-navigation/native';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import 'react-native-reanimated';

import '../global.css';
import { NEURAL_BLACK, MATRIX_GREEN, NEURAL_GRAY, NEURAL_BORDER } from '@/constants/theme';
import { useAuthStore } from '@/store/auth.store';
import { authService } from '@/services/auth';

// ============================================
// REACT QUERY CLIENT
// ============================================

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

// ============================================
// NEURAL-DARK THEME
// ============================================

const NeuralDarkTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    primary: MATRIX_GREEN,
    background: NEURAL_BLACK,
    card: NEURAL_GRAY,
    text: '#FFFFFF',
    border: NEURAL_BORDER,
    notification: MATRIX_GREEN,
  },
};

// ============================================
// AUTH PROTECTION HOOK
// ============================================

function useProtectedRoute() {
  const segments = useSegments();
  const router = useRouter();
  const { isAuthenticated, isInitialized } = useAuthStore();

  useEffect(() => {
    if (!isInitialized) return;

    const firstSegment = segments[0] as string;
    const inAuthGroup = firstSegment === '(auth)';

    if (!isAuthenticated && !inAuthGroup) {
      // Redirect to login if not authenticated
      router.replace('/login' as any);
    } else if (isAuthenticated && inAuthGroup) {
      // Redirect to main app if already authenticated
      router.replace('/' as any);
    }
  }, [isAuthenticated, segments, isInitialized, router]);
}

// ============================================
// ROOT LAYOUT
// ============================================

export const unstable_settings = {
  initialRouteName: '(auth)',
};

export default function RootLayout() {
  const [isReady, setIsReady] = useState(false);
  const { setInitialized } = useAuthStore();

  // Initialize auth on app start
  useEffect(() => {
    const initAuth = async () => {
      try {
        await authService.initialize();
      } catch (error) {
        console.warn('[Auth] Initialization failed:', error);
        setInitialized(true);
      } finally {
        setIsReady(true);
      }
    };

    initAuth();
  }, [setInitialized]);

  // Protect routes based on auth state
  useProtectedRoute();

  // Show loading screen while initializing
  if (!isReady) {
    return (
      <View style={{ 
        flex: 1, 
        backgroundColor: NEURAL_BLACK, 
        justifyContent: 'center', 
        alignItems: 'center' 
      }}>
        <ActivityIndicator size="large" color={MATRIX_GREEN} />
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider value={NeuralDarkTheme}>
            <Stack
              screenOptions={{
                headerShown: false,
                contentStyle: { backgroundColor: NEURAL_BLACK },
                animation: 'fade',
              }}
            >
              <Stack.Screen name="(auth)" options={{ headerShown: false }} />
              <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
              <Stack.Screen
                name="modal"
                options={{
                  presentation: 'modal',
                  title: 'Details',
                  headerShown: true,
                  headerStyle: { backgroundColor: NEURAL_GRAY },
                  headerTintColor: MATRIX_GREEN,
                }}
              />
            </Stack>
            <StatusBar style="light" backgroundColor={NEURAL_BLACK} />
          </ThemeProvider>
        </QueryClientProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
