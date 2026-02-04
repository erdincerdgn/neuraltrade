/**
 * Social Authentication Hooks
 * ============================
 * Google and Apple Sign-In integration using Expo Auth Session
 */

import { useState, useCallback, useEffect } from 'react';
import { Platform } from 'react-native';
import * as WebBrowser from 'expo-web-browser';
import * as Google from 'expo-auth-session/providers/google';
import * as AppleAuthentication from 'expo-apple-authentication';
import authService from '@/services/auth';

// Complete web browser auth session
WebBrowser.maybeCompleteAuthSession();

// ============================================
// GOOGLE AUTH CONFIG
// ============================================
// Note: Replace with your actual Google OAuth client IDs
const GOOGLE_CONFIG = {
  iosClientId: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
  androidClientId: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
  webClientId: process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
};

// ============================================
// GOOGLE SIGN-IN HOOK
// ============================================

export function useGoogleAuth() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [request, response, promptAsync] = Google.useAuthRequest({
    iosClientId: GOOGLE_CONFIG.iosClientId,
    androidClientId: GOOGLE_CONFIG.androidClientId,
    webClientId: GOOGLE_CONFIG.webClientId,
  });

  // Handle Google auth response
  useEffect(() => {
    if (response?.type === 'success') {
      const { authentication } = response;
      if (authentication?.idToken) {
        handleGoogleToken(authentication.idToken);
      }
    } else if (response?.type === 'error') {
      setError('Google sign-in was cancelled or failed');
      setIsLoading(false);
    }
  }, [response]);

  const handleGoogleToken = async (idToken: string) => {
    try {
      await authService.socialAuth({
        provider: 'google',
        idToken,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to authenticate with Google');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const signInWithGoogle = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await promptAsync();
      if (result.type !== 'success') {
        setIsLoading(false);
        if (result.type === 'cancel') {
          throw new Error('Google sign-in was cancelled');
        }
        throw new Error('Google sign-in failed');
      }
    } catch (err: any) {
      setIsLoading(false);
      setError(err.message);
      throw err;
    }
  }, [promptAsync]);

  return {
    signInWithGoogle,
    isLoading,
    error,
    isReady: !!request,
  };
}

// ============================================
// APPLE SIGN-IN HOOK
// ============================================

export function useAppleAuth() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAvailable, setIsAvailable] = useState(false);

  // Check if Apple Sign-In is available
  useEffect(() => {
    const checkAvailability = async () => {
      if (Platform.OS === 'ios') {
        const available = await AppleAuthentication.isAvailableAsync();
        setIsAvailable(available);
      }
    };
    checkAvailability();
  }, []);

  const signInWithApple = useCallback(async () => {
    if (Platform.OS !== 'ios') {
      throw new Error('Apple Sign-In is only available on iOS');
    }

    setIsLoading(true);
    setError(null);

    try {
      const credential = await AppleAuthentication.signInAsync({
        requestedScopes: [
          AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
          AppleAuthentication.AppleAuthenticationScope.EMAIL,
        ],
      });

      // Get user info from credential
      const { identityToken, email, fullName } = credential;

      if (!identityToken) {
        throw new Error('No identity token received from Apple');
      }

      // Build name from fullName components
      let name = '';
      if (fullName) {
        const nameParts = [fullName.givenName, fullName.familyName].filter(Boolean);
        name = nameParts.join(' ');
      }

      // Send to backend
      await authService.socialAuth({
        provider: 'apple',
        idToken: identityToken,
        email: email || undefined,
        name: name || undefined,
      });
    } catch (err: any) {
      if (err.code === 'ERR_REQUEST_CANCELED') {
        setError('Apple sign-in was cancelled');
      } else {
        setError(err.message || 'Failed to authenticate with Apple');
      }
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    signInWithApple,
    isLoading,
    error,
    isAvailable,
  };
}

// ============================================
// AUTH STATE HOOK
// ============================================

export function useAuthState() {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const initAuth = async () => {
      try {
        await authService.initialize();
      } catch (error) {
        console.warn('[Auth] Initialization failed:', error);
      } finally {
        setIsReady(true);
      }
    };

    initAuth();
  }, []);

  return { isReady };
}
