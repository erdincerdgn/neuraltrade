/**
 * Login Screen
 * =============
 * User authentication with email/password
 * Supports Google and Apple Sign-In
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, Link } from 'expo-router';
import { Zap, Mail, Lock, Eye, EyeOff } from 'lucide-react-native';
import authService from '@/services/auth';
import { useAuthStore } from '@/store/auth.store';

// Social auth - conditionally imported
let useGoogleAuth: any;
let useAppleAuth: any;

try {
  const socialAuth = require('@/hooks/use-social-auth');
  useGoogleAuth = socialAuth.useGoogleAuth;
  useAppleAuth = socialAuth.useAppleAuth;
} catch (e) {
  // Social auth not available
  useGoogleAuth = () => ({ signInWithGoogle: async () => {}, isLoading: false, isReady: false });
  useAppleAuth = () => ({ signInWithApple: async () => {}, isLoading: false, isAvailable: false });
}

// ============================================
// THEME COLORS
// ============================================
const COLORS = {
  background: '#000000',
  surface: '#0D0D0D',
  border: '#1A1A1A',
  primary: '#00FF41',
  primaryDim: 'rgba(0, 255, 65, 0.1)',
  text: '#FFFFFF',
  textSecondary: '#888888',
  error: '#FF3B30',
  google: '#FFFFFF',
  apple: '#FFFFFF',
};

export default function LoginScreen() {
  const router = useRouter();
  const { setLoading, isLoading } = useAuthStore();
  
  // Form state
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState('');

  // Social auth hooks
  const { signInWithGoogle, isLoading: googleLoading } = useGoogleAuth();
  const { signInWithApple, isLoading: appleLoading, isAvailable: appleAvailable } = useAppleAuth();

  // ============================================
  // HANDLERS
  // ============================================

  const handleLogin = useCallback(async () => {
    // Validate
    if (!email.trim()) {
      setError('Email is required');
      return;
    }
    if (!password) {
      setError('Password is required');
      return;
    }

    setError('');
    setLoading(true);

    try {
      await authService.login({
        email: email.trim().toLowerCase(),
        password,
        rememberMe,
      });
      
      // Navigate to main app
      router.replace('/(tabs)');
    } catch (err: any) {
      const message = err.response?.data?.message || 'Login failed. Please try again.';
      setError(message);
      Alert.alert('Login Failed', message);
    } finally {
      setLoading(false);
    }
  }, [email, password, rememberMe, router, setLoading]);

  const handleGoogleLogin = useCallback(async () => {
    try {
      await signInWithGoogle();
      router.replace('/(tabs)');
    } catch (err: any) {
      Alert.alert('Google Sign-In Failed', err.message || 'Please try again');
    }
  }, [signInWithGoogle, router]);

  const handleAppleLogin = useCallback(async () => {
    try {
      await signInWithApple();
      router.replace('/(tabs)');
    } catch (err: any) {
      Alert.alert('Apple Sign-In Failed', err.message || 'Please try again');
    }
  }, [signInWithApple, router]);

  const isSubmitting = isLoading || googleLoading || appleLoading;

  // ============================================
  // RENDER
  // ============================================

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: COLORS.background }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, justifyContent: 'center', padding: 24 }}
          keyboardShouldPersistTaps="handled"
        >
          {/* Logo & Title */}
          <View style={{ alignItems: 'center', marginBottom: 48 }}>
            <View style={{
              width: 80,
              height: 80,
              borderRadius: 20,
              backgroundColor: COLORS.primaryDim,
              justifyContent: 'center',
              alignItems: 'center',
              marginBottom: 24,
            }}>
              <Zap size={40} color={COLORS.primary} />
            </View>
            <Text style={{
              fontSize: 32,
              fontWeight: '700',
              color: COLORS.text,
              marginBottom: 8,
            }}>
              NeuralTrade
            </Text>
            <Text style={{ fontSize: 16, color: COLORS.textSecondary }}>
              AI-Powered Trading Platform
            </Text>
          </View>

          {/* Error Message */}
          {error ? (
            <View style={{
              backgroundColor: 'rgba(255, 59, 48, 0.1)',
              padding: 12,
              borderRadius: 8,
              marginBottom: 16,
            }}>
              <Text style={{ color: COLORS.error, textAlign: 'center' }}>{error}</Text>
            </View>
          ) : null}

          {/* Email Input */}
          <View style={{
            flexDirection: 'row',
            alignItems: 'center',
            backgroundColor: COLORS.surface,
            borderRadius: 12,
            borderWidth: 1,
            borderColor: COLORS.border,
            paddingHorizontal: 16,
            marginBottom: 16,
          }}>
            <Mail size={20} color={COLORS.textSecondary} />
            <TextInput
              style={{
                flex: 1,
                paddingVertical: 16,
                paddingHorizontal: 12,
                fontSize: 16,
                color: COLORS.text,
              }}
              placeholder="Email"
              placeholderTextColor={COLORS.textSecondary}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
              value={email}
              onChangeText={setEmail}
              editable={!isSubmitting}
            />
          </View>

          {/* Password Input */}
          <View style={{
            flexDirection: 'row',
            alignItems: 'center',
            backgroundColor: COLORS.surface,
            borderRadius: 12,
            borderWidth: 1,
            borderColor: COLORS.border,
            paddingHorizontal: 16,
            marginBottom: 16,
          }}>
            <Lock size={20} color={COLORS.textSecondary} />
            <TextInput
              style={{
                flex: 1,
                paddingVertical: 16,
                paddingHorizontal: 12,
                fontSize: 16,
                color: COLORS.text,
              }}
              placeholder="Password"
              placeholderTextColor={COLORS.textSecondary}
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
              editable={!isSubmitting}
            />
            <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
              {showPassword ? (
                <EyeOff size={20} color={COLORS.textSecondary} />
              ) : (
                <Eye size={20} color={COLORS.textSecondary} />
              )}
            </TouchableOpacity>
          </View>

          {/* Remember Me & Forgot Password */}
          <View style={{
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 24,
          }}>
            <TouchableOpacity
              style={{ flexDirection: 'row', alignItems: 'center' }}
              onPress={() => setRememberMe(!rememberMe)}
            >
              <View style={{
                width: 20,
                height: 20,
                borderRadius: 4,
                borderWidth: 2,
                borderColor: rememberMe ? COLORS.primary : COLORS.border,
                backgroundColor: rememberMe ? COLORS.primary : 'transparent',
                justifyContent: 'center',
                alignItems: 'center',
                marginRight: 8,
              }}>
                {rememberMe && <Text style={{ color: '#000', fontSize: 12, fontWeight: 'bold' }}>✓</Text>}
              </View>
              <Text style={{ color: COLORS.textSecondary }}>Remember me</Text>
            </TouchableOpacity>
            <TouchableOpacity>
              <Text style={{ color: COLORS.primary }}>Forgot Password?</Text>
            </TouchableOpacity>
          </View>

          {/* Login Button */}
          <TouchableOpacity
            style={{
              backgroundColor: COLORS.primary,
              paddingVertical: 16,
              borderRadius: 12,
              alignItems: 'center',
              marginBottom: 24,
              opacity: isSubmitting ? 0.6 : 1,
            }}
            onPress={handleLogin}
            disabled={isSubmitting}
          >
            {isLoading ? (
              <ActivityIndicator color="#000" />
            ) : (
              <Text style={{ color: '#000', fontSize: 18, fontWeight: '600' }}>
                Sign In
              </Text>
            )}
          </TouchableOpacity>

          {/* Divider */}
          <View style={{
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 24,
          }}>
            <View style={{ flex: 1, height: 1, backgroundColor: COLORS.border }} />
            <Text style={{ color: COLORS.textSecondary, paddingHorizontal: 16 }}>
              or continue with
            </Text>
            <View style={{ flex: 1, height: 1, backgroundColor: COLORS.border }} />
          </View>

          {/* Social Login Buttons */}
          <View style={{ flexDirection: 'row', gap: 12, marginBottom: 32 }}>
            {/* Google Sign-In */}
            <TouchableOpacity
              style={{
                flex: 1,
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: COLORS.surface,
                paddingVertical: 14,
                borderRadius: 12,
                borderWidth: 1,
                borderColor: COLORS.border,
                opacity: googleLoading ? 0.6 : 1,
              }}
              onPress={handleGoogleLogin}
              disabled={isSubmitting}
            >
              {googleLoading ? (
                <ActivityIndicator color={COLORS.text} />
              ) : (
                <>
                  <Text style={{ fontSize: 20, marginRight: 8 }}>G</Text>
                  <Text style={{ color: COLORS.text, fontWeight: '500' }}>Google</Text>
                </>
              )}
            </TouchableOpacity>

            {/* Apple Sign-In */}
            {(Platform.OS === 'ios' || appleAvailable) && (
              <TouchableOpacity
                style={{
                  flex: 1,
                  flexDirection: 'row',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: COLORS.surface,
                  paddingVertical: 14,
                  borderRadius: 12,
                  borderWidth: 1,
                  borderColor: COLORS.border,
                  opacity: appleLoading ? 0.6 : 1,
                }}
                onPress={handleAppleLogin}
                disabled={isSubmitting}
              >
                {appleLoading ? (
                  <ActivityIndicator color={COLORS.text} />
                ) : (
                  <>
                    <Text style={{ fontSize: 20, marginRight: 8 }}></Text>
                    <Text style={{ color: COLORS.text, fontWeight: '500' }}>Apple</Text>
                  </>
                )}
              </TouchableOpacity>
            )}
          </View>

          {/* Register Link */}
          <View style={{ flexDirection: 'row', justifyContent: 'center' }}>
            <Text style={{ color: COLORS.textSecondary }}>Don't have an account? </Text>
            <TouchableOpacity onPress={() => router.push('/register' as any)}>
              <Text style={{ color: COLORS.primary, fontWeight: '600' }}>Sign Up</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
