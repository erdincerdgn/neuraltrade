/**
 * ScreenWrapper Component
 * ========================
 * Reusable wrapper that applies:
 * - SafeAreaView for notch/system bar handling
 * - Neural-Dark theme background
 * - Consistent padding and layout
 * 
 * Usage:
 * ```tsx
 * <ScreenWrapper>
 *   <YourContent />
 * </ScreenWrapper>
 * ```
 */

import React from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { SafeAreaView, Edge } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { NEURAL_BLACK } from '@/constants/theme';

interface ScreenWrapperProps {
  children: React.ReactNode;
  /** Apply SafeArea edges - default: top, left, right */
  edges?: Edge[];
  /** Add horizontal padding - default: true */
  padded?: boolean;
  /** Custom background color - default: #000000 */
  backgroundColor?: string;
  /** Additional styles for the container */
  style?: ViewStyle;
  /** Show status bar - default: true */
  showStatusBar?: boolean;
  /** Status bar style - default: light */
  statusBarStyle?: 'auto' | 'inverted' | 'light' | 'dark';
  /** Make content scrollable */
  scrollable?: boolean;
}

export function ScreenWrapper({
  children,
  edges = ['top', 'left', 'right'],
  padded = true,
  backgroundColor = NEURAL_BLACK,
  style,
  showStatusBar = true,
  statusBarStyle = 'light',
}: ScreenWrapperProps) {
  return (
    <SafeAreaView
      edges={edges}
      style={[styles.container, { backgroundColor }, style]}
    >
      {showStatusBar && <StatusBar style={statusBarStyle} />}
      <View style={[styles.content, padded && styles.padded]}>
        {children}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  padded: {
    paddingHorizontal: 16,
  },
});

export default ScreenWrapper;
