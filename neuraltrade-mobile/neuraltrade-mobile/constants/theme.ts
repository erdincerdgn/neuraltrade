/**
 * NeuralTrade Theme Configuration
 * Neural-Dark Theme with Matrix Green accents
 * Aligned with Layer 1: Presentation & Client Layer architecture
 */

import { Platform } from 'react-native';

// ============================================
// NEURAL-DARK THEME COLORS
// ============================================

/** Primary accent color - Matrix Green */
export const MATRIX_GREEN = '#00FF41';
export const MATRIX_GREEN_DIM = '#00CC34';
export const MATRIX_GREEN_BRIGHT = '#33FF66';
export const MATRIX_GREEN_GLOW = 'rgba(0, 255, 65, 0.3)';

/** Background colors */
export const NEURAL_BLACK = '#000000';
export const NEURAL_DARK = '#0A0A0A';
export const NEURAL_GRAY = '#1A1A1A';
export const NEURAL_BORDER = '#2A2A2A';

/** Status colors for trading signals */
export const STATUS_BULLISH = '#00FF41';
export const STATUS_BEARISH = '#FF3B30';
export const STATUS_NEUTRAL = '#FFD60A';
export const STATUS_INFO = '#0A84FF';

export const Colors = {
  light: {
    text: '#11181C',
    textSecondary: '#687076',
    background: '#FFFFFF',
    backgroundSecondary: '#F5F5F5',
    tint: MATRIX_GREEN_DIM,
    icon: '#687076',
    tabIconDefault: '#687076',
    tabIconSelected: MATRIX_GREEN_DIM,
    border: '#E0E0E0',
    card: '#FFFFFF',
  },
  dark: {
    text: '#FFFFFF',
    textSecondary: '#9BA1A6',
    background: NEURAL_BLACK,
    backgroundSecondary: NEURAL_DARK,
    tint: MATRIX_GREEN,
    icon: '#6B7280',
    tabIconDefault: '#6B7280',
    tabIconSelected: MATRIX_GREEN,
    border: NEURAL_BORDER,
    card: NEURAL_GRAY,
  },
};

// ============================================
// SPACING & LAYOUT
// ============================================

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const BorderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
};

// ============================================
// TYPOGRAPHY
// ============================================

export const Fonts = Platform.select({
  ios: {
    /** iOS `UIFontDescriptorSystemDesignDefault` */
    sans: 'system-ui',
    /** iOS `UIFontDescriptorSystemDesignSerif` */
    serif: 'ui-serif',
    /** iOS `UIFontDescriptorSystemDesignRounded` */
    rounded: 'ui-rounded',
    /** iOS `UIFontDescriptorSystemDesignMonospaced` */
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
  web: {
    sans: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    serif: "Georgia, 'Times New Roman', serif",
    rounded: "'SF Pro Rounded', 'Hiragino Maru Gothic ProN', Meiryo, 'MS PGothic', sans-serif",
    mono: "SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
  },
});
