/**
 * Hooks Index
 * ============
 * Barrel export for all custom hooks
 */

export { useColorScheme } from './use-color-scheme';
export { useThemeColor } from './use-theme-color';
export { useSocket } from './use-socket';
export { useMarketQuote, useMarketHistory } from './use-market-data';
export type { MarketQuote, MarketHistory } from './use-market-data';

// Auth hooks
export { useGoogleAuth, useAppleAuth, useAuthState } from './use-social-auth';
