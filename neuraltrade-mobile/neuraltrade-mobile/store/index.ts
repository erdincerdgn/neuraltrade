/**
 * Store Index
 * ============
 * Barrel export for all Zustand stores
 */

export { useAuthStore, selectUser, selectIsAuthenticated, selectAccessToken, selectIsLoading, selectSubscription } from './auth.store';
export type { User, Subscription, AuthState } from './auth.store';

export { useMarketStore, selectTick, selectActiveSymbolTick, selectWatchlist, selectIsConnected } from './market.store';
export type { WatchlistItem } from './market.store';

export { usePortfolioStore, selectSummary, selectPositions, selectOpenOrders, selectTotalPnL } from './portfolio.store';
export type { Position, Order, PortfolioSummary } from './portfolio.store';

export { useSignalsStore, selectFilteredSignals, selectLatestSignal, selectAgentThoughts, selectSignalsBySymbol } from './signals.store';
