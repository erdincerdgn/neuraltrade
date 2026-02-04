/**
 * Market Store
 * =============
 * Zustand store for real-time market data state
 * Optimized for high-frequency updates from Redis Pub/Sub
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { MarketTick } from '@/services/socket';

// ============================================
// TYPES
// ============================================

export interface WatchlistItem {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  lastUpdate: number;
}

interface MarketState {
  // Real-time ticks (symbol -> tick data)
  ticks: Map<string, MarketTick>;
  
  // User's watchlist
  watchlist: WatchlistItem[];
  
  // Currently selected symbol
  activeSymbol: string | null;
  
  // Connection status
  isConnected: boolean;
  lastHeartbeat: number | null;
  
  // Actions
  updateTick: (tick: MarketTick) => void;
  setActiveSymbol: (symbol: string | null) => void;
  addToWatchlist: (item: WatchlistItem) => void;
  removeFromWatchlist: (symbol: string) => void;
  setConnectionStatus: (connected: boolean) => void;
  updateHeartbeat: () => void;
  reset: () => void;
}

// ============================================
// STORE
// ============================================

export const useMarketStore = create<MarketState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    ticks: new Map(),
    watchlist: [],
    activeSymbol: null,
    isConnected: false,
    lastHeartbeat: null,

    // Update a single tick (memoized for performance)
    updateTick: (tick: MarketTick) => {
      set((state) => {
        const newTicks = new Map(state.ticks);
        newTicks.set(tick.symbol, tick);
        
        // Also update watchlist if symbol is in it
        const watchlistIndex = state.watchlist.findIndex(
          (item) => item.symbol === tick.symbol
        );
        
        if (watchlistIndex !== -1) {
          const newWatchlist = [...state.watchlist];
          newWatchlist[watchlistIndex] = {
            ...newWatchlist[watchlistIndex],
            price: tick.price,
            change: tick.change,
            changePercent: tick.changePercent,
            volume: tick.volume,
            lastUpdate: tick.timestamp,
          };
          return { ticks: newTicks, watchlist: newWatchlist };
        }
        
        return { ticks: newTicks };
      });
    },

    setActiveSymbol: (symbol) => set({ activeSymbol: symbol }),

    addToWatchlist: (item) =>
      set((state) => ({
        watchlist: state.watchlist.some((w) => w.symbol === item.symbol)
          ? state.watchlist
          : [...state.watchlist, item],
      })),

    removeFromWatchlist: (symbol) =>
      set((state) => ({
        watchlist: state.watchlist.filter((item) => item.symbol !== symbol),
      })),

    setConnectionStatus: (connected) => set({ isConnected: connected }),

    updateHeartbeat: () => set({ lastHeartbeat: Date.now() }),

    reset: () =>
      set({
        ticks: new Map(),
        watchlist: [],
        activeSymbol: null,
        isConnected: false,
        lastHeartbeat: null,
      }),
  }))
);

// ============================================
// SELECTORS (for optimized subscriptions)
// ============================================

export const selectTick = (symbol: string) => (state: MarketState) =>
  state.ticks.get(symbol);

export const selectActiveSymbolTick = (state: MarketState) =>
  state.activeSymbol ? state.ticks.get(state.activeSymbol) : null;

export const selectWatchlist = (state: MarketState) => state.watchlist;

export const selectIsConnected = (state: MarketState) => state.isConnected;
