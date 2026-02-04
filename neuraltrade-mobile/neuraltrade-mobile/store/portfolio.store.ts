/**
 * Portfolio Store
 * ================
 * Zustand store for portfolio state management
 * Handles positions, P&L tracking, and order history
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ============================================
// TYPES
// ============================================

export interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  openedAt: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT' | 'STOP';
  quantity: number;
  price: number;
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  filledAt?: string;
  createdAt: string;
}

export interface PortfolioSummary {
  totalValue: number;
  totalCost: number;
  totalPnL: number;
  totalPnLPercent: number;
  dayPnL: number;
  dayPnLPercent: number;
  availableCash: number;
  marginUsed: number;
}

interface PortfolioState {
  // Portfolio data
  summary: PortfolioSummary | null;
  positions: Position[];
  orders: Order[];
  
  // Loading states
  isLoading: boolean;
  lastSync: number | null;
  
  // Actions
  setSummary: (summary: PortfolioSummary) => void;
  setPositions: (positions: Position[]) => void;
  updatePosition: (position: Position) => void;
  removePosition: (positionId: string) => void;
  addOrder: (order: Order) => void;
  updateOrder: (order: Order) => void;
  setLoading: (loading: boolean) => void;
  syncComplete: () => void;
  reset: () => void;
}

// ============================================
// INITIAL STATE
// ============================================

const initialSummary: PortfolioSummary = {
  totalValue: 0,
  totalCost: 0,
  totalPnL: 0,
  totalPnLPercent: 0,
  dayPnL: 0,
  dayPnLPercent: 0,
  availableCash: 0,
  marginUsed: 0,
};

// ============================================
// STORE (with persistence)
// ============================================

export const usePortfolioStore = create<PortfolioState>()(
  persist(
    (set, get) => ({
      // Initial state
      summary: null,
      positions: [],
      orders: [],
      isLoading: false,
      lastSync: null,

      setSummary: (summary) => set({ summary }),

      setPositions: (positions) => set({ positions }),

      updatePosition: (position) =>
        set((state) => ({
          positions: state.positions.map((p) =>
            p.id === position.id ? position : p
          ),
        })),

      removePosition: (positionId) =>
        set((state) => ({
          positions: state.positions.filter((p) => p.id !== positionId),
        })),

      addOrder: (order) =>
        set((state) => ({
          orders: [order, ...state.orders].slice(0, 100), // Keep last 100 orders
        })),

      updateOrder: (order) =>
        set((state) => ({
          orders: state.orders.map((o) => (o.id === order.id ? order : o)),
        })),

      setLoading: (isLoading) => set({ isLoading }),

      syncComplete: () => set({ lastSync: Date.now(), isLoading: false }),

      reset: () =>
        set({
          summary: null,
          positions: [],
          orders: [],
          isLoading: false,
          lastSync: null,
        }),
    }),
    {
      name: 'neuraltrade-portfolio',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        // Only persist summary and positions, not orders
        summary: state.summary,
        positions: state.positions,
        lastSync: state.lastSync,
      }),
    }
  )
);

// ============================================
// SELECTORS
// ============================================

export const selectSummary = (state: PortfolioState) => state.summary;
export const selectPositions = (state: PortfolioState) => state.positions;
export const selectOpenOrders = (state: PortfolioState) =>
  state.orders.filter((o) => o.status === 'PENDING');
export const selectTotalPnL = (state: PortfolioState) =>
  state.summary?.totalPnL ?? 0;
