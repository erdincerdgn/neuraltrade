/**
 * AI Signals Store
 * ==================
 * Zustand store for AI-generated trading signals
 * Receives signals from Layer 4 Python AI Engine via WebSocket
 */

import { create } from 'zustand';
import type { AISignal, AgentThought } from '@/services/socket';

// ============================================
// TYPES
// ============================================

interface SignalsState {
  // Active signals
  signals: AISignal[];
  
  // Agent thoughts (terminal view)
  agentThoughts: AgentThought[];
  
  // Filters
  filterAction: 'ALL' | 'BUY' | 'SELL' | 'HOLD';
  minConfidence: number;
  
  // Actions
  addSignal: (signal: AISignal) => void;
  updateSignal: (signal: AISignal) => void;
  addAgentThought: (thought: AgentThought) => void;
  clearAgentThoughts: () => void;
  setFilterAction: (action: 'ALL' | 'BUY' | 'SELL' | 'HOLD') => void;
  setMinConfidence: (confidence: number) => void;
  clearSignals: () => void;
}

// ============================================
// STORE
// ============================================

export const useSignalsStore = create<SignalsState>((set) => ({
  // Initial state
  signals: [],
  agentThoughts: [],
  filterAction: 'ALL',
  minConfidence: 0,

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 50), // Keep last 50 signals
    })),

  updateSignal: (signal) =>
    set((state) => ({
      signals: state.signals.map((s) => (s.id === signal.id ? signal : s)),
    })),

  addAgentThought: (thought) =>
    set((state) => ({
      agentThoughts: [...state.agentThoughts, thought].slice(-100), // Keep last 100 thoughts
    })),

  clearAgentThoughts: () => set({ agentThoughts: [] }),

  setFilterAction: (filterAction) => set({ filterAction }),

  setMinConfidence: (minConfidence) => set({ minConfidence }),

  clearSignals: () => set({ signals: [] }),
}));

// ============================================
// SELECTORS
// ============================================

export const selectFilteredSignals = (state: SignalsState) => {
  let filtered = state.signals;
  
  if (state.filterAction !== 'ALL') {
    filtered = filtered.filter((s) => s.action === state.filterAction);
  }
  
  if (state.minConfidence > 0) {
    filtered = filtered.filter((s) => s.confidence >= state.minConfidence);
  }
  
  return filtered;
};

export const selectLatestSignal = (state: SignalsState) =>
  state.signals[0] ?? null;

export const selectAgentThoughts = (state: SignalsState) => state.agentThoughts;

export const selectSignalsBySymbol = (symbol: string) => (state: SignalsState) =>
  state.signals.filter((s) => s.symbol === symbol);
