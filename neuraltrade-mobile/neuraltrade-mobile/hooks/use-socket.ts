/**
 * useSocket Hook
 * ===============
 * Memoized WebSocket connection hook for Layer 3 Gateway
 * Optimized for high-frequency market data without unnecessary re-renders
 * 
 * Usage:
 * ```tsx
 * const { isConnected, subscribeToSymbol, unsubscribeFromSymbol } = useSocket();
 * 
 * useEffect(() => {
 *   subscribeToSymbol('AAPL');
 *   return () => unsubscribeFromSymbol('AAPL');
 * }, []);
 * ```
 */

import { useEffect, useCallback, useRef } from 'react';
import { socketService, MarketTick, AISignal, AgentThought } from '@/services/socket';
import { useMarketStore } from '@/store/market.store';
import { useSignalsStore } from '@/store/signals.store';

interface UseSocketOptions {
  autoConnect?: boolean;
  authToken?: string;
  enableSignals?: boolean;
  enableAgentThoughts?: boolean;
}

interface UseSocketReturn {
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
  subscribeToSymbol: (symbol: string) => void;
  unsubscribeFromSymbol: (symbol: string) => void;
  subscribeToSignals: () => void;
  subscribeToAgentThoughts: () => void;
}

export function useSocket(options: UseSocketOptions = {}): UseSocketReturn {
  const {
    autoConnect = true,
    authToken,
    enableSignals = true,
    enableAgentThoughts = false,
  } = options;

  // Refs to avoid stale closures in callbacks
  const isInitialized = useRef(false);
  const subscribedSymbols = useRef<Set<string>>(new Set());

  // Store actions (stable references)
  const updateTick = useMarketStore((state) => state.updateTick);
  const setConnectionStatus = useMarketStore((state) => state.setConnectionStatus);
  const addSignal = useSignalsStore((state) => state.addSignal);
  const addAgentThought = useSignalsStore((state) => state.addAgentThought);
  
  // Connection status (subscribed for reactivity)
  const isConnected = useMarketStore((state) => state.isConnected);

  // Memoized handlers to prevent re-renders on every tick
  const handleMarketTick = useCallback(
    (tick: MarketTick) => {
      // Direct store update without causing component re-render
      updateTick(tick);
    },
    [updateTick]
  );

  const handleSignal = useCallback(
    (signal: AISignal) => {
      addSignal(signal);
    },
    [addSignal]
  );

  const handleAgentThought = useCallback(
    (thought: AgentThought) => {
      addAgentThought(thought);
    },
    [addAgentThought]
  );

  const handleConnect = useCallback(() => {
    setConnectionStatus(true);
    console.log('[useSocket] Connected to WebSocket Gateway');
  }, [setConnectionStatus]);

  const handleDisconnect = useCallback(() => {
    setConnectionStatus(false);
    console.log('[useSocket] Disconnected from WebSocket Gateway');
  }, [setConnectionStatus]);

  // Connect to socket
  const connect = useCallback(() => {
    const socket = socketService.connect(authToken);

    // Setup listeners with memoized handlers
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('market:tick', handleMarketTick);

    if (enableSignals) {
      socket.on('signal:new', handleSignal);
      socketService.subscribeToSignals();
    }

    if (enableAgentThoughts) {
      socket.on('agent:thought', handleAgentThought);
      socketService.subscribeToAgentThoughts();
    }
  }, [
    authToken,
    enableSignals,
    enableAgentThoughts,
    handleConnect,
    handleDisconnect,
    handleMarketTick,
    handleSignal,
    handleAgentThought,
  ]);

  // Disconnect from socket
  const disconnect = useCallback(() => {
    socketService.disconnect();
    subscribedSymbols.current.clear();
  }, []);

  // Subscribe to a symbol's market data
  const subscribeToSymbol = useCallback((symbol: string) => {
    if (!subscribedSymbols.current.has(symbol)) {
      socketService.subscribeToSymbol(symbol);
      subscribedSymbols.current.add(symbol);
    }
  }, []);

  // Unsubscribe from a symbol's market data
  const unsubscribeFromSymbol = useCallback((symbol: string) => {
    if (subscribedSymbols.current.has(symbol)) {
      socketService.unsubscribeFromSymbol(symbol);
      subscribedSymbols.current.delete(symbol);
    }
  }, []);

  // Subscribe to AI signals
  const subscribeToSignals = useCallback(() => {
    socketService.subscribeToSignals();
  }, []);

  // Subscribe to agent thoughts (terminal view)
  const subscribeToAgentThoughts = useCallback(() => {
    socketService.subscribeToAgentThoughts();
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !isInitialized.current) {
      isInitialized.current = true;
      connect();
    }

    return () => {
      // Cleanup on unmount
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    isConnected,
    connect,
    disconnect,
    subscribeToSymbol,
    unsubscribeFromSymbol,
    subscribeToSignals,
    subscribeToAgentThoughts,
  };
}

export default useSocket;
