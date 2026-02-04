/**
 * useMarketData Hook
 * ===================
 * React Query hook for fetching market data from Fastify API (Layer 3)
 * Combined with real-time updates from useSocket
 */

import { useQuery, UseQueryOptions } from '@tanstack/react-query';
import { apiClient, endpoints } from '@/services/api';

// ============================================
// TYPES
// ============================================

export interface MarketQuote {
  symbol: string;
  name: string;
  price: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  changePercent: number;
  marketCap?: number;
  pe?: number;
  timestamp: number;
}

export interface MarketHistory {
  symbol: string;
  data: {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
}

// ============================================
// QUERY FUNCTIONS
// ============================================

const fetchQuote = async (symbol: string): Promise<MarketQuote> => {
  const response = await apiClient.get(endpoints.market.quote(symbol));
  return response.data;
};

const fetchHistory = async (
  symbol: string,
  period: '1D' | '1W' | '1M' | '3M' | '1Y' = '1D'
): Promise<MarketHistory> => {
  const response = await apiClient.get(endpoints.market.history(symbol), {
    params: { period },
  });
  return response.data;
};

// ============================================
// HOOKS
// ============================================

/**
 * Fetch market quote for a symbol
 */
export function useMarketQuote(
  symbol: string,
  options?: Omit<UseQueryOptions<MarketQuote>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: ['market', 'quote', symbol],
    queryFn: () => fetchQuote(symbol),
    staleTime: 10000, // 10 seconds (real-time updates via socket)
    refetchInterval: 30000, // Fallback refetch every 30s
    enabled: !!symbol,
    ...options,
  });
}

/**
 * Fetch market history for a symbol
 */
export function useMarketHistory(
  symbol: string,
  period: '1D' | '1W' | '1M' | '3M' | '1Y' = '1D',
  options?: Omit<UseQueryOptions<MarketHistory>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: ['market', 'history', symbol, period],
    queryFn: () => fetchHistory(symbol, period),
    staleTime: 60000, // 1 minute
    enabled: !!symbol,
    ...options,
  });
}

export default useMarketQuote;
