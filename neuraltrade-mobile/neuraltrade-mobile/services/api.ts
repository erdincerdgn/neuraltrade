/**
 * NeuralTrade API Service
 * ========================
 * Axios instance configured for NestJS Backend (Layer 3)
 * Endpoint: http://localhost:8000/api (or production URL)
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import * as SecureStore from 'expo-secure-store';

// ============================================
// API CONFIGURATION
// ============================================

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:4000/api/v1';
const AI_ENGINE_URL = process.env.EXPO_PUBLIC_AI_URL || 'http://localhost:8000';

// ============================================
// AXIOS INSTANCES
// ============================================

/** Main API client for NestJS Backend */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/** AI Engine client for Python Layer 4 */
export const aiClient: AxiosInstance = axios.create({
  baseURL: AI_ENGINE_URL,
  timeout: 60000, // AI requests may take longer
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================
// REQUEST INTERCEPTORS
// ============================================

apiClient.interceptors.request.use(
  async (config: InternalAxiosRequestConfig) => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.warn('[API] Failed to get auth token:', error);
    }
    return config;
  },
  (error: AxiosError) => Promise.reject(error)
);

// ============================================
// RESPONSE INTERCEPTORS
// ============================================

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    if (error.response?.status === 401) {
      console.warn('[API] Unauthorized - token may be expired');
      // Clear stored token on 401
      try {
        await SecureStore.deleteItemAsync('auth_token');
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    return Promise.reject(error);
  }
);

// ============================================
// API ENDPOINTS
// ============================================

export const endpoints = {
  // Auth
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    refresh: '/auth/refresh',
    logout: '/auth/logout',
  },
  // Portfolio
  portfolio: {
    summary: '/portfolio/summary',
    positions: '/portfolio/positions',
    history: '/portfolio/history',
    pnl: '/portfolio/pnl',
  },
  // Market Data
  market: {
    quote: (symbol: string) => `/market/quote/${symbol}`,
    history: (symbol: string) => `/market/history/${symbol}`,
    search: '/market/search',
  },
  // Trading
  trade: {
    order: '/trade/order',
    orders: '/trade/orders',
    cancel: (orderId: string) => `/trade/order/${orderId}`,
  },
  // AI Signals
  signals: {
    latest: '/signals/latest',
    history: '/signals/history',
    subscribe: '/signals/subscribe',
  },
};

export default apiClient;
