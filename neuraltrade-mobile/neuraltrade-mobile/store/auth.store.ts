/**
 * Auth Store
 * ===========
 * Zustand store for authentication state management
 * Uses expo-secure-store for token persistence
 */

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import * as SecureStore from 'expo-secure-store';

// ============================================
// TYPES
// ============================================

export interface User {
  id: number;
  email: string;
  username: string;
  name: string;
  surname: string;
  role: 'USER' | 'ADMIN' | 'TRADER';
  status: 'ACTIVE' | 'INACTIVE' | 'SUSPENDED';
  phoneNumber?: string;
  profilePhoto?: string | null;
  gender?: string;
  riskProfile?: 'LOW' | 'MODERATE' | 'HIGH' | 'AGGRESSIVE';
  tradingEnabled: boolean;
  emailVerified: boolean;
}

export interface Subscription {
  id: string;
  plan: string;
  status: string;
  expiresAt: string;
}

export interface AuthState {
  // State
  user: User | null;
  accessToken: string | null;
  expiresIn: number | null;
  subscription: Subscription | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isInitialized: boolean;
  
  // Actions
  setAuth: (user: User, accessToken: string, expiresIn: number, subscription?: Subscription | null) => void;
  setUser: (user: User) => void;
  setLoading: (loading: boolean) => void;
  setInitialized: (initialized: boolean) => void;
  logout: () => void;
  clearAuth: () => void;
}

// ============================================
// SECURE STORAGE ADAPTER
// ============================================

const secureStorage = {
  getItem: async (name: string): Promise<string | null> => {
    try {
      return await SecureStore.getItemAsync(name);
    } catch (error) {
      console.warn('[SecureStore] Failed to get item:', error);
      return null;
    }
  },
  setItem: async (name: string, value: string): Promise<void> => {
    try {
      await SecureStore.setItemAsync(name, value);
    } catch (error) {
      console.warn('[SecureStore] Failed to set item:', error);
    }
  },
  removeItem: async (name: string): Promise<void> => {
    try {
      await SecureStore.deleteItemAsync(name);
    } catch (error) {
      console.warn('[SecureStore] Failed to remove item:', error);
    }
  },
};

// ============================================
// STORE
// ============================================

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      // Initial state
      user: null,
      accessToken: null,
      expiresIn: null,
      subscription: null,
      isAuthenticated: false,
      isLoading: false,
      isInitialized: false,

      setAuth: (user, accessToken, expiresIn, subscription = null) =>
        set({
          user,
          accessToken,
          expiresIn,
          subscription,
          isAuthenticated: true,
          isLoading: false,
        }),

      setUser: (user) => set({ user }),

      setLoading: (isLoading) => set({ isLoading }),

      setInitialized: (isInitialized) => set({ isInitialized }),

      logout: () =>
        set({
          user: null,
          accessToken: null,
          expiresIn: null,
          subscription: null,
          isAuthenticated: false,
          isLoading: false,
        }),

      clearAuth: () =>
        set({
          user: null,
          accessToken: null,
          expiresIn: null,
          subscription: null,
          isAuthenticated: false,
          isLoading: false,
        }),
    }),
    {
      name: 'neuraltrade-auth',
      storage: createJSONStorage(() => secureStorage),
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        expiresIn: state.expiresIn,
        subscription: state.subscription,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// ============================================
// SELECTORS
// ============================================

export const selectUser = (state: AuthState) => state.user;
export const selectIsAuthenticated = (state: AuthState) => state.isAuthenticated;
export const selectAccessToken = (state: AuthState) => state.accessToken;
export const selectIsLoading = (state: AuthState) => state.isLoading;
export const selectSubscription = (state: AuthState) => state.subscription;
