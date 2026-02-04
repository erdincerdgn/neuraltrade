/**
 * Auth Service
 * =============
 * Authentication API integration for NestJS Backend
 * Handles login, register, token refresh, and social auth
 */

import { apiClient, endpoints } from './api';
import { useAuthStore, User, Subscription } from '@/store/auth.store';
import * as SecureStore from 'expo-secure-store';

// ============================================
// TYPES
// ============================================

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterRequest {
  email: string;
  username: string;
  name: string;
  surname: string;
  password: string;
  phoneNumber?: string;
  gender?: 'Male' | 'Female' | 'Unspecified';
  dateOfBirth?: string;
  riskProfile?: 'LOW' | 'MODERATE' | 'HIGH' | 'AGGRESSIVE';
  profileDescription?: string;
}

export interface AuthResponse {
  accessToken: string;
  expiresIn: number;
  user: User;
  subscription?: Subscription | null;
}

export interface SocialAuthRequest {
  provider: 'google' | 'apple';
  idToken: string;
  email?: string;
  name?: string;
}

// ============================================
// AUTH SERVICE
// ============================================

class AuthService {
  /**
   * Login with email and password
   */
  async login(data: LoginRequest): Promise<AuthResponse> {
    try {
      // Debug: Log the full URL being called
      const fullUrl = `${apiClient.defaults.baseURL}${endpoints.auth.login}`;
      console.log('[Auth] Attempting login to:', fullUrl);
      
      const response = await apiClient.post<AuthResponse>(endpoints.auth.login, data);
      const { accessToken, expiresIn, user, subscription } = response.data;
      
      // Store token
      await this.setToken(accessToken);
      
      // Update store
      useAuthStore.getState().setAuth(user, accessToken, expiresIn, subscription);
      
      return response.data;
    } catch (error: any) {
      console.error('[Auth] Login failed:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Register new user account
   */
  async register(data: RegisterRequest): Promise<AuthResponse> {
    try {
      const response = await apiClient.post<AuthResponse>(endpoints.auth.register, data);
      const { accessToken, expiresIn, user, subscription } = response.data;
      
      // Store token
      await this.setToken(accessToken);
      
      // Update store
      useAuthStore.getState().setAuth(user, accessToken, expiresIn, subscription);
      
      return response.data;
    } catch (error: any) {
      console.error('[Auth] Register failed:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Social authentication (Google/Apple)
   */
  async socialAuth(data: SocialAuthRequest): Promise<AuthResponse> {
    try {
      const response = await apiClient.post<AuthResponse>(`/auth/${data.provider}`, data);
      const { accessToken, expiresIn, user, subscription } = response.data;
      
      // Store token
      await this.setToken(accessToken);
      
      // Update store
      useAuthStore.getState().setAuth(user, accessToken, expiresIn, subscription);
      
      return response.data;
    } catch (error: any) {
      console.error('[Auth] Social auth failed:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Refresh access token
   */
  async refreshToken(): Promise<AuthResponse | null> {
    try {
      const response = await apiClient.post<AuthResponse>(endpoints.auth.refresh);
      const { accessToken, expiresIn, user, subscription } = response.data;
      
      // Store new token
      await this.setToken(accessToken);
      
      // Update store
      useAuthStore.getState().setAuth(user, accessToken, expiresIn, subscription);
      
      return response.data;
    } catch (error) {
      console.error('[Auth] Token refresh failed:', error);
      return null;
    }
  }

  /**
   * Logout and clear all auth data
   */
  async logout(): Promise<void> {
    try {
      // Call logout endpoint
      await apiClient.post(endpoints.auth.logout);
    } catch (error) {
      console.warn('[Auth] Logout API call failed:', error);
    } finally {
      // Always clear local state
      await this.clearToken();
      useAuthStore.getState().logout();
    }
  }

  /**
   * Initialize auth state from storage
   */
  async initialize(): Promise<boolean> {
    try {
      const token = await this.getToken();
      
      if (token) {
        // Verify token is still valid
        const refreshed = await this.refreshToken();
        useAuthStore.getState().setInitialized(true);
        return !!refreshed;
      }
      
      useAuthStore.getState().setInitialized(true);
      return false;
    } catch (error) {
      console.error('[Auth] Initialize failed:', error);
      useAuthStore.getState().setInitialized(true);
      return false;
    }
  }

  // ============================================
  // TOKEN MANAGEMENT
  // ============================================

  private async setToken(token: string): Promise<void> {
    await SecureStore.setItemAsync('auth_token', token);
  }

  async getToken(): Promise<string | null> {
    return await SecureStore.getItemAsync('auth_token');
  }

  private async clearToken(): Promise<void> {
    await SecureStore.deleteItemAsync('auth_token');
  }
}

// Singleton instance
export const authService = new AuthService();
export default authService;
