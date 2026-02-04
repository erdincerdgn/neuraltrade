/**
 * Services Index
 * ===============
 * Barrel export for all service modules
 */

export { apiClient, aiClient, endpoints } from './api';
export { socketService } from './socket';
export type { MarketTick, AISignal, AgentThought, SocketEventType } from './socket';

export { authService } from './auth';
export type { LoginRequest, RegisterRequest, AuthResponse, SocialAuthRequest } from './auth';
