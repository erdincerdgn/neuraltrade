/**
 * NeuralTrade WebSocket Service
 * ==============================
 * Socket.io client for real-time connection to NestJS Gateway (Layer 3)
 * Handles: Market ticks, AI signals, Order updates, Portfolio changes
 */

import { io, Socket } from 'socket.io-client';

// ============================================
// SOCKET CONFIGURATION
// ============================================

const SOCKET_URL = process.env.EXPO_PUBLIC_WS_URL || 'http://localhost:4000';

export type SocketEventType = 
  | 'market:tick'
  | 'signal:new'
  | 'signal:update'
  | 'order:filled'
  | 'order:cancelled'
  | 'portfolio:update'
  | 'agent:thought'  // For AI Swarm terminal view
  | 'error';

export interface MarketTick {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

export interface AISignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string;
  agentId: string;
  timestamp: number;
}

export interface AgentThought {
  agentId: string;
  agentName: string;
  thought: string;
  stage: string;
  timestamp: number;
}

// ============================================
// SOCKET CLASS
// ============================================

class SocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  /**
   * Initialize socket connection
   */
  connect(authToken?: string): Socket {
    if (this.socket?.connected) {
      return this.socket;
    }

    this.socket = io(SOCKET_URL, {
      transports: ['websocket'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      auth: authToken ? { token: authToken } : undefined,
    });

    this.setupBaseListeners();
    return this.socket;
  }

  /**
   * Setup base connection listeners
   */
  private setupBaseListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('[Socket] Connected to WebSocket Gateway');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.warn('[Socket] Disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      console.error('[Socket] Connection error:', error.message);
      this.reconnectAttempts++;
    });

    this.socket.on('error', (error) => {
      console.error('[Socket] Error:', error);
    });
  }

  /**
   * Subscribe to a specific symbol's market data
   */
  subscribeToSymbol(symbol: string): void {
    this.socket?.emit('market:subscribe', { symbol });
  }

  /**
   * Unsubscribe from a symbol's market data
   */
  unsubscribeFromSymbol(symbol: string): void {
    this.socket?.emit('market:unsubscribe', { symbol });
  }

  /**
   * Subscribe to AI signals
   */
  subscribeToSignals(): void {
    this.socket?.emit('signals:subscribe');
  }

  /**
   * Subscribe to agent thoughts (terminal view)
   */
  subscribeToAgentThoughts(): void {
    this.socket?.emit('agent:subscribe');
  }

  /**
   * Add event listener
   */
  on<T>(event: SocketEventType, callback: (data: T) => void): void {
    this.socket?.on(event, callback);
  }

  /**
   * Remove event listener
   */
  off(event: SocketEventType, callback?: (...args: unknown[]) => void): void {
    if (callback) {
      this.socket?.off(event, callback);
    } else {
      this.socket?.off(event);
    }
  }

  /**
   * Disconnect socket
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Get connection status
   */
  get isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  /**
   * Get socket instance
   */
  get instance(): Socket | null {
    return this.socket;
  }
}

// Singleton instance
export const socketService = new SocketService();
export default socketService;
