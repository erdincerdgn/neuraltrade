import {WebSocketGateway,
    WebSocketServer,
    SubscribeMessage,
    OnGatewayInit,
    OnGatewayConnection,
    OnGatewayDisconnect,
    MessageBody,
    ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

@WebSocketGateway({
    cors: {
        origin: '*',
    },
    namespace: '/trading',
    pingInterval: 10000,
    pingTimeout: 5000,
})
export class EventsGateway implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect {
    @WebSocketServer()
    server: Server;

    private readonly logger = new Logger(EventsGateway.name);
    private connectedClients: Map<string, ClientInfo> = new Map();
    private redisAdapter: boolean = false;

    constructor(private readonly jwtService: JwtService) {}

    async afterInit(_server: Server) {
        this.logger.log('🔌 WebSocket Gateway initializing...');
        this.logger.log('✅ WebSocket Gateway initialized');
    }

    async handleConnection(client: Socket) {
        try {
            const token = client.handshake.auth?.token ||
                client.handshake.headers?.authorization?.split(' ')[1];

            if (!token) {
                this.logger.warn(`Client ${client.id} rejected: No token`);
                client.emit('error', { code: 'AUTH_REQUIRED', message: 'Authentication required' });
                client.disconnect();
                return;
            }

            const payload = this.jwtService.verify(token);
            const userId = payload.sub;

            this.connectedClients.set(client.id, {
                userId,
                socket: client,
                connectedAt: new Date(),
                subscriptions: [],
            });

            client.join(`user:${userId}`);

            this.logger.log(`✅ Connected: ${client.id} (User: ${userId}) | Total: ${this.connectedClients.size}`);

            client.emit('connected', {
                message: 'Connected to NeuralTrade',
                userId,
                timestamp: new Date().toISOString(),
                features: {
                    redisEnabled: this.redisAdapter,
                },
            });

            this.setupHeartbeat(client);
        } catch (error) {
            this.logger.error(`Connection error ${client.id}: ${error.message}`);
            client.emit('error', { code: 'AUTH_FAILED', message: 'Invalid token' });
            client.disconnect();
        }
    }

    handleDisconnect(client: Socket) {
        const clientInfo = this.connectedClients.get(client.id);
        if (clientInfo) {
            this.logger.log(`👋 Disconnected: ${client.id} (User: ${clientInfo.userId}) | Total: ${this.connectedClients.size - 1}`);
            this.connectedClients.delete(client.id);
        }
    }

    private setupHeartbeat(client: Socket): void {
        const interval = setInterval(() => {
            if (!this.connectedClients.has(client.id)) {
                clearInterval(interval);
                return;
            }
            client.emit('heartbeat', { timestamp: Date.now() });
        }, 30000);

        client.on('disconnect', () => clearInterval(interval));
    }

    //==========================================
    // SUBSCRIPTION HANDLERS
    // ==========================================

    @SubscribeMessage('subscribe:ticker')
    handleSubscribeTicker(
        @ConnectedSocket() client: Socket,
        @MessageBody() data: { symbols: string[] },
    ) {
        const symbols = data.symbols || [];
        symbols.forEach(symbol => {
            client.join(`ticker:${symbol}`);
        });

        const clientInfo = this.connectedClients.get(client.id);
        if (clientInfo) {
            clientInfo.subscriptions.push(...symbols.map(s => `ticker:${s}`));
        }

        this.logger.debug(`${client.id} subscribed to tickers: ${symbols.join(', ')}`);
        return { event: 'subscribed', data: { channel: 'ticker', symbols } };
    }

    @SubscribeMessage('unsubscribe:ticker')
    handleUnsubscribeTicker(
        @ConnectedSocket() client: Socket,
        @MessageBody() data: { symbols: string[] },
    ) {
        const symbols = data.symbols || [];
        symbols.forEach(symbol => {
            client.leave(`ticker:${symbol}`);
        });
        return { event: 'unsubscribed', data: { channel: 'ticker', symbols } };
    }

    @SubscribeMessage('subscribe:signals')
    handleSubscribeSignals(@ConnectedSocket() client: Socket) {
        client.join('signals');
        this.logger.debug(`${client.id} subscribed to AI signals`);
        return { event: 'subscribed', data: { channel: 'signals' } };
    }

    @SubscribeMessage('subscribe:portfolio')
    handleSubscribePortfolio(
        @ConnectedSocket() client: Socket,
        @MessageBody() data: { portfolioId: number },
    ) {
        client.join(`portfolio:${data.portfolioId}`);
        return { event: 'subscribed', data: { channel: 'portfolio', portfolioId: data.portfolioId } };
    }

    @SubscribeMessage('subscribe:orders')
    handleSubscribeOrders(@ConnectedSocket() client: Socket) {
        const clientInfo = this.connectedClients.get(client.id);
        if (clientInfo) {
            client.join(`orders:${clientInfo.userId}`);
            return { event: 'subscribed', data: { channel: 'orders' } };
        }
        return { event: 'error', data: { message: 'Not authenticated' } };
    }

    // ==========================================
    // BROADCAST METHODS (called from services)
    // ==========================================

    broadcastTicker(symbol: string, data: TickerData) {
        this.server.to(`ticker:${symbol}`).emit('ticker', {
            symbol,
            ...data,
            timestamp: new Date().toISOString(),
        });
    }

    broadcastSignal(signal: AISignalBroadcast) {
        this.server.to('signals').emit('signal', {
            ...signal,
            timestamp: new Date().toISOString(),
        });
        this.logger.log(`📡 Signal broadcast: ${signal.symbol} - ${signal.action}`);
    }

    /**
     * NEW: Broadcast to specific room
     */
    broadcastToRoom(room: string, event: string, payload: any): void {
        this.server.to(room).emit(event, {
            ...payload,
            timestamp: new Date().toISOString(),
        });
    }

    /**
     * NEW: Broadcast alert to all connected clients
     */
    broadcastAlert(alert: SignalAlertPayload): void {
        this.server.emit('alert:signal', {
            ...alert,
            timestamp: new Date().toISOString(),
        });this.logger.log(`🚨 Alert broadcast: ${alert.symbol} - ${alert.type} [${alert.priority}]`);
    }

    /**
     * NEW: Broadcast to all connected clients
     */
    broadcast(event: string, payload: any): void {
        this.server.emit(event, {
            ...payload,
            timestamp: new Date().toISOString(),
        });
    }

    sendPortfolioUpdate(userId: number, portfolioId: number, data: Record<string, unknown>) {
        this.server.to(`user:${userId}`).emit('portfolio:update', {
            portfolioId,
            ...data,
            timestamp: new Date().toISOString(),
        });
    }

    sendOrderUpdate(userId: number, orderId: number, status: string, data: Record<string, unknown>) {
        this.server.to(`user:${userId}`).emit('order:update', {
            orderId,
            status,
            ...data,
            timestamp: new Date().toISOString(),
        });
    }

    sendNotification(userId: number, notification: UserNotification) {
        this.server.to(`user:${userId}`).emit('notification', {
            ...notification,
            timestamp: new Date().toISOString(),
        });
    }

    broadcastSystemMessage(message: string, type: 'info' | 'warning' | 'error' = 'info') {
        this.server.emit('system', {
            message,
            type,
            timestamp: new Date().toISOString(),
        });
    }

    // ==========================================
    // UTILITY METHODS
    // ==========================================

    getConnectedClientsCount(): number {
        return this.connectedClients.size;
    }

    isUserConnected(userId: number): boolean {
        for (const [, info] of this.connectedClients) {
            if (info.userId === userId) return true;
        }
        return false;
    }

    getClientsByUserId(userId: number): ClientInfo[] {
        const clients: ClientInfo[] = [];
        for (const [, info] of this.connectedClients) {
            if (info.userId === userId) clients.push(info);
        }
        return clients;
    }

    isRedisAdapterEnabled(): boolean {
        return this.redisAdapter;
    }

    getServer(): Server {
        return this.server;
    }
}

//==========================================
// TYPES
// ==========================================

interface ClientInfo {
    userId: number;
    socket: Socket;
    connectedAt: Date;
    subscriptions: string[];
}

interface TickerData {
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    bid?: number;
    ask?: number;
}

interface AISignalBroadcast {
    symbol: string;
    action: 'BUY' | 'SELL' | 'HOLD' | 'CLOSE' | string;
    confidence: number;
    reasoning?: string;
    targetPrice?: number;
    stopLoss?: number;
    modelUsed?: string;
    regime?: string;
}

interface UserNotification {
    title: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
    action?: string;
}

interface SignalAlertPayload {
    type: 'BUY' | 'SELL' | 'CLOSE' | 'WARNING' | string;
    symbol: string;
    action: string;
    confidence: number;
    message: string;
    priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export { AISignalBroadcast, SignalAlertPayload, TickerData, UserNotification, ClientInfo };