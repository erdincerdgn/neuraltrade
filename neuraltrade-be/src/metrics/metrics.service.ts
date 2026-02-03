import { Injectable, OnModuleInit } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import * as client from 'prom-client';

/**
 * Metrics Service
 * 
 * Prometheus metrics for NeuralTrade:
 * - HTTP request metrics
 * - Order execution metrics
 * - Trading performance metrics
 * - System health metrics
 */
@Injectable()
export class MetricsService implements OnModuleInit {
    private registry: client.Registry;

    // HTTP Metrics
    public httpRequestsTotal: client.Counter<string>;
    public httpRequestDuration: client.Histogram<string>;

    // Order Metrics
    public ordersTotal: client.Counter<string>;
    public orderLatency: client.Histogram<string>;
    public orderFillRate: client.Gauge<string>;

    // Trading Metrics
    public positionsOpen: client.Gauge<string>;
    public portfolioValue: client.Gauge<string>;
    public dailyPnL: client.Gauge<string>;
    public signalsReceived: client.Counter<string>;

    // System Metrics
    public circuitBreakerState: client.Gauge<string>;
    public websocketConnections: client.Gauge<string>;
    public redisLatency: client.Histogram<string>;

    constructor() {
        this.registry = new client.Registry();

        // Add default metrics (CPU, memory, etc.)
        client.collectDefaultMetrics({ register: this.registry });

        this.initializeMetrics();
    }

    onModuleInit() {
        // Metrics are already initialized in constructor
    }

    private initializeMetrics(): void {
        // ==========================================
        // HTTP METRICS
        // ==========================================

        this.httpRequestsTotal = new client.Counter({
            name: 'neuraltrade_http_requests_total',
            help: 'Total number of HTTP requests',
            labelNames: ['method', 'path', 'status'],
            registers: [this.registry],
        });

        this.httpRequestDuration = new client.Histogram({
            name: 'neuraltrade_http_request_duration_seconds',
            help: 'HTTP request duration in seconds',
            labelNames: ['method', 'path', 'status'],
            buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
            registers: [this.registry],
        });

        // ==========================================
        // ORDER METRICS
        // ==========================================

        this.ordersTotal = new client.Counter({
            name: 'neuraltrade_orders_total',
            help: 'Total number of orders placed',
            labelNames: ['side', 'type', 'status', 'exchange'],
            registers: [this.registry],
        });

        this.orderLatency = new client.Histogram({
            name: 'neuraltrade_order_latency_seconds',
            help: 'Order execution latency in seconds',
            labelNames: ['exchange', 'type'],
            buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            registers: [this.registry],
        });

        this.orderFillRate = new client.Gauge({
            name: 'neuraltrade_order_fill_rate',
            help: 'Order fill rate (0-1)',
            labelNames: ['exchange'],
            registers: [this.registry],
        });

        // ==========================================
        // TRADING METRICS
        // ==========================================

        this.positionsOpen = new client.Gauge({
            name: 'neuraltrade_positions_open',
            help: 'Number of open positions',
            labelNames: ['portfolio_id'],
            registers: [this.registry],
        });

        this.portfolioValue = new client.Gauge({
            name: 'neuraltrade_portfolio_value_usd',
            help: 'Portfolio total value in USD',
            labelNames: ['portfolio_id', 'user_id'],
            registers: [this.registry],
        });

        this.dailyPnL = new client.Gauge({
            name: 'neuraltrade_daily_pnl_usd',
            help: 'Daily PnL in USD',
            labelNames: ['portfolio_id'],
            registers: [this.registry],
        });

        this.signalsReceived = new client.Counter({
            name: 'neuraltrade_ai_signals_total',
            help: 'Total AI signals received',
            labelNames: ['action', 'symbol', 'status'],
            registers: [this.registry],
        });

        // ==========================================
        // SYSTEM METRICS
        // ==========================================

        this.circuitBreakerState = new client.Gauge({
            name: 'neuraltrade_circuit_breaker_state',
            help: 'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            labelNames: ['portfolio_id', 'type'],
            registers: [this.registry],
        });

        this.websocketConnections = new client.Gauge({
            name: 'neuraltrade_websocket_connections',
            help: 'Number of active WebSocket connections',
            registers: [this.registry],
        });

        this.redisLatency = new client.Histogram({
            name: 'neuraltrade_redis_latency_seconds',
            help: 'Redis operation latency',
            labelNames: ['operation'],
            buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registers: [this.registry],
        });
    }

    // ==========================================
    // METRICS ENDPOINT
    // ==========================================

    async getMetrics(): Promise<string> {
        return this.registry.metrics();
    }

    getContentType(): string {
        return this.registry.contentType;
    }

    // ==========================================
    // EVENT HANDLERS
    // ==========================================

    @OnEvent('order.placed')
    handleOrderPlaced(payload: { side: string; type: string; exchange?: string }): void {
        this.ordersTotal.inc({
            side: payload.side,
            type: payload.type,
            status: 'placed',
            exchange: payload.exchange || 'mock',
        });
    }

    @OnEvent('order.filled')
    handleOrderFilled(payload: { side: string; type: string; exchange?: string; latency?: number }): void {
        this.ordersTotal.inc({
            side: payload.side,
            type: payload.type,
            status: 'filled',
            exchange: payload.exchange || 'mock',
        });

        if (payload.latency) {
            this.orderLatency.observe(
                { exchange: payload.exchange || 'mock', type: payload.type },
                payload.latency / 1000, // Convert to seconds
            );
        }
    }

    @OnEvent('ai.signal.received')
    handleSignalReceived(payload: { action: string; symbol: string }): void {
        this.signalsReceived.inc({
            action: payload.action,
            symbol: payload.symbol,
            status: 'received',
        });
    }

    @OnEvent('circuit.tripped')
    handleCircuitTripped(payload: { portfolioId: number; type: string }): void {
        this.circuitBreakerState.set(
            { portfolio_id: payload.portfolioId.toString(), type: payload.type },
            1, // OPEN
        );
    }

    @OnEvent('circuit.reset')
    handleCircuitReset(payload: { portfolioId: number }): void {
        this.circuitBreakerState.set(
            { portfolio_id: payload.portfolioId.toString(), type: 'all' },
            0, // CLOSED
        );
    }

    // ==========================================
    // UTILITY METHODS
    // ==========================================

    recordHttpRequest(method: string, path: string, status: number, duration: number): void {
        const labels = { method, path: this.normalizePath(path), status: status.toString() };
        this.httpRequestsTotal.inc(labels);
        this.httpRequestDuration.observe(labels, duration);
    }

    setWebSocketConnections(count: number): void {
        this.websocketConnections.set(count);
    }

    recordRedisLatency(operation: string, duration: number): void {
        this.redisLatency.observe({ operation }, duration);
    }

    updatePortfolioMetrics(portfolioId: number, userId: number, value: number, pnl: number, positions: number): void {
        this.portfolioValue.set({ portfolio_id: portfolioId.toString(), user_id: userId.toString() }, value);
        this.dailyPnL.set({ portfolio_id: portfolioId.toString() }, pnl);
        this.positionsOpen.set({ portfolio_id: portfolioId.toString() }, positions);
    }



    
    private normalizePath(path: string): string {
        // 🛡️ BURASI KRİTİK: Eğer path yoksa (undefined/null) boş string veya 'unknown' dön
        if (!path) return 'unknown'; 

        return path
            .replace(/\/\d+/g, '/:id')
            .replace(/\/[a-f0-9-]{36}/g, '/:uuid');
    }

    // private normalizePath(path: string): string {
    //     // Replace IDs with placeholders to reduce cardinality
    //     return path
    //         .replace(/\/\d+/g, '/:id')
    //         .replace(/\/[a-f0-9-]{36}/g, '/:uuid');
    // }
}
