import { Injectable, Logger } from '@nestjs/common';
import Decimal from 'decimal.js';
import { v4 as uuidv4 } from 'uuid';
import {
    IExchangeAdapter,
    TickerData,
    OrderBook,
    OHLCV,
    OrderRequest,
    OrderResponse,
    BalanceInfo,
} from '../exchange.types';

/**
 * Mock Exchange Adapter
 * 
 * Paper trading simulator for development and testing.
 * Phase 1: Simple implementation (500ms → FILLED)
 * 
 * Features:
 * - Simulates order execution with configurable delay
 * - Virtual balance management per user
 * - Generates mock ticker data
 */
@Injectable()
export class MockExchangeAdapter implements IExchangeAdapter {
    private readonly logger = new Logger(MockExchangeAdapter.name);

    // Virtual balances (in-memory for Phase 1)
    private balances: Map<string, BalanceInfo> = new Map();

    // Orders storage
    private orders: Map<string, OrderResponse> = new Map();

    // Simulated price data
    private prices: Map<string, Decimal> = new Map();

    // Configuration
    private readonly FILL_DELAY_MS = 500;
    private readonly DEFAULT_BALANCE = '10000'; // Default USD balance

    constructor() {
        this.initializeDefaultPrices();
    }

    // ==========================================
    // Metadata
    // ==========================================

    getExchangeId(): 'mock' {
        return 'mock';
    }

    isSandboxMode(): boolean {
        return true; // Mock is always sandbox
    }

    async initialize(): Promise<void> {
        this.logger.log('🎮 Mock Exchange initialized (Paper Trading Mode)');
    }

    // ==========================================
    // Market Data
    // ==========================================

    async fetchTicker(symbol: string): Promise<TickerData | null> {
        const price = this.getPrice(symbol);
        const change = price.mul(new Decimal(Math.random() * 0.02 - 0.01)); // -1% to +1%

        return {
            symbol,
            last: price.toString(),
            bid: price.mul(0.9999).toString(),
            ask: price.mul(1.0001).toString(),
            high: price.mul(1.02).toString(),
            low: price.mul(0.98).toString(),
            volume: new Decimal(Math.random() * 10000000).toFixed(2),
            change: change.toString(),
            changePercent: change.div(price).mul(100).toFixed(2),
            timestamp: Date.now(),
        };
    }

    async fetchOrderBook(symbol: string, limit: number = 20): Promise<OrderBook | null> {
        const price = this.getPrice(symbol);
        const bids: Array<{ price: string; amount: string }> = [];
        const asks: Array<{ price: string; amount: string }> = [];

        for (let i = 0; i < limit; i++) {
            bids.push({
                price: price.mul(1 - 0.0001 * (i + 1)).toString(),
                amount: new Decimal(Math.random() * 100).toFixed(4),
            });
            asks.push({
                price: price.mul(1 + 0.0001 * (i + 1)).toString(),
                amount: new Decimal(Math.random() * 100).toFixed(4),
            });
        }

        return {
            symbol,
            bids,
            asks,
            timestamp: Date.now(),
        };
    }

    async fetchOHLCV(symbol: string, timeframe: string = '1h', limit: number = 100): Promise<OHLCV[]> {
        const basePrice = this.getPrice(symbol);
        const candles: OHLCV[] = [];
        const now = Date.now();
        const timeframeMs = this.parseTimeframe(timeframe);

        let currentPrice = basePrice.toNumber();

        for (let i = limit - 1; i >= 0; i--) {
            const volatility = 0.02;
            const open = currentPrice;
            const close = open * (1 + (Math.random() - 0.5) * volatility);
            const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.5);
            const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.5);

            candles.push({
                timestamp: now - (i * timeframeMs),
                open: new Decimal(open).toString(),
                high: new Decimal(high).toString(),
                low: new Decimal(low).toString(),
                close: new Decimal(close).toString(),
                volume: new Decimal(Math.random() * 1000000).toFixed(2),
            });

            currentPrice = close;
        }

        return candles;
    }

    // ==========================================
    // Trading (Simplified Phase 1)
    // ==========================================

    async createOrder(order: OrderRequest): Promise<OrderResponse> {
        const orderId = uuidv4();
        const price = order.price || this.getPrice(order.symbol).toString();
        const now = Date.now();

        // Create pending order
        const orderResponse: OrderResponse = {
            id: orderId,
            clientOrderId: order.clientOrderId,
            symbol: order.symbol,
            side: order.side,
            type: order.type,
            status: 'pending',
            amount: order.amount,
            filled: '0',
            remaining: order.amount,
            price,
            timestamp: now,
        };

        this.orders.set(orderId, orderResponse);
        this.logger.log(`📝 Mock order created: ${orderId} - ${order.symbol} ${order.side} ${order.amount}`);

        // Simulate fill after delay (async, non-blocking)
        this.simulateFill(orderId, price);

        return orderResponse;
    }

    async cancelOrder(orderId: string, _symbol: string): Promise<boolean> {
        const order = this.orders.get(orderId);
        if (!order) return false;

        if (order.status === 'filled') {
            this.logger.warn(`Cannot cancel filled order: ${orderId}`);
            return false;
        }

        order.status = 'cancelled';
        this.orders.set(orderId, order);
        this.logger.log(`❌ Mock order cancelled: ${orderId}`);
        return true;
    }

    async fetchOrder(orderId: string, _symbol: string): Promise<OrderResponse | null> {
        return this.orders.get(orderId) || null;
    }

    async fetchOpenOrders(symbol?: string): Promise<OrderResponse[]> {
        const openOrders: OrderResponse[] = [];

        for (const order of this.orders.values()) {
            if (order.status === 'open' || order.status === 'pending') {
                if (!symbol || order.symbol === symbol) {
                    openOrders.push(order);
                }
            }
        }

        return openOrders;
    }

    // ==========================================
    // Account
    // ==========================================

    async fetchBalance(): Promise<BalanceInfo> {
        // Return default balance if none set
        return this.getOrCreateBalance('default');
    }

    /**
     * Set virtual balance for a user (for testing)
     */
    setBalance(userId: string, currency: string, amount: string): void {
        const balance = this.getOrCreateBalance(userId);
        balance[currency] = {
            free: amount,
            locked: '0',
            total: amount,
        };
        this.balances.set(userId, balance);
    }

    // ==========================================
    // Private Helpers
    // ==========================================

    private initializeDefaultPrices(): void {
        // Default prices for common pairs
        this.prices.set('BTC/USDT', new Decimal(45000));
        this.prices.set('ETH/USDT', new Decimal(2500));
        this.prices.set('BNB/USDT', new Decimal(300));
        this.prices.set('SOL/USDT', new Decimal(100));
        this.prices.set('XRP/USDT', new Decimal(0.5));
        this.prices.set('DOGE/USDT', new Decimal(0.08));
    }

    private getPrice(symbol: string): Decimal {
        let price = this.prices.get(symbol);
        if (!price) {
            // Generate random price for unknown symbols
            price = new Decimal(100 + Math.random() * 100);
            this.prices.set(symbol, price);
        }
        return price;
    }

    private getOrCreateBalance(userId: string): BalanceInfo {
        let balance = this.balances.get(userId);
        if (!balance) {
            balance = {
                USDT: {
                    free: this.DEFAULT_BALANCE,
                    locked: '0',
                    total: this.DEFAULT_BALANCE,
                },
                BTC: {
                    free: '1',
                    locked: '0',
                    total: '1',
                },
                ETH: {
                    free: '10',
                    locked: '0',
                    total: '10',
                },
            };
            this.balances.set(userId, balance);
        }
        return balance;
    }

    private async simulateFill(orderId: string, fillPrice: string): Promise<void> {
        // Wait for simulated execution time
        await new Promise(resolve => setTimeout(resolve, this.FILL_DELAY_MS));

        const order = this.orders.get(orderId);
        if (!order || order.status === 'cancelled') return;

        // Update order to filled
        order.status = 'filled';
        order.filled = order.amount;
        order.remaining = '0';
        order.avgFillPrice = fillPrice;
        order.updatedAt = Date.now();

        this.orders.set(orderId, order);
        this.logger.log(`✅ Mock order filled: ${orderId} @ ${fillPrice}`);
    }

    private parseTimeframe(timeframe: string): number {
        const unit = timeframe.slice(-1);
        const amount = parseInt(timeframe.slice(0, -1)) || 1;

        switch (unit) {
            case 'm': return amount * 60 * 1000;
            case 'h': return amount * 60 * 60 * 1000;
            case 'd': return amount * 24 * 60 * 60 * 1000;
            case 'w': return amount * 7 * 24 * 60 * 60 * 1000;
            default: return 60 * 60 * 1000; // Default 1h
        }
    }
}
