import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as ccxt from 'ccxt';
import Decimal from 'decimal.js';
import {
    IExchangeAdapter,
    ExchangeConfig,
    TickerData,
    OrderBook,
    OHLCV,
    OrderRequest,
    OrderResponse,
    BalanceInfo,
    OrderStatus,
} from '../exchange.types';

/**
 * Base CCXT Adapter
 * 
 * Shared implementation for CCXT-based exchanges.
 * Handles common CCXT operations and type conversions.
 */
@Injectable()
export abstract class BaseCCXTAdapter implements IExchangeAdapter {
    protected readonly logger: Logger;
    protected exchange: ccxt.Exchange;
    protected sandboxMode: boolean = true;
    protected initialized: boolean = false;

    constructor(
        protected readonly config: ConfigService,
        protected readonly exchangeConfig: ExchangeConfig,
    ) {
        this.logger = new Logger(this.constructor.name);
    }

    // ==========================================
    // Abstract methods (exchange-specific)
    // ==========================================

    abstract getExchangeId(): 'binance' | 'bybit';
    protected abstract createExchangeInstance(): ccxt.Exchange;

    // ==========================================
    // Initialization
    // ==========================================

    async initialize(): Promise<void> {
        if (this.initialized) return;

        try {
            this.exchange = this.createExchangeInstance();

            // Enable sandbox/testnet mode
            this.sandboxMode = this.exchangeConfig.sandbox ?? true;
            if (this.sandboxMode && this.exchange.setSandboxMode) {
                this.exchange.setSandboxMode(true);
            }

            // Load markets
            await this.exchange.loadMarkets();

            this.initialized = true;
            this.logger.log(`✅ ${this.getExchangeId().toUpperCase()} initialized in ${this.sandboxMode ? 'SANDBOX' : 'LIVE'} mode`);
            this.logger.log(`📊 Loaded ${Object.keys(this.exchange.markets).length} markets`);
        } catch (error) {
            this.logger.error(`Failed to initialize: ${error.message}`);
            throw error;
        }
    }

    isSandboxMode(): boolean {
        return this.sandboxMode;
    }

    // ==========================================
    // Market Data
    // ==========================================

    async fetchTicker(symbol: string): Promise<TickerData | null> {
        try {
            await this.ensureInitialized();
            const ticker = await this.exchange.fetchTicker(symbol);

            return {
                symbol: ticker.symbol,
                last: this.safeString(ticker.last),
                bid: this.safeString(ticker.bid),
                ask: this.safeString(ticker.ask),
                high: this.safeString(ticker.high),
                low: this.safeString(ticker.low),
                volume: this.safeString(ticker.baseVolume),
                change: this.safeString(ticker.change),
                changePercent: this.safeString(ticker.percentage),
                timestamp: ticker.timestamp || Date.now(),
            };
        } catch (error) {
            this.logger.error(`fetchTicker error: ${error.message}`);
            return null;
        }
    }

    async fetchOrderBook(symbol: string, limit: number = 20): Promise<OrderBook | null> {
        try {
            await this.ensureInitialized();
            const orderBook = await this.exchange.fetchOrderBook(symbol, limit);

            return {
                symbol,
                bids: orderBook.bids.map(([price, amount]) => ({
                    price: this.safeString(price),
                    amount: this.safeString(amount),
                })),
                asks: orderBook.asks.map(([price, amount]) => ({
                    price: this.safeString(price),
                    amount: this.safeString(amount),
                })),
                timestamp: orderBook.timestamp || Date.now(),
            };
        } catch (error) {
            this.logger.error(`fetchOrderBook error: ${error.message}`);
            return null;
        }
    }

    async fetchOHLCV(symbol: string, timeframe: string = '1h', limit: number = 100): Promise<OHLCV[]> {
        try {
            await this.ensureInitialized();
            const ohlcv = await this.exchange.fetchOHLCV(symbol, timeframe, undefined, limit);

            return ohlcv.map(([timestamp, open, high, low, close, volume]) => ({
                timestamp: timestamp as number,
                open: this.safeString(open),
                high: this.safeString(high),
                low: this.safeString(low),
                close: this.safeString(close),
                volume: this.safeString(volume),
            }));
        } catch (error) {
            this.logger.error(`fetchOHLCV error: ${error.message}`);
            return [];
        }
    }

    // ==========================================
    // Trading
    // ==========================================

    async createOrder(order: OrderRequest): Promise<OrderResponse> {
        await this.ensureInitialized();

        try {
            const ccxtOrder = await this.exchange.createOrder(
                order.symbol,
                order.type,
                order.side,
                parseFloat(order.amount),
                order.price ? parseFloat(order.price) : undefined,
            );

            return this.mapOrderResponse(ccxtOrder);
        } catch (error) {
            this.logger.error(`createOrder error: ${error.message}`);
            throw error;
        }
    }

    async cancelOrder(orderId: string, symbol: string): Promise<boolean> {
        try {
            await this.ensureInitialized();
            await this.exchange.cancelOrder(orderId, symbol);
            this.logger.log(`❌ Order cancelled: ${orderId}`);
            return true;
        } catch (error) {
            this.logger.error(`cancelOrder error: ${error.message}`);
            return false;
        }
    }

    async fetchOrder(orderId: string, symbol: string): Promise<OrderResponse | null> {
        try {
            await this.ensureInitialized();
            const order = await this.exchange.fetchOrder(orderId, symbol);
            return this.mapOrderResponse(order);
        } catch (error) {
            this.logger.error(`fetchOrder error: ${error.message}`);
            return null;
        }
    }

    async fetchOpenOrders(symbol?: string): Promise<OrderResponse[]> {
        try {
            await this.ensureInitialized();
            const orders = await this.exchange.fetchOpenOrders(symbol);
            return orders.map(o => this.mapOrderResponse(o));
        } catch (error) {
            this.logger.error(`fetchOpenOrders error: ${error.message}`);
            return [];
        }
    }

    // ==========================================
    // Account
    // ==========================================

    async fetchBalance(): Promise<BalanceInfo> {
        try {
            await this.ensureInitialized();
            const balance = await this.exchange.fetchBalance();

            const result: BalanceInfo = {};

            for (const [currency, data] of Object.entries(balance)) {
                if (typeof data === 'object' && data !== null && 'free' in data) {
                    const balanceData = data as { free: number; used: number; total: number };
                    if (balanceData.total > 0) {
                        result[currency] = {
                            free: this.safeString(balanceData.free),
                            locked: this.safeString(balanceData.used),
                            total: this.safeString(balanceData.total),
                        };
                    }
                }
            }

            return result;
        } catch (error) {
            this.logger.error(`fetchBalance error: ${error.message}`);
            return {};
        }
    }

    // ==========================================
    // Helper Methods
    // ==========================================

    protected async ensureInitialized(): Promise<void> {
        if (!this.initialized) {
            await this.initialize();
        }
    }

    protected safeString(value: any): string {
        if (value === null || value === undefined) return '0';
        return new Decimal(value).toString();
    }

    protected mapOrderStatus(status: string): OrderStatus {
        const statusMap: Record<string, OrderStatus> = {
            'open': 'open',
            'closed': 'filled',
            'canceled': 'cancelled',
            'cancelled': 'cancelled',
            'rejected': 'rejected',
            'expired': 'cancelled',
        };
        return statusMap[status] || 'pending';
    }

    protected mapOrderResponse(order: ccxt.Order): OrderResponse {
        return {
            id: order.id,
            clientOrderId: order.clientOrderId,
            symbol: order.symbol,
            side: order.side as 'buy' | 'sell',
            type: order.type as 'market' | 'limit' | 'stop_limit',
            status: this.mapOrderStatus(order.status),
            amount: this.safeString(order.amount),
            filled: this.safeString(order.filled),
            remaining: this.safeString(order.remaining),
            price: order.price ? this.safeString(order.price) : undefined,
            avgFillPrice: order.average ? this.safeString(order.average) : undefined,
            fee: order.fee?.cost ? this.safeString(order.fee.cost) : undefined,
            feeCurrency: order.fee?.currency,
            timestamp: order.timestamp || Date.now(),
            updatedAt: order.lastTradeTimestamp,
        };
    }
}
