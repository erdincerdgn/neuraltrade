import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { EventEmitter2 } from '@nestjs/event-emitter';
import * as ccxt from 'ccxt';

/**
 * CCXT Exchange Service
 * 
 * Unified exchange interface using CCXT library.
 * Supports: Binance, Bybit, Kraken, Alpaca, and 100+ exchanges
 * 
 * Mode:
 * - Sandbox: Use exchange testnet (paper trading)
 * - Live: Real trading (requires API keys)
 */
@Injectable()
export class CCXTExchangeService implements OnModuleInit {
    private readonly logger = new Logger(CCXTExchangeService.name);

    // Exchange instances
    private exchanges: Map<string, ccxt.Exchange> = new Map();

    // Default exchange
    private defaultExchange: ccxt.Exchange | null = null;

    // Sandbox mode
    private sandboxMode = true;

    constructor(
        private readonly config: ConfigService,
        private readonly eventEmitter: EventEmitter2,
    ) { }

    async onModuleInit() {
        await this.initializeExchanges();
    }

    // ==========================================
    // INITIALIZATION
    // ==========================================

    private async initializeExchanges() {
        this.sandboxMode = this.config.get('EXCHANGE_SANDBOX', 'true') === 'true';

        // Initialize Binance (default)
        await this.addExchange('binance', {
            apiKey: this.config.get('BINANCE_API_KEY'),
            secret: this.config.get('BINANCE_API_SECRET'),
            sandbox: this.sandboxMode,
        });

        // Initialize Bybit (optional)
        const bybitKey = this.config.get('BYBIT_API_KEY');
        if (bybitKey) {
            await this.addExchange('bybit', {
                apiKey: bybitKey,
                secret: this.config.get('BYBIT_API_SECRET'),
                sandbox: this.sandboxMode,
            });
        }

        this.logger.log(`🏦 CCXT initialized in ${this.sandboxMode ? 'SANDBOX' : 'LIVE'} mode`);
    }

    async addExchange(
        exchangeId: string,
        options: { apiKey?: string; secret?: string; sandbox?: boolean },
    ): Promise<boolean> {
        try {
            const ExchangeClass = ccxt[exchangeId as keyof typeof ccxt];
            if (!ExchangeClass) {
                this.logger.error(`Exchange not found: ${exchangeId}`);
                return false;
            }

            const exchange = new (ExchangeClass as any)({
                apiKey: options.apiKey,
                secret: options.secret,
                enableRateLimit: true,
                options: {
                    defaultType: 'spot',
                },
            });

            // Enable sandbox/testnet mode
            if (options.sandbox) {
                exchange.setSandboxMode(true);
            }

            // Load markets
            await exchange.loadMarkets();

            this.exchanges.set(exchangeId, exchange);

            // Set as default if first exchange
            if (!this.defaultExchange) {
                this.defaultExchange = exchange;
            }

            this.logger.log(`✅ Exchange added: ${exchangeId} (${Object.keys(exchange.markets).length} markets)`);
            return true;
        } catch (error) {
            this.logger.error(`Failed to add exchange ${exchangeId}: ${error.message}`);
            return false;
        }
    }

    getExchange(exchangeId?: string): ccxt.Exchange | null {
        if (exchangeId) {
            return this.exchanges.get(exchangeId) || null;
        }
        return this.defaultExchange;
    }

    // ==========================================
    // MARKET DATA
    // ==========================================

    async fetchTicker(symbol: string, exchangeId?: string): Promise<ccxt.Ticker | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            return await exchange.fetchTicker(symbol);
        } catch (error) {
            this.logger.error(`fetchTicker error: ${error.message}`);
            return null;
        }
    }

    async fetchOrderBook(symbol: string, limit = 20, exchangeId?: string): Promise<ccxt.OrderBook | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            return await exchange.fetchOrderBook(symbol, limit);
        } catch (error) {
            this.logger.error(`fetchOrderBook error: ${error.message}`);
            return null;
        }
    }

    async fetchOHLCV(
        symbol: string,
        timeframe = '1h',
        limit = 100,
        exchangeId?: string,
    ): Promise<ccxt.OHLCV[] | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            return await exchange.fetchOHLCV(symbol, timeframe, undefined, limit);
        } catch (error) {
            this.logger.error(`fetchOHLCV error: ${error.message}`);
            return null;
        }
    }

    // ==========================================
    // TRADING
    // ==========================================

    async createOrder(
        symbol: string,
        type: 'market' | 'limit',
        side: 'buy' | 'sell',
        amount: number,
        price?: number,
        exchangeId?: string,
    ): Promise<ccxt.Order | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            const order = await exchange.createOrder(symbol, type, side, amount, price);

            this.eventEmitter.emit('order.created', {
                exchangeId: exchangeId || 'binance',
                orderId: order.id,
                symbol,
                type,
                side,
                amount,
                price,
            });

            this.logger.log(`📝 Order created: ${order.id} - ${symbol} ${side} ${amount}`);
            return order;
        } catch (error) {
            this.logger.error(`createOrder error: ${error.message}`);
            this.eventEmitter.emit('order.error', { symbol, error: error.message });
            return null;
        }
    }

    async cancelOrder(orderId: string, symbol: string, exchangeId?: string): Promise<boolean> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return false;

        try {
            await exchange.cancelOrder(orderId, symbol);
            this.eventEmitter.emit('order.cancelled', { orderId, symbol });
            this.logger.log(`❌ Order cancelled: ${orderId}`);
            return true;
        } catch (error) {
            this.logger.error(`cancelOrder error: ${error.message}`);
            return false;
        }
    }

    async fetchOrder(orderId: string, symbol: string, exchangeId?: string): Promise<ccxt.Order | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            return await exchange.fetchOrder(orderId, symbol);
        } catch (error) {
            this.logger.error(`fetchOrder error: ${error.message}`);
            return null;
        }
    }

    async fetchOpenOrders(symbol?: string, exchangeId?: string): Promise<ccxt.Order[]> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return [];

        try {
            return await exchange.fetchOpenOrders(symbol);
        } catch (error) {
            this.logger.error(`fetchOpenOrders error: ${error.message}`);
            return [];
        }
    }

    // ==========================================
    // ACCOUNT / BALANCE
    // ==========================================

    async fetchBalance(exchangeId?: string): Promise<ccxt.Balances | null> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return null;

        try {
            return await exchange.fetchBalance();
        } catch (error) {
            this.logger.error(`fetchBalance error: ${error.message}`);
            return null;
        }
    }

    async fetchPositions(symbol?: string, exchangeId?: string): Promise<any[]> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return [];

        try {
            if (exchange.has['fetchPositions']) {
                return await (exchange as any).fetchPositions(symbol ? [symbol] : undefined);
            }
            return [];
        } catch (error) {
            this.logger.error(`fetchPositions error: ${error.message}`);
            return [];
        }
    }

    // ==========================================
    // UTILITY
    // ==========================================

    getSupportedExchanges(): string[] {
        return Array.from(this.exchanges.keys());
    }

    isSandboxMode(): boolean {
        return this.sandboxMode;
    }

    async getMarkets(exchangeId?: string): Promise<ccxt.Market[]> {
        const exchange = this.getExchange(exchangeId);
        if (!exchange) return [];
        return Object.values(exchange.markets);
    }
}
