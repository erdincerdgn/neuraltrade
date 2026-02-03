import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { CCXTExchangeService } from './ccxt-exchange.service';

/**
 * Price Feed Service
 * 
 * Provides real-time price updates from exchanges via CCXT.
 * Broadcasts price ticks via EventEmitter.
 */
@Injectable()
export class PriceFeedService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(PriceFeedService.name);
    private priceInterval: NodeJS.Timeout | null = null;

    // Subscribed symbols
    private subscribedSymbols: Set<string> = new Set(['BTC/USDT', 'ETH/USDT']);

    // Cached prices
    private prices: Map<string, PriceData> = new Map();

    // Update interval (ms)
    private readonly updateInterval = 5000; // 5 seconds

    constructor(
        private readonly ccxt: CCXTExchangeService,
        private readonly eventEmitter: EventEmitter2,
    ) { }

    async onModuleInit() {
        // Start after a delay to allow CCXT to initialize
        setTimeout(() => this.startPriceFeed(), 3000);
    }

    onModuleDestroy() {
        this.stopPriceFeed();
    }

    private startPriceFeed() {
        this.logger.log('📈 Starting price feed...');

        this.priceInterval = setInterval(async () => {
            await this.updatePrices();
        }, this.updateInterval);

        // Initial fetch
        this.updatePrices();
    }

    private stopPriceFeed() {
        if (this.priceInterval) {
            clearInterval(this.priceInterval);
            this.priceInterval = null;
            this.logger.log('📉 Price feed stopped');
        }
    }

    private async updatePrices() {
        for (const symbol of this.subscribedSymbols) {
            try {
                const ticker = await this.ccxt.fetchTicker(symbol);
                if (!ticker) continue;

                const priceData: PriceData = {
                    symbol,
                    price: ticker.last || 0,
                    bid: ticker.bid || 0,
                    ask: ticker.ask || 0,
                    volume: ticker.baseVolume || 0,
                    change: ticker.change || 0,
                    changePercent: ticker.percentage || 0,
                    high24h: ticker.high || 0,
                    low24h: ticker.low || 0,
                    timestamp: new Date(),
                };

                this.prices.set(symbol, priceData);

                // Emit price update event
                this.eventEmitter.emit('price.update', priceData);
            } catch (error) {
                this.logger.error(`Price fetch error for ${symbol}: ${error.message}`);
            }
        }
    }

    // ==========================================
    // PUBLIC METHODS
    // ==========================================

    subscribe(symbol: string): void {
        this.subscribedSymbols.add(symbol);
        this.logger.log(`➕ Subscribed to: ${symbol}`);
    }

    unsubscribe(symbol: string): void {
        this.subscribedSymbols.delete(symbol);
        this.logger.log(`➖ Unsubscribed from: ${symbol}`);
    }

    getPrice(symbol: string): PriceData | undefined {
        return this.prices.get(symbol);
    }

    getAllPrices(): PriceData[] {
        return Array.from(this.prices.values());
    }

    getSubscribedSymbols(): string[] {
        return Array.from(this.subscribedSymbols);
    }
}

// ==========================================
// TYPES
// ==========================================

interface PriceData {
    symbol: string;
    price: number;
    bid: number;
    ask: number;
    volume: number;
    change: number;
    changePercent: number;
    high24h: number;
    low24h: number;
    timestamp: Date;
}
