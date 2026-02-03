import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
    IExchangeAdapter,
    ExchangeType,
    ExchangeConfig,
} from './exchange.types';
import { BinanceAdapter } from './adapters/binance.adapter';
import { BybitAdapter } from './adapters/bybit.adapter';
import { MockExchangeAdapter } from './adapters/mock.adapter';

/**
 * Exchange Factory
 * 
 * Factory pattern for creating exchange adapters.
 * Creates appropriate adapter based on exchange type and configuration.
 * 
 * Usage:
 * ```typescript
 * const adapter = await this.exchangeFactory.createAdapter('binance', { sandbox: true });
 * const ticker = await adapter.fetchTicker('BTC/USDT');
 * ```
 */
@Injectable()
export class ExchangeFactory {
    private readonly logger = new Logger(ExchangeFactory.name);

    // Cache adapters to avoid reinitializing
    private adapters: Map<string, IExchangeAdapter> = new Map();

    constructor(private readonly config: ConfigService) { }

    /**
     * Create or get cached adapter for an exchange
     */
    async createAdapter(
        exchangeType: ExchangeType,
        config?: Partial<ExchangeConfig>,
    ): Promise<IExchangeAdapter> {
        const cacheKey = this.getCacheKey(exchangeType, config);

        // Return cached adapter if exists
        const cached = this.adapters.get(cacheKey);
        if (cached) {
            return cached;
        }

        // Create new adapter
        const adapter = await this.instantiateAdapter(exchangeType, config);
        await adapter.initialize();

        // Cache it
        this.adapters.set(cacheKey, adapter);

        return adapter;
    }

    /**
     * Get the default adapter based on environment configuration
     */
    async getDefaultAdapter(): Promise<IExchangeAdapter> {
        const exchangeType = this.config.get<ExchangeType>('EXCHANGE_TYPE', 'mock');
        const sandbox = this.config.get('EXCHANGE_SANDBOX', 'true') === 'true';

        return this.createAdapter(exchangeType, { sandbox });
    }

    /**
     * Get adapter for a specific user (based on their exchange settings)
     */
    async getAdapterForUser(_userId: number, exchangeType?: ExchangeType): Promise<IExchangeAdapter> {
        // TODO: In Phase 2, fetch user's exchange credentials from database
        // For now, use default adapter

        const type = exchangeType || this.config.get<ExchangeType>('EXCHANGE_TYPE', 'mock');
        return this.createAdapter(type);
    }

    /**
     * Get mock adapter (for paper trading)
     */
    async getMockAdapter(): Promise<MockExchangeAdapter> {
        const adapter = await this.createAdapter('mock');
        return adapter as MockExchangeAdapter;
    }

    /**
     * List all available exchange types
     */
    getAvailableExchanges(): ExchangeType[] {
        return ['binance', 'bybit', 'mock'];
    }

    /**
     * Check if sandbox mode is enabled globally
     */
    isSandboxModeEnabled(): boolean {
        return this.config.get('EXCHANGE_SANDBOX', 'true') === 'true';
    }

    // ==========================================
    // Private Methods
    // ==========================================

    private async instantiateAdapter(
        exchangeType: ExchangeType,
        config?: Partial<ExchangeConfig>,
    ): Promise<IExchangeAdapter> {
        const exchangeConfig = this.buildConfig(exchangeType, config);

        switch (exchangeType) {
            case 'binance':
                return new BinanceAdapter(this.config, exchangeConfig);

            case 'bybit':
                return new BybitAdapter(this.config, exchangeConfig);

            case 'mock':
                return new MockExchangeAdapter();

            default:
                this.logger.warn(`Unknown exchange type: ${exchangeType}, falling back to mock`);
                return new MockExchangeAdapter();
        }
    }

    private buildConfig(
        exchangeType: ExchangeType,
        override?: Partial<ExchangeConfig>,
    ): ExchangeConfig {
        const apiKeyEnv = `${exchangeType.toUpperCase()}_API_KEY`;
        const secretEnv = `${exchangeType.toUpperCase()}_API_SECRET`;

        return {
            apiKey: override?.apiKey || this.config.get(apiKeyEnv),
            secret: override?.secret || this.config.get(secretEnv),
            sandbox: override?.sandbox ?? (this.config.get('EXCHANGE_SANDBOX', 'true') === 'true'),
            timeout: override?.timeout || 30000,
            rateLimit: override?.rateLimit ?? true,
        };
    }

    private getCacheKey(exchangeType: ExchangeType, config?: Partial<ExchangeConfig>): string {
        const sandbox = config?.sandbox ?? true;
        return `${exchangeType}:${sandbox ? 'sandbox' : 'live'}`;
    }
}
