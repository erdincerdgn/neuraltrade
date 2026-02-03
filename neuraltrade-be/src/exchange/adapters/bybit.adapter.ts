import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as ccxt from 'ccxt';
import { BaseCCXTAdapter } from './base-ccxt.adapter';
import { ExchangeConfig } from '../exchange.types';

/**
 * Bybit Exchange Adapter
 * 
 * CCXT-based adapter for Bybit exchange.
 * Known for fast matching engine and flexible API limits.
 * 
 * Priority 2: Preferred by professional algo traders.
 */
@Injectable()
export class BybitAdapter extends BaseCCXTAdapter {
    constructor(
        config: ConfigService,
        exchangeConfig: ExchangeConfig,
    ) {
        super(config, exchangeConfig);
    }

    getExchangeId(): 'bybit' {
        return 'bybit';
    }

    protected createExchangeInstance(): ccxt.Exchange {
        return new ccxt.bybit({
            apiKey: this.exchangeConfig.apiKey,
            secret: this.exchangeConfig.secret,
            enableRateLimit: true,
            options: {
                defaultType: 'spot', // Can be 'spot', 'linear', 'inverse'
            },
        });
    }

    /**
     * Switch to USDT perpetual futures
     */
    async setLinearMode(): Promise<void> {
        await this.ensureInitialized();
        this.exchange.options['defaultType'] = 'linear';
        await this.exchange.loadMarkets(true);
        this.logger.log('🔄 Switched to Bybit Linear (USDT) Futures');
    }

    /**
     * Switch to spot trading
     */
    async setSpotMode(): Promise<void> {
        await this.ensureInitialized();
        this.exchange.options['defaultType'] = 'spot';
        await this.exchange.loadMarkets(true);
        this.logger.log('🔄 Switched to Bybit Spot mode');
    }
}
