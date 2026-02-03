import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as ccxt from 'ccxt';
import { BaseCCXTAdapter } from './base-ccxt.adapter';
import { ExchangeConfig } from '../exchange.types';

/**
 * Binance Exchange Adapter
 * 
 * CCXT-based adapter for Binance exchange.
 * Supports both Spot and USDT-M Futures trading.
 * 
 * Priority 1: World's largest exchange by volume.
 */
@Injectable()
export class BinanceAdapter extends BaseCCXTAdapter {
    constructor(
        config: ConfigService,
        exchangeConfig: ExchangeConfig,
    ) {
        super(config, exchangeConfig);
    }

    getExchangeId(): 'binance' {
        return 'binance';
    }

    protected createExchangeInstance(): ccxt.Exchange {
        return new ccxt.binance({
            apiKey: this.exchangeConfig.apiKey,
            secret: this.exchangeConfig.secret,
            enableRateLimit: true,
            options: {
                defaultType: 'spot', // Can be 'spot', 'future', 'margin'
                adjustForTimeDifference: true,
            },
        });
    }

    /**
     * Switch to futures trading mode
     */
    async setFuturesMode(): Promise<void> {
        await this.ensureInitialized();
        this.exchange.options['defaultType'] = 'future';
        await this.exchange.loadMarkets(true); // Reload markets for futures
        this.logger.log('🔄 Switched to Binance Futures mode');
    }

    /**
     * Switch to spot trading mode
     */
    async setSpotMode(): Promise<void> {
        await this.ensureInitialized();
        this.exchange.options['defaultType'] = 'spot';
        await this.exchange.loadMarkets(true);
        this.logger.log('🔄 Switched to Binance Spot mode');
    }
}
