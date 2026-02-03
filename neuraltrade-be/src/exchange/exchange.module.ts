import { Module, Global } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ExchangeFactory } from './exchange.factory';
import { CCXTExchangeService } from './ccxt-exchange.service';
import { PriceFeedService } from './price-feed.service';

/**
 * Exchange Module
 * 
 * Provides multi-exchange connectivity via Adapter Pattern.
 * 
 * Key exports:
 * - ExchangeFactory: Create adapters for Binance, Bybit, Mock
 * - CCXTExchangeService: (Legacy) Direct CCXT access - DEPRECATED
 * - PriceFeedService: Real-time price streaming
 */
@Global()
@Module({
    imports: [ConfigModule],
    providers: [
        ExchangeFactory,
        CCXTExchangeService,  // Keep for backward compatibility
        PriceFeedService,
    ],
    exports: [
        ExchangeFactory,
        CCXTExchangeService,  // Will be removed in Phase 2
        PriceFeedService,
    ],
})
export class ExchangeModule { }
