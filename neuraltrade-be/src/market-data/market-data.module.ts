import { Module } from '@nestjs/common';
import { MarketDataService } from './market-data.service';
import { MarketDataController } from './market-data.controller';

/**
 * Market Data Module
 * 
 * Phase 2: TimescaleDB data pipeline
 * - OHLCV ingestion from exchanges
 * - Signal history persistence
 * - Data export API for Python ML
 */
@Module({
    imports: [],
    controllers: [MarketDataController],
    providers: [MarketDataService],
    exports: [MarketDataService],
})
export class MarketDataModule { }
