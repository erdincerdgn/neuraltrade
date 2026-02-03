import { Controller, Get, Post, Body, Param, Query } from '@nestjs/common';
import { MarketDataService } from './market-data.service';

/**
 * Market Data Controller
 * 
 * API endpoints for:
 * - Manual OHLCV ingestion
 * - Signal history queries
 * - Data export for ML
 */
@Controller('api/v1/market-data')
export class MarketDataController {
    constructor(private readonly marketDataService: MarketDataService) { }

    // ==========================================
    // OHLCV INGESTION
    // ==========================================

    @Post('ingest/:symbol')
    async ingestOHLCV(
        @Param('symbol') symbol: string,
        @Query('timeframe') timeframe: string = '1h',
        @Query('limit') limit: string = '100',
    ) {
        const count = await this.marketDataService.ingestOHLCV(
            symbol.replace('-', '/'),
            timeframe,
            parseInt(limit),
        );
        return { symbol, timeframe, ingested: count };
    }

    @Post('ingest/batch')
    async batchIngest(
        @Body() body: { symbols: string[]; timeframe?: string },
    ) {
        const results = await this.marketDataService.batchIngestOHLCV(
            body.symbols,
            body.timeframe || '1h',
        );
        return { results };
    }

    // ==========================================
    // SIGNAL HISTORY
    // ==========================================

    @Get('signals/:symbol')
    async getSignalHistory(
        @Param('symbol') symbol: string,
        @Query('limit') limit: string = '100',
    ) {
        const signals = await this.marketDataService.getSignalHistory(
            symbol.replace('-', '/'),
            parseInt(limit),
        );
        return { symbol, count: signals.length, signals };
    }

    // ==========================================
    // DATA EXPORT (for ML)
    // ==========================================

    @Get('export/:symbol')
    async exportOHLCV(
        @Param('symbol') symbol: string,
        @Query('timeframe') timeframe: string = '1h',
        @Query('days') days: string = '30',
    ) {
        const startDate = new Date(Date.now() - parseInt(days) * 24 * 60 * 60 * 1000);
        const data = await this.marketDataService.exportOHLCVForML(
            symbol.replace('-', '/'),
            timeframe,
            startDate,
        );
        return { symbol, timeframe, count: data.length, data };
    }

    @Get('stats/:symbol')
    async getAggregatedStats(
        @Param('symbol') symbol: string,
        @Query('hours') hours: string = '24',
    ) {
        const stats = await this.marketDataService.getAggregatedStats(
            symbol.replace('-', '/'),
            parseInt(hours),
        );
        return { symbol, hours: parseInt(hours), stats };
    }

    // ==========================================
    // TRACKING STATUS
    // ==========================================

    @Get('status')
    getIngestionStatus() {
        return {
            trackedSymbols: this.marketDataService.getTrackedSymbols(),
            lastIngestion: this.marketDataService.getIngestionStatus(),
        };
    }

    @Post('track/:symbol')
    addTrackedSymbol(@Param('symbol') symbol: string) {
        this.marketDataService.addTrackedSymbol(symbol.replace('-', '/'));
        return { message: `Tracking ${symbol}`, symbols: this.marketDataService.getTrackedSymbols() };
    }
}
