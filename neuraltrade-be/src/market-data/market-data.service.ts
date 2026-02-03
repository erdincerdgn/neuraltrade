import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../core/prisma/prisma.service';
import { ExchangeFactory } from '../exchange/exchange.factory';
import { OHLCV } from '../exchange/exchange.types';

/**
 * Market Data Service
 * 
 * Phase 2: TimescaleDB data pipeline for:
 * - OHLCV ingestion from exchanges to hypertables
 * - Signal history persistence
 * - Data export for Python ML training
 */
@Injectable()
export class MarketDataService {
    private readonly logger = new Logger(MarketDataService.name);

    // Tracked symbols for automatic ingestion
    private trackedSymbols: Set<string> = new Set(['BTC/USDT', 'ETH/USDT']);

    // Ingestion status
    private lastIngestion: Map<string, Date> = new Map();

    constructor(
        private readonly prisma: PrismaService,
        private readonly exchangeFactory: ExchangeFactory,
    ) { }

    // ==========================================
    // OHLCV INGESTION
    // ==========================================

    /**
     * Ingest OHLCV data for a symbol
     */
    async ingestOHLCV(
        symbol: string,
        timeframe: string = '1h',
        limit: number = 100,
    ): Promise<number> {
        try {
            const adapter = await this.exchangeFactory.getDefaultAdapter();
            const candles = await adapter.fetchOHLCV(symbol, timeframe, limit);

            if (candles.length === 0) {
                this.logger.warn(`No OHLCV data received for ${symbol}`);
                return 0;
            }

            // Upsert candles to TimescaleDB
            let inserted = 0;
            for (const candle of candles) {
                try {
                    await this.upsertOHLCV(symbol, timeframe, candle);
                    inserted++;
                } catch (error) {
                    // Skip duplicates
                }
            }

            this.lastIngestion.set(symbol, new Date());
            this.logger.log(`📊 Ingested ${inserted} candles for ${symbol} (${timeframe})`);

            return inserted;
        } catch (error) {
            this.logger.error(`OHLCV ingestion failed for ${symbol}: ${error.message}`);
            return 0;
        }
    }

    /**
     * Upsert single OHLCV candle
     */
    private async upsertOHLCV(
        symbol: string,
        timeframe: string,
        candle: OHLCV,
    ): Promise<void> {
        // Using raw SQL for TimescaleDB hypertable upsert
        await this.prisma.$executeRaw`
            INSERT INTO ohlcv_data (timestamp, symbol, timeframe, exchange, open, high, low, close, volume)
            VALUES (
                to_timestamp(${candle.timestamp / 1000}),
                ${symbol},
                ${timeframe},
                'binance',
                ${parseFloat(candle.open)},
                ${parseFloat(candle.high)},
                ${parseFloat(candle.low)},
                ${parseFloat(candle.close)},
                ${parseFloat(candle.volume)}
            )
            ON CONFLICT (symbol, exchange, timeframe, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        `;
    }

    /**
     * Batch ingest for multiple symbols
     */
    async batchIngestOHLCV(
        symbols: string[],
        timeframe: string = '1h',
    ): Promise<Record<string, number>> {
        const results: Record<string, number> = {};

        for (const symbol of symbols) {
            results[symbol] = await this.ingestOHLCV(symbol, timeframe);
            // Small delay to avoid rate limiting
            await new Promise(r => setTimeout(r, 100));
        }

        return results;
    }

    // ==========================================
    // SIGNAL HISTORY
    // ==========================================

    /**
     * Persist AI signal to signal_history hypertable
     */
    async persistSignal(signal: {
        symbol: string;
        action: string;
        confidence: number;
        models?: string[];
        reasoning?: string;
    }): Promise<void> {
        try {
            await this.prisma.$executeRaw`
                INSERT INTO signal_history (time, symbol, action, confidence, models, reasoning)
                VALUES (
                    NOW(),
                    ${signal.symbol},
                    ${signal.action},
                    ${signal.confidence},
                    ${signal.models?.join(',') || ''},
                    ${signal.reasoning || ''}
                )
            `;

            this.logger.debug(`📝 Signal persisted: ${signal.symbol} ${signal.action}`);
        } catch (error) {
            this.logger.error(`Failed to persist signal: ${error.message}`);
        }
    }

    /**
     * Get signal history for a symbol
     */
    async getSignalHistory(
        symbol: string,
        limit: number = 100,
    ): Promise<SignalRecord[]> {
        const signals = await this.prisma.$queryRaw<SignalRecord[]>`
            SELECT 
                time,
                symbol,
                action,
                confidence,
                models,
                reasoning
            FROM signal_history
            WHERE symbol = ${symbol}
            ORDER BY time DESC
            LIMIT ${limit}
        `;

        return signals;
    }

    // ==========================================
    // DATA EXPORT (for Python ML)
    // ==========================================

    /**
     * Export OHLCV data for ML training
     */
    async exportOHLCVForML(
        symbol: string,
        timeframe: string = '1h',
        startDate?: Date,
        endDate?: Date,
    ): Promise<OHLCVExport[]> {
        const start = startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days
        const end = endDate || new Date();

        const data = await this.prisma.$queryRaw<OHLCVExport[]>`
            SELECT 
                time,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_data
            WHERE symbol = ${symbol}
                AND timeframe = ${timeframe}
                AND time BETWEEN ${start} AND ${end}
            ORDER BY time ASC
        `;

        return data;
    }

    /**
     * Get aggregated stats for ML features
     */
    async getAggregatedStats(
        symbol: string,
        hours: number = 24,
    ): Promise<AggregatedStats | null> {
        const result = await this.prisma.$queryRaw<AggregatedStats[]>`
            SELECT 
                COUNT(*) as candle_count,
                AVG(close) as avg_price,
                MAX(high) as max_high,
                MIN(low) as min_low,
                SUM(volume) as total_volume,
                STDDEV(close) as price_volatility
            FROM ohlcv_data
            WHERE symbol = ${symbol}
                AND time > NOW() - INTERVAL '${hours} hours'
        `;

        return result[0] || null;
    }

    // ==========================================
    // PERIODIC INGESTION (Cron)
    // ==========================================

    /**
     * Hourly OHLCV ingestion for tracked symbols
     */
    @Cron(CronExpression.EVERY_HOUR)
    async periodicIngestion(): Promise<void> {
        if (this.trackedSymbols.size === 0) return;

        this.logger.log(`⏰ Periodic ingestion for ${this.trackedSymbols.size} symbols`);

        for (const symbol of this.trackedSymbols) {
            await this.ingestOHLCV(symbol, '1h', 24);
        }
    }

    // ==========================================
    // SYMBOL TRACKING
    // ==========================================

    addTrackedSymbol(symbol: string): void {
        this.trackedSymbols.add(symbol);
        this.logger.log(`➕ Tracking symbol: ${symbol}`);
    }

    removeTrackedSymbol(symbol: string): void {
        this.trackedSymbols.delete(symbol);
        this.logger.log(`➖ Untracked symbol: ${symbol}`);
    }

    getTrackedSymbols(): string[] {
        return Array.from(this.trackedSymbols);
    }

    getIngestionStatus(): Record<string, string | null> {
        const status: Record<string, string | null> = {};
        for (const symbol of this.trackedSymbols) {
            const lastTime = this.lastIngestion.get(symbol);
            status[symbol] = lastTime ? lastTime.toISOString() : null;
        }
        return status;
    }
}

// ==========================================
// TYPES
// ==========================================

export interface SignalRecord {
    time: Date;
    symbol: string;
    action: string;
    confidence: number;
    models: string;
    reasoning: string;
}

export interface OHLCVExport {
    time: Date;
    symbol: string;
    timeframe: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface AggregatedStats {
    candle_count: number;
    avg_price: number;
    max_high: number;
    min_low: number;
    total_volume: number;
    price_volatility: number;
}
