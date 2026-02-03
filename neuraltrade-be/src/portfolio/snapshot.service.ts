import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../core/prisma/prisma.service';

/**
 * Portfolio Snapshot
 */
export interface PortfolioSnapshot {
    id: number;
    portfolioId: number;
    userId: number;
    timestamp: Date;
    totalValue: number;
    positionsValue: number;
    positionCount: number;
}

/**
 * Portfolio Snapshot Service
 * 
 * Phase 9: Creates and manages historical portfolio snapshots
 * - Daily snapshots via cron job
 * - On-demand snapshots for specific events
 * - Historical data retrieval
 */
@Injectable()
export class PortfolioSnapshotService {
    private readonly logger = new Logger(PortfolioSnapshotService.name);

    // In-memory cache for snapshots (would be persisted to DB in production)
    private snapshots: Map<number, PortfolioSnapshot[]> = new Map();

    constructor(private readonly prisma: PrismaService) { }

    /**
     * Create daily snapshots for all portfolios
     * Runs at midnight every day
     */
    @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT)
    async createDailySnapshots(): Promise<void> {
        this.logger.log('Starting daily portfolio snapshots...');

        const portfolios = await this.prisma.portfolio.findMany({
            include: { positions: true },
        });

        let successCount = 0;
        let errorCount = 0;

        for (const portfolio of portfolios) {
            try {
                await this.createSnapshot(portfolio.id, portfolio.userId);
                successCount++;
            } catch (error) {
                this.logger.error(`Failed to snapshot portfolio ${portfolio.id}: ${(error as Error).message}`);
                errorCount++;
            }
        }

        this.logger.log(`Daily snapshots complete: ${successCount} success, ${errorCount} errors`);
    }

    /**
     * Create a single portfolio snapshot
     */
    async createSnapshot(portfolioId: number, userId: number): Promise<PortfolioSnapshot> {
        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
            include: { positions: true },
        });

        if (!portfolio) {
            throw new Error(`Portfolio ${portfolioId} not found`);
        }

        // Calculate positions value using marketValue from positions
        const positionsValue = portfolio.positions.reduce(
            (sum, pos) => sum + (pos.marketValue?.toNumber() || 0),
            0,
        );

        const snapshot: PortfolioSnapshot = {
            id: Date.now(),
            portfolioId,
            userId,
            timestamp: new Date(),
            totalValue: positionsValue,
            positionsValue,
            positionCount: portfolio.positions.length,
        };

        // Store in memory
        const existing = this.snapshots.get(portfolioId) || [];
        existing.push(snapshot);
        this.snapshots.set(portfolioId, existing);

        this.logger.debug(`Snapshot created for portfolio ${portfolioId}: $${positionsValue.toFixed(2)}`);

        return snapshot;
    }

    /**
     * Get latest snapshot for a portfolio
     */
    async getLatestSnapshot(portfolioId: number): Promise<PortfolioSnapshot | null> {
        const history = this.snapshots.get(portfolioId);
        if (!history || history.length === 0) return null;
        return history[history.length - 1];
    }

    /**
     * Get historical snapshots for a portfolio
     */
    async getHistory(
        portfolioId: number,
        startDate: Date,
        endDate: Date,
    ): Promise<PortfolioSnapshot[]> {
        const history = this.snapshots.get(portfolioId) || [];
        return history.filter(
            (s) => s.timestamp >= startDate && s.timestamp <= endDate,
        );
    }

    /**
     * Get portfolio performance metrics from snapshots
     */
    async getPerformanceMetrics(portfolioId: number, days: number = 30): Promise<{
        totalReturn: number;
        avgDailyReturn: number;
        maxDrawdown: number;
        sharpeRatio: number;
        volatility: number;
    }> {
        const endDate = new Date();
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);

        const snapshots = await this.getHistory(portfolioId, startDate, endDate);

        if (snapshots.length < 2) {
            return {
                totalReturn: 0,
                avgDailyReturn: 0,
                maxDrawdown: 0,
                sharpeRatio: 0,
                volatility: 0,
            };
        }

        // Calculate daily returns
        const dailyReturns: number[] = [];
        for (let i = 1; i < snapshots.length; i++) {
            const prevValue = snapshots[i - 1].totalValue;
            const currValue = snapshots[i].totalValue;
            if (prevValue > 0) {
                dailyReturns.push((currValue - prevValue) / prevValue);
            }
        }

        if (dailyReturns.length === 0) {
            return { totalReturn: 0, avgDailyReturn: 0, maxDrawdown: 0, sharpeRatio: 0, volatility: 0 };
        }

        // Calculate metrics
        const totalReturn = dailyReturns.reduce((sum, r) => sum + r, 0);
        const avgDailyReturn = totalReturn / dailyReturns.length;

        // Volatility (std dev of returns)
        const variance = dailyReturns.reduce(
            (sum, r) => sum + Math.pow(r - avgDailyReturn, 2),
            0,
        ) / dailyReturns.length;
        const volatility = Math.sqrt(variance);

        // Sharpe Ratio (assuming 0% risk-free rate)
        const sharpeRatio = volatility > 0 ? avgDailyReturn / volatility : 0;

        // Max Drawdown
        let peak = snapshots[0].totalValue;
        let maxDrawdown = 0;
        for (const snapshot of snapshots) {
            const value = snapshot.totalValue;
            if (value > peak) peak = value;
            const drawdown = peak > 0 ? (peak - value) / peak : 0;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }

        return {
            totalReturn: totalReturn * 100,
            avgDailyReturn: avgDailyReturn * 100,
            maxDrawdown: maxDrawdown * 100,
            sharpeRatio,
            volatility: volatility * 100,
        };
    }
}
