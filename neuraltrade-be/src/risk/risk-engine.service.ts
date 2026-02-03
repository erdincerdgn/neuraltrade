import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import Decimal from 'decimal.js';
import {
    PositionSizingMethod,
    PositionSizeRequest,
    PositionSizeResult,
    ExposureCheck,
    ExposureResult,
    RiskProfile,
    RiskLevels,
    RiskMetrics,
    RiskTolerance,
    StopLossType,
    DEFAULT_RISK_PROFILES,
} from './risk.types';

/**
 * Risk Engine Service
 * 
 * Core risk management functionality:
 * - Position sizing (Kelly, volatility-based, fixed fractional)
 * - Exposure limit enforcement
 * - Stop-loss/Take-profit calculation
 * - Portfolio risk metrics
 */
@Injectable()
export class RiskEngineService {
    private readonly logger = new Logger(RiskEngineService.name);

    constructor(
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
    ) { }

    // ==========================================
    // POSITION SIZING
    // ==========================================

    /**
     * Calculate optimal position size based on risk profile
     */
    calculatePositionSize(
        request: PositionSizeRequest,
        profile: RiskProfile,
    ): PositionSizeResult {
        const portfolioValue = new Decimal(request.portfolioValue);
        const price = new Decimal(request.price);

        let result: PositionSizeResult;

        switch (profile.sizingMethod) {
            case PositionSizingMethod.KELLY_CRITERION:
                result = this.kellyPositionSize(request, profile, portfolioValue, price);
                break;
            case PositionSizingMethod.VOLATILITY_BASED:
                result = this.volatilityPositionSize(request, profile, portfolioValue, price);
                break;
            case PositionSizingMethod.FIXED_AMOUNT:
                result = this.fixedAmountSize(request, profile, portfolioValue, price);
                break;
            case PositionSizingMethod.FIXED_FRACTIONAL:
            default:
                result = this.fixedFractionalSize(request, profile, portfolioValue, price);
        }

        // Apply position size cap from exposure limits
        const maxPositionValue = portfolioValue.mul(profile.exposureLimits.maxPositionPercent).div(100);
        const positionValue = new Decimal(result.dollarAmount);

        if (positionValue.gt(maxPositionValue)) {
            const cappedQuantity = maxPositionValue.div(price);
            result = {
                ...result,
                quantity: cappedQuantity.toFixed(8),
                dollarAmount: maxPositionValue.toFixed(2),
                reasoning: `${result.reasoning} (capped by ${profile.exposureLimits.maxPositionPercent}% max position limit)`,
            };
        }

        return result;
    }

    /**
     * Fixed Fractional: Risk fixed % of portfolio per trade
     */
    private fixedFractionalSize(
        request: PositionSizeRequest,
        profile: RiskProfile,
        portfolioValue: Decimal,
        price: Decimal,
    ): PositionSizeResult {
        const riskPercent = new Decimal(request.maxRiskPerTrade || profile.maxRiskPerTrade);
        const riskAmount = portfolioValue.mul(riskPercent).div(100);

        // Calculate position size based on stop loss distance
        const stopLossPercent = profile.stopLossConfig.percent || 2;
        const stopLossDistance = price.mul(stopLossPercent).div(100);

        // Position size = Risk Amount / Stop Loss Distance
        const quantity = riskAmount.div(stopLossDistance);
        const dollarAmount = quantity.mul(price);

        return {
            method: PositionSizingMethod.FIXED_FRACTIONAL,
            quantity: quantity.toFixed(8),
            dollarAmount: dollarAmount.toFixed(2),
            riskPercent: riskPercent.toString(),
            stopLossPrice: price.minus(stopLossDistance).toFixed(8),
            takeProfitPrice: price.plus(stopLossDistance.mul(profile.takeProfitConfig.riskRewardRatio)).toFixed(8),
            reasoning: `Fixed ${riskPercent}% risk with ${stopLossPercent}% stop loss`,
        };
    }

    /**
     * Kelly Criterion: Optimal growth position sizing
     * Kelly % = W - [(1-W) / R]
     * W = Win rate, R = Win/Loss ratio
     */
    private kellyPositionSize(
        request: PositionSizeRequest,
        profile: RiskProfile,
        portfolioValue: Decimal,
        price: Decimal,
    ): PositionSizeResult {
        // Default values if not provided
        const winRate = request.winRate || 0.55;
        const avgWin = new Decimal(request.avgWin || '1.5');
        const avgLoss = new Decimal(request.avgLoss || '1');

        // Calculate Kelly percentage
        const winLossRatio = avgWin.div(avgLoss);
        const kellyPercent = new Decimal(winRate).minus(
            new Decimal(1 - winRate).div(winLossRatio)
        );

        // Use half-Kelly for safety
        const halfKelly = kellyPercent.div(2).mul(100);

        // Cap at max risk per trade
        const actualPercent = Decimal.min(halfKelly, new Decimal(profile.maxRiskPerTrade));

        if (actualPercent.lte(0)) {
            return {
                method: PositionSizingMethod.KELLY_CRITERION,
                quantity: '0',
                dollarAmount: '0',
                riskPercent: '0',
                reasoning: 'Kelly criterion negative - edge insufficient for position',
            };
        }

        const dollarAmount = portfolioValue.mul(actualPercent).div(100);
        const quantity = dollarAmount.div(price);

        return {
            method: PositionSizingMethod.KELLY_CRITERION,
            quantity: quantity.toFixed(8),
            dollarAmount: dollarAmount.toFixed(2),
            riskPercent: actualPercent.toFixed(2),
            reasoning: `Half-Kelly: ${halfKelly.toFixed(2)}% (W:${(winRate * 100).toFixed(0)}%, R:${winLossRatio.toFixed(2)})`,
        };
    }

    /**
     * Volatility-Based: Adjust size based on ATR
     */
    private volatilityPositionSize(
        request: PositionSizeRequest,
        profile: RiskProfile,
        portfolioValue: Decimal,
        price: Decimal,
    ): PositionSizeResult {
        const volatility = new Decimal(request.volatility || '0.02'); // Default 2% volatility
        const riskPercent = new Decimal(profile.maxRiskPerTrade);
        const riskAmount = portfolioValue.mul(riskPercent).div(100);

        // ATR multiplier for stop loss
        const atrMultiplier = profile.stopLossConfig.atrMultiplier || 2;
        const stopLossDistance = volatility.mul(price).mul(atrMultiplier);

        // Position size = Risk Amount / Stop Loss Distance
        const quantity = riskAmount.div(stopLossDistance);
        const dollarAmount = quantity.mul(price);

        return {
            method: PositionSizingMethod.VOLATILITY_BASED,
            quantity: quantity.toFixed(8),
            dollarAmount: dollarAmount.toFixed(2),
            riskPercent: riskPercent.toString(),
            stopLossPrice: price.minus(stopLossDistance).toFixed(8),
            takeProfitPrice: price.plus(stopLossDistance.mul(profile.takeProfitConfig.riskRewardRatio)).toFixed(8),
            reasoning: `Volatility-adjusted: ${atrMultiplier}x ATR stop (${volatility.mul(100).toFixed(1)}% volatility)`,
        };
    }

    /**
     * Fixed Amount: Same dollar amount per trade
     */
    private fixedAmountSize(
        _request: PositionSizeRequest,
        _profile: RiskProfile,
        portfolioValue: Decimal,
        price: Decimal,
    ): PositionSizeResult {
        // Fixed amount = 1% of starting capital as default
        const dollarAmount = portfolioValue.mul(0.01);
        const quantity = dollarAmount.div(price);
        const riskPercent = dollarAmount.div(portfolioValue).mul(100);

        return {
            method: PositionSizingMethod.FIXED_AMOUNT,
            quantity: quantity.toFixed(8),
            dollarAmount: dollarAmount.toFixed(2),
            riskPercent: riskPercent.toFixed(2),
            reasoning: 'Fixed dollar amount position sizing',
        };
    }

    // ==========================================
    // EXPOSURE LIMITS
    // ==========================================

    /**
     * Check if a proposed trade would breach exposure limits
     */
    async checkExposure(
        _userId: number,
        portfolioId: number,
        check: ExposureCheck,
        profile: RiskProfile,
    ): Promise<ExposureResult> {
        // Get current positions
        const positions = await this.prisma.position.findMany({
            where: { portfolioId },
        });

        // Get portfolio value
        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
        });

        if (!portfolio) {
            return {
                allowed: false,
                currentExposure: '0',
                proposedExposure: '0',
                limitBreached: 'Portfolio not found',
                reasoning: 'Portfolio does not exist',
            };
        }

        const portfolioValue = new Decimal(portfolio.totalValue?.toString() || '0');

        // Calculate current exposure (using avgCost field from Position model)
        const currentExposure = positions.reduce(
            (sum, pos) => sum.plus(new Decimal(pos.quantity.toString()).mul(pos.avgCost.toString())),
            new Decimal(0)
        );
        const currentExposurePercent = currentExposure.div(portfolioValue).mul(100);

        // Calculate proposed exposure
        const proposedAmount = new Decimal(check.proposedAmount);
        const newExposure = check.side === 'BUY'
            ? currentExposure.plus(proposedAmount)
            : currentExposure.minus(proposedAmount);
        const proposedExposurePercent = newExposure.div(portfolioValue).mul(100);

        // Check per-position limit
        const positionPercent = proposedAmount.div(portfolioValue).mul(100);
        if (positionPercent.gt(profile.exposureLimits.maxPositionPercent)) {
            return {
                allowed: false,
                currentExposure: currentExposurePercent.toFixed(2),
                proposedExposure: proposedExposurePercent.toFixed(2),
                limitBreached: 'maxPositionPercent',
                reasoning: `Position ${positionPercent.toFixed(1)}% exceeds ${profile.exposureLimits.maxPositionPercent}% limit`,
            };
        }

        // Check total exposure limit
        if (proposedExposurePercent.gt(profile.exposureLimits.maxTotalExposure)) {
            return {
                allowed: false,
                currentExposure: currentExposurePercent.toFixed(2),
                proposedExposure: proposedExposurePercent.toFixed(2),
                limitBreached: 'maxTotalExposure',
                reasoning: `Total exposure ${proposedExposurePercent.toFixed(1)}% exceeds ${profile.exposureLimits.maxTotalExposure}% limit`,
            };
        }

        return {
            allowed: true,
            currentExposure: currentExposurePercent.toFixed(2),
            proposedExposure: proposedExposurePercent.toFixed(2),
            reasoning: 'Trade within exposure limits',
        };
    }

    // ==========================================
    // STOP-LOSS / TAKE-PROFIT
    // ==========================================

    /**
     * Calculate risk levels for a trade
     */
    calculateRiskLevels(
        entryPrice: string,
        quantity: string,
        side: 'BUY' | 'SELL',
        profile: RiskProfile,
        volatility?: string,
    ): RiskLevels {
        const entry = new Decimal(entryPrice);
        const qty = new Decimal(quantity);
        let stopLossDistance: Decimal;

        // Calculate stop loss based on config
        switch (profile.stopLossConfig.type) {
            case StopLossType.ATR_BASED:
                const vol = new Decimal(volatility || '0.02');
                const multiplier = profile.stopLossConfig.atrMultiplier || 2;
                stopLossDistance = entry.mul(vol).mul(multiplier);
                break;
            case StopLossType.TRAILING:
                stopLossDistance = entry.mul(profile.stopLossConfig.trailingPercent || 5).div(100);
                break;
            case StopLossType.FIXED_PERCENT:
            default:
                stopLossDistance = entry.mul(profile.stopLossConfig.percent || 2).div(100);
        }

        const stopLoss = side === 'BUY'
            ? entry.minus(stopLossDistance)
            : entry.plus(stopLossDistance);

        const takeProfitDistance = stopLossDistance.mul(profile.takeProfitConfig.riskRewardRatio);
        const takeProfit = side === 'BUY'
            ? entry.plus(takeProfitDistance)
            : entry.minus(takeProfitDistance);

        const riskAmount = stopLossDistance.mul(qty);
        const rewardAmount = takeProfitDistance.mul(qty);

        return {
            entryPrice: entry.toFixed(8),
            stopLoss: stopLoss.toFixed(8),
            takeProfit: takeProfit.toFixed(8),
            riskAmount: riskAmount.toFixed(2),
            rewardAmount: rewardAmount.toFixed(2),
            riskRewardRatio: profile.takeProfitConfig.riskRewardRatio.toString(),
        };
    }

    // ==========================================
    // RISK METRICS
    // ==========================================

    /**
     * Get comprehensive risk metrics for a portfolio
     */
    async getPortfolioRiskMetrics(
        _userId: number,
        portfolioId: number,
    ): Promise<RiskMetrics> {
        // Get portfolio
        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
        });

        if (!portfolio) {
            throw new Error('Portfolio not found');
        }

        // Get positions separately
        const positions = await this.prisma.position.findMany({
            where: { portfolioId },
        });

        const portfolioValue = new Decimal(portfolio.totalValue?.toString() || '0');

        const totalExposure = positions.reduce(
            (sum, pos) => sum.plus(new Decimal(pos.quantity.toString()).mul(pos.avgCost.toString()).abs()),
            new Decimal(0)
        );

        const unrealizedPnL = positions.reduce(
            (sum, pos) => sum.plus(pos.unrealizedPnL?.toString() || '0'),
            new Decimal(0)
        );

        // Get daily PnL from cache
        const dailyPnLKey = `daily_pnl:${portfolioId}:${new Date().toISOString().split('T')[0]}`;
        const cachedDailyPnL = await this.redis.get<string>(dailyPnLKey);
        const dailyPnL = new Decimal(cachedDailyPnL || '0');

        // Find largest position
        let largestPosition = null;
        if (positions.length > 0) {
            const sorted = [...positions].sort((a, b) => {
                const aValue = new Decimal(a.quantity.toString()).mul(a.avgCost.toString()).abs();
                const bValue = new Decimal(b.quantity.toString()).mul(b.avgCost.toString()).abs();
                return bValue.minus(aValue).toNumber();
            });
            const largest = sorted[0];
            const largestValue = new Decimal(largest.quantity.toString()).mul(largest.avgCost.toString()).abs();
            largestPosition = {
                symbol: largest.symbol,
                percent: portfolioValue.gt(0) ? largestValue.div(portfolioValue).mul(100).toFixed(2) : '0',
            };
        }

        return {
            portfolioValue: portfolioValue.toFixed(2),
            totalExposure: totalExposure.toFixed(2),
            exposurePercent: portfolioValue.gt(0) ? totalExposure.div(portfolioValue).mul(100).toFixed(2) : '0',
            unrealizedPnL: unrealizedPnL.toFixed(2),
            dailyPnL: dailyPnL.toFixed(2),
            dailyPnLPercent: portfolioValue.gt(0) ? dailyPnL.div(portfolioValue).mul(100).toFixed(2) : '0',
            maxDrawdown: '0',
            currentDrawdown: '0',
            positionCount: positions.length,
            largestPosition,
        };
    }

    // ==========================================
    // PROFILE MANAGEMENT
    // ==========================================

    /**
     * Get risk profile for a user (from cache or default)
     */
    async getUserRiskProfile(userId: number): Promise<RiskProfile> {
        const cacheKey = `risk_profile:${userId}`;
        const cached = await this.redis.get<RiskProfile>(cacheKey);

        if (cached) {
            return cached;
        }

        // Default to MODERATE
        return DEFAULT_RISK_PROFILES[RiskTolerance.MODERATE];
    }

    /**
     * Set user's risk profile
     */
    async setUserRiskProfile(userId: number, profile: RiskProfile): Promise<void> {
        const cacheKey = `risk_profile:${userId}`;
        await this.redis.set(cacheKey, profile, 86400 * 30); // 30 days
        this.logger.log(`Risk profile set for user ${userId}: ${profile.tolerance}`);
    }
}
