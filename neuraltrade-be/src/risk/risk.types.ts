/**
 * Risk Module Types
 * 
 * Comprehensive type definitions for risk management.
 */

// ==========================================
// POSITION SIZING
// ==========================================

export enum PositionSizingMethod {
    FIXED_FRACTIONAL = 'FIXED_FRACTIONAL',  // Fixed % of portfolio
    KELLY_CRITERION = 'KELLY_CRITERION',     // Optimal growth
    VOLATILITY_BASED = 'VOLATILITY_BASED',   // ATR-adjusted
    FIXED_AMOUNT = 'FIXED_AMOUNT',           // Fixed USD amount
}

export interface PositionSizeRequest {
    portfolioValue: string;          // Total portfolio in quote currency
    symbol: string;
    price: string;
    side: 'BUY' | 'SELL';
    winRate?: number;                // For Kelly: historical win rate (0-1)
    avgWin?: string;                 // Average winning trade
    avgLoss?: string;                // Average losing trade
    volatility?: string;             // ATR or std dev
    maxRiskPerTrade?: string;        // Override default risk %
}

export interface PositionSizeResult {
    method: PositionSizingMethod;
    quantity: string;
    dollarAmount: string;
    riskPercent: string;
    stopLossPrice?: string;
    takeProfitPrice?: string;
    reasoning: string;
}

// ==========================================
// EXPOSURE LIMITS
// ==========================================

export interface ExposureLimits {
    maxPositionPercent: number;      // Max % per single position (e.g., 5%)
    maxSectorPercent: number;        // Max % per sector (e.g., 25%)
    maxTotalExposure: number;        // Max portfolio exposure (e.g., 100%)
    maxCorrelatedPercent: number;    // Max correlated assets (e.g., 30%)
    maxDailyTrades: number;          // Rate limit on trades
}

export interface ExposureCheck {
    symbol: string;
    proposedAmount: string;
    side: 'BUY' | 'SELL';
}

export interface ExposureResult {
    allowed: boolean;
    currentExposure: string;         // Current exposure %
    proposedExposure: string;        // After proposed trade
    limitBreached?: string;          // Which limit would be breached
    reasoning: string;
}

// ==========================================
// STOP-LOSS / TAKE-PROFIT
// ==========================================

export enum StopLossType {
    FIXED_PERCENT = 'FIXED_PERCENT',     // e.g., 2% below entry
    ATR_BASED = 'ATR_BASED',             // e.g., 2x ATR below
    SUPPORT_LEVEL = 'SUPPORT_LEVEL',     // Technical support
    TRAILING = 'TRAILING',               // Trailing stop
}

export interface StopLossConfig {
    type: StopLossType;
    percent?: number;                // For FIXED_PERCENT
    atrMultiplier?: number;          // For ATR_BASED
    trailingPercent?: number;        // For TRAILING
}

export interface TakeProfitConfig {
    riskRewardRatio: number;         // e.g., 2:1 = TP at 2x SL distance
    partialTakeProfit?: {
        percent: number;             // % of position to close
        atRatio: number;             // At what R:R to take partial
    }[];
}

export interface RiskLevels {
    entryPrice: string;
    stopLoss: string;
    takeProfit: string;
    riskAmount: string;              // $ at risk
    rewardAmount: string;            // $ potential reward
    riskRewardRatio: string;
}

// ==========================================
// RISK PROFILE
// ==========================================

export enum RiskTolerance {
    CONSERVATIVE = 'CONSERVATIVE',   // 0.5% per trade
    MODERATE = 'MODERATE',           // 1% per trade
    AGGRESSIVE = 'AGGRESSIVE',       // 2% per trade
    VERY_AGGRESSIVE = 'VERY_AGGRESSIVE', // 5% per trade
}

export interface RiskProfile {
    tolerance: RiskTolerance;
    maxRiskPerTrade: number;         // % of portfolio
    maxDailyLoss: number;            // % of portfolio
    maxDrawdown: number;             // % of portfolio
    sizingMethod: PositionSizingMethod;
    stopLossConfig: StopLossConfig;
    takeProfitConfig: TakeProfitConfig;
    exposureLimits: ExposureLimits;
}

// ==========================================
// RISK METRICS
// ==========================================

export interface RiskMetrics {
    portfolioValue: string;
    totalExposure: string;           // Sum of all positions
    exposurePercent: string;
    unrealizedPnL: string;
    dailyPnL: string;
    dailyPnLPercent: string;
    maxDrawdown: string;
    currentDrawdown: string;
    sharpeRatio?: string;
    var95?: string;                  // Value at Risk (95%)
    positionCount: number;
    largestPosition: {
        symbol: string;
        percent: string;
    } | null;
}

// ==========================================
// DEFAULT PROFILES
// ==========================================

export const DEFAULT_RISK_PROFILES: Record<RiskTolerance, RiskProfile> = {
    [RiskTolerance.CONSERVATIVE]: {
        tolerance: RiskTolerance.CONSERVATIVE,
        maxRiskPerTrade: 0.5,
        maxDailyLoss: 2,
        maxDrawdown: 10,
        sizingMethod: PositionSizingMethod.FIXED_FRACTIONAL,
        stopLossConfig: { type: StopLossType.FIXED_PERCENT, percent: 2 },
        takeProfitConfig: { riskRewardRatio: 2 },
        exposureLimits: {
            maxPositionPercent: 5,
            maxSectorPercent: 15,
            maxTotalExposure: 50,
            maxCorrelatedPercent: 20,
            maxDailyTrades: 5,
        },
    },
    [RiskTolerance.MODERATE]: {
        tolerance: RiskTolerance.MODERATE,
        maxRiskPerTrade: 1,
        maxDailyLoss: 3,
        maxDrawdown: 15,
        sizingMethod: PositionSizingMethod.FIXED_FRACTIONAL,
        stopLossConfig: { type: StopLossType.FIXED_PERCENT, percent: 3 },
        takeProfitConfig: { riskRewardRatio: 2 },
        exposureLimits: {
            maxPositionPercent: 10,
            maxSectorPercent: 25,
            maxTotalExposure: 80,
            maxCorrelatedPercent: 30,
            maxDailyTrades: 10,
        },
    },
    [RiskTolerance.AGGRESSIVE]: {
        tolerance: RiskTolerance.AGGRESSIVE,
        maxRiskPerTrade: 2,
        maxDailyLoss: 5,
        maxDrawdown: 25,
        sizingMethod: PositionSizingMethod.KELLY_CRITERION,
        stopLossConfig: { type: StopLossType.ATR_BASED, atrMultiplier: 2 },
        takeProfitConfig: { riskRewardRatio: 2.5 },
        exposureLimits: {
            maxPositionPercent: 15,
            maxSectorPercent: 35,
            maxTotalExposure: 100,
            maxCorrelatedPercent: 40,
            maxDailyTrades: 20,
        },
    },
    [RiskTolerance.VERY_AGGRESSIVE]: {
        tolerance: RiskTolerance.VERY_AGGRESSIVE,
        maxRiskPerTrade: 5,
        maxDailyLoss: 10,
        maxDrawdown: 40,
        sizingMethod: PositionSizingMethod.KELLY_CRITERION,
        stopLossConfig: { type: StopLossType.ATR_BASED, atrMultiplier: 3 },
        takeProfitConfig: { riskRewardRatio: 3 },
        exposureLimits: {
            maxPositionPercent: 25,
            maxSectorPercent: 50,
            maxTotalExposure: 150, // Leveraged
            maxCorrelatedPercent: 50,
            maxDailyTrades: 50,
        },
    },
};
