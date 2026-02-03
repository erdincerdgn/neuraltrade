import { Injectable, Logger } from '@nestjs/common';
import { RedisService } from '../core/redis/redis.service';
import {
    GrpcClientService,
    GrpcConnectionError,
    SignalAction,
    MarketRegime,
    RiskProfile,
    ExecutionType,
    RiskLimits,
    StrategyPerformance,
} from './grpc-client.service';

/**
 * Strategy Definition
 */
export interface Strategy {
    id: string;
    name: string;
    description: string;
    regimes: MarketRegime[];
    riskProfiles: RiskProfile[];
    isActive: boolean;
    performance?: StrategyPerformance;
}

/**
 * Strategy Routing Result
 */
export interface StrategyRoutingResult {
    strategy: Strategy;
    executionType: ExecutionType;
    positionSize: number;
    positionSizeModifier: number;
    riskLimits: RiskLimits;
    reasoning: string;
    fallback: boolean;
}

/**
 * Strategy Router Service v2.0
 */
@Injectable()
export class StrategyRouterService {
    private readonly logger = new Logger(StrategyRouterService.name);
    private readonly CACHE_TTL = 60;

    private readonly strategies: Map<string, Strategy> = new Map([
        ['trend_following', {
            id: 'trend_following',
            name: 'Trend Following',
            description: 'Follow strong directional moves with momentum confirmation',
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            riskProfiles: [RiskProfile.MODERATE, RiskProfile.AGGRESSIVE],
            isActive: true,}],
        ['mean_reversion', {
            id: 'mean_reversion',
            name: 'Mean Reversion',
            description: 'Trade reversions to mean in ranging markets with Bollinger Bands',
            regimes: [MarketRegime.RANGING],
            riskProfiles: [RiskProfile.CONSERVATIVE, RiskProfile.MODERATE],
            isActive: true,
        }],
        ['momentum_breakout', {
            id: 'momentum_breakout',
            name: 'Momentum Breakout',
            description: 'Capture momentum in strong breakout moves',
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE],
            riskProfiles: [RiskProfile.AGGRESSIVE],
            isActive: true,
        }],
        ['volatility_scalping', {
            id: 'volatility_scalping',
            name: 'Volatility Scalping',
            description: 'Quick trades exploiting volatility in ranging markets',
            regimes: [MarketRegime.RANGING, MarketRegime.VOLATILE],
            riskProfiles: [RiskProfile.MODERATE, RiskProfile.AGGRESSIVE],
            isActive: true,
        }],
        ['defensive_hedge', {
            id: 'defensive_hedge',
            name: 'Defensive Hedge',
            description: 'Capital preservation with hedging in uncertain/crisis markets',
            regimes: [MarketRegime.VOLATILE, MarketRegime.CRISIS],
            riskProfiles: [RiskProfile.CONSERVATIVE],
            isActive: true,
        }],
        ['statistical_arbitrage', {
            id: 'statistical_arbitrage',
            name: 'Statistical Arbitrage',
            description: 'Pairs trading and statistical mean reversion',
            regimes: [MarketRegime.RANGING],
            riskProfiles: [RiskProfile.MODERATE],
            isActive: true,
        }],
    ]);

    constructor(
        private readonly redis: RedisService,
        private readonly grpcClient: GrpcClientService,
    ) {}

    async routeSignal(params: {
        symbol: string;
        action: SignalAction;
        confidence: number;
        regime: MarketRegime;
        volatility: number;
        userRiskProfile: RiskProfile;
        accountBalance?: number;
        currentExposure?: number;
    }): Promise<StrategyRoutingResult> {
        const cacheKey = `strategy_route:${params.symbol}:${params.regime}:${params.action}:${params.userRiskProfile}`;

        const cached = await this.redis.get<StrategyRoutingResult>(cacheKey);
        if (cached && params.confidence >= 0.7) {
            this.logger.debug(`Cache hit for strategy routing: ${params.symbol}`);
            return cached;
        }

        if (this.grpcClient.isGrpcConnected()) {
            try {
                const result = await this.grpcClient.routeStrategy({
                    symbol: params.symbol,
                    signalAction: params.action,
                    confidence: params.confidence,
                    marketContext: {
                        regime: MarketRegime[params.regime],
                        volatility: params.volatility.toString(),
                    },
                    userRiskProfile: params.userRiskProfile,
                    accountBalance: params.accountBalance || 0,
                    currentExposure: params.currentExposure || 0,
                });

                const strategy = this.strategies.get(result.strategyId) || this.getDefaultStrategy();

                const routing: StrategyRoutingResult = {
                    strategy,
                    executionType: result.executionType,
                    positionSize: result.positionSize,
                    positionSizeModifier: result.positionSizeModifier,
                    riskLimits: result.riskLimits,
                    reasoning: result.reasoning,
                    fallback: false,
                };

                await this.redis.set(cacheKey, routing, this.CACHE_TTL);
                this.logger.log(`[gRPC] Strategy routed: ${result.strategyId} for ${params.symbol} (${ExecutionType[result.executionType]})`);
                return routing;
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) {
                    throw error;
                }
                this.logger.warn(`gRPC strategy routing failed, using local: ${error.message}`);
            }
        }

        return this.localRoute(params, cacheKey);
    }

    private async localRoute(
        params: {
            symbol: string;
            action: SignalAction;
            confidence: number;
            regime: MarketRegime;
            volatility: number;
            userRiskProfile: RiskProfile;
            accountBalance?: number;
            currentExposure?: number;
        },
        cacheKey: string,
    ): Promise<StrategyRoutingResult> {
        let selectedStrategy: Strategy | undefined;
        let executionType: ExecutionType = ExecutionType.NEUTRAL;
        let positionSizeModifier = 1.0;

        for (const strategy of this.strategies.values()) {
            if (
                strategy.isActive &&
                strategy.regimes.includes(params.regime) &&
                strategy.riskProfiles.includes(params.userRiskProfile)
            ) {
                selectedStrategy = strategy;
                break;
            }
        }

        if (!selectedStrategy) {
            selectedStrategy = this.getDefaultStrategy();
        }

        if (params.confidence >= 0.85) {
            executionType = ExecutionType.AGGRESSIVE;positionSizeModifier = 1.3;
        } else if (params.confidence >= 0.7) {
            executionType = ExecutionType.NEUTRAL;
            positionSizeModifier = 1.0;
        } else if (params.confidence >= 0.6) {
            executionType = ExecutionType.CONSERVATIVE;
            positionSizeModifier = 0.7;
        } else {
            executionType = ExecutionType.CONSERVATIVE;
            positionSizeModifier = 0.5;
        }

        if (params.volatility > 0.05) {
            positionSizeModifier *= 0.7;
            if (executionType === ExecutionType.AGGRESSIVE) {
                executionType = ExecutionType.NEUTRAL;
            }
        } else if (params.volatility > 0.03) {
            positionSizeModifier *= 0.85;
        }

        if (params.regime === MarketRegime.CRISIS) {
            positionSizeModifier *= 0.5;
            executionType = ExecutionType.CONSERVATIVE;
        }

        const accountBalance = params.accountBalance || 100000;
        const positionValue = accountBalance * positionSizeModifier *0.1;
        if (positionValue > 50000) {
            executionType = ExecutionType.TWAP;
        }

        const riskLimits: RiskLimits = this.calculateRiskLimits(
            params.userRiskProfile,
            params.volatility,
            accountBalance,
        );

        const result: StrategyRoutingResult = {
            strategy: selectedStrategy,
            executionType,
            positionSize: positionValue,
            positionSizeModifier,
            riskLimits,
            reasoning: `Selected ${selectedStrategy.name} for ${MarketRegime[params.regime]} regime` +
                `with ${(params.confidence * 100).toFixed(0)}% confidence. ` +
                `Execution: ${ExecutionType[executionType]}, Size modifier: ${positionSizeModifier.toFixed(2)}x`,
            fallback: true,
        };

        await this.redis.set(cacheKey, result, this.CACHE_TTL);
        this.logger.log(`[Local] Strategy routed: ${selectedStrategy.id} for ${params.symbol}`);

        return result;
    }

    private calculateRiskLimits(
        riskProfile: RiskProfile,
        volatility: number,
        accountBalance: number,
    ): RiskLimits {
        const baseConfig = {
            [RiskProfile.CONSERVATIVE]: {
                maxPositionPct: 0.05,
                maxDailyLossPct: 0.02,
                maxDrawdownPct: 0.05,
                stopLossPct: 0.02,
                takeProfitPct: 0.04,
            },
            [RiskProfile.MODERATE]: {
                maxPositionPct: 0.10,
                maxDailyLossPct: 0.03,
                maxDrawdownPct: 0.10,
                stopLossPct: 0.03,
                takeProfitPct: 0.06,
            },
            [RiskProfile.AGGRESSIVE]: {
                maxPositionPct: 0.20,
                maxDailyLossPct: 0.05,
                maxDrawdownPct: 0.15,
                stopLossPct: 0.05,
                takeProfitPct: 0.10,
            },
        };

        const config = baseConfig[riskProfile] || baseConfig[RiskProfile.MODERATE];
        const volAdjustment = volatility > 0.03 ? 0.8 : 1.0;

        return {
            maxPositionSize: accountBalance * config.maxPositionPct * volAdjustment,
            maxDailyLoss: accountBalance * config.maxDailyLossPct,
            maxDrawdown: config.maxDrawdownPct,
            stopLossPct: config.stopLossPct * (1 + volatility),
            takeProfitPct: config.takeProfitPct * (1 + volatility * 0.5),
        };
    }

    private getDefaultStrategy(): Strategy {
        return this.strategies.get('mean_reversion')!;
    }

    async listStrategies(regimeFilter?: MarketRegime): Promise<Strategy[]> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const response = await this.grpcClient.listStrategies({
                    regimeFilter,
                });
                return response.strategies.map(s => ({
                    id: s.id,
                    name: s.name,
                    description: s.description,
                    regimes: s.supportedRegimes,
                    riskProfiles: [RiskProfile.MODERATE],
                    isActive: s.isActive,
                    performance: s.performance,
                }));
            } catch (error) {
                this.logger.warn(`Failed to list strategies from gRPC: ${error.message}`);
            }
        }

        let strategies = Array.from(this.strategies.values());
        if (regimeFilter) {
            strategies = strategies.filter(s => s.regimes.includes(regimeFilter));
        }
        return strategies;
    }

    async backtestStrategy(params: {
        strategyId: string;
        symbol: string;
        startTime: number;
        endTime: number;
        initialCapital?: number;
        parameters?: Record<string, string>;
    }): Promise<StrategyPerformance | null> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const response = await this.grpcClient.backtestStrategy({
                    strategyId: params.strategyId,
                    symbol: params.symbol,
                    startTime: params.startTime,
                    endTime: params.endTime,
                    initialCapital: params.initialCapital || 100000,
                    parameters: params.parameters || {},
                });
                return response.performance;
            } catch (error) {
                this.logger.error(`Backtest failed: ${error.message}`);
            }
        }
        return null;
    }

    getStrategy(id: string): Strategy | undefined {
        return this.strategies.get(id);
    }

    getStrategiesForRegime(regime: MarketRegime): Strategy[] {
        return Array.from(this.strategies.values()).filter(
            s => s.isActive && s.regimes.includes(regime)
        );
    }
}