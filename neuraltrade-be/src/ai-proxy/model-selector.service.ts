import { Injectable, Logger } from '@nestjs/common';
import { RedisService } from '../core/redis/redis.service';
import {
    GrpcClientService,
    GrpcConnectionError,
    ModelType,
    MarketRegime,
    ModelInfo as GrpcModelInfo,
    ModelPerformance,
} from './grpc-client.service';

/**
 * Model Information
 */
export interface ModelInfo {
    id: string;
    name: string;
    type: ModelType;
    version: string;
    accuracy: number;
    regimes: MarketRegime[];
    isActive: boolean;
    lastTrained?: Date;
    performance?: ModelPerformance;
}

/**
 * Model Selection Result
 */
export interface ModelSelectionResult {
    model: ModelInfo;
    confidence: number;
    reasoning: string;
    fallback: boolean;
    performance?: ModelPerformance;
}

/**
 * Model Selector Service v2.0
 *
 * Dynamically selects the best ML model based on:
 * - Market regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CRISIS)
 * - Volatility level
 * - Liquidity conditions
 * - Historical accuracy & performance metrics
 * - Model availability
 * 
 * Aligned with ai_service.proto v2.0
 * 
 * @version 2.0.0
 * @author Senior Quant Developer
 */
@Injectable()
export class ModelSelectorService {
    private readonly logger = new Logger(ModelSelectorService.name);
    private readonly CACHE_TTL = 300; // 5 minutes

    // Available models (local registry)
    private readonly models: Map<string, ModelInfo> = new Map([
        ['lstm_trend', {
            id: 'lstm_trend',
            name: 'LSTM Trend Predictor',
            type: ModelType.LSTM,
            version: '2.1.0',
            accuracy: 0.72,
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            isActive: true,
        }],
        ['transformer_v1', {
            id: 'transformer_v1',
            name: 'Transformer Signal',
            type: ModelType.TRANSFORMER,
            version: '1.0.0',
            accuracy: 0.75,
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.RANGING, MarketRegime.VOLATILE],
            isActive: true,
        }],
        ['drl_agent', {
            id: 'drl_agent',
            name: 'DRL Trading Agent',
            type: ModelType.DRL,
            version: '3.0.0',
            accuracy: 0.68,
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE],
            isActive: true,
        }],
        ['ensemble_conservative', {
            id: 'ensemble_conservative',
            name: 'Ensemble Conservative',
            type: ModelType.ENSEMBLE,
            version: '1.5.0',
            accuracy: 0.78,
            regimes: [MarketRegime.RANGING, MarketRegime.CRISIS],
            isActive: true,
        }],
        ['xgb_momentum', {
            id: 'xgb_momentum',
            name: 'XGBoost Momentum',
            type: ModelType.XGB,
            version: '2.0.0',
            accuracy: 0.74,
            regimes: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            isActive: true,
        }],
        ['lightgbm_scalper', {
            id: 'lightgbm_scalper',
            name: 'LightGBM Scalper',
            type: ModelType.LIGHTGBM,
            version: '1.0.0',
            accuracy: 0.71,
            regimes: [MarketRegime.RANGING, MarketRegime.VOLATILE],
            isActive: true,
        }],
    ]);

    constructor(
        private readonly redis: RedisService,
        private readonly grpcClient: GrpcClientService,
    ) {}

    /**
     * Select optimal model for current market conditions
     */
    async selectModel(params: {
        symbol: string;
        regime: MarketRegime;
        volatility: number;
        liquidity?: number;
        context?: Record<string, string>;
    }): Promise<ModelSelectionResult> {
        const cacheKey = `model_selection:${params.symbol}:${params.regime}:${Math.round(params.volatility * 100)}`;

        // Check cache
        const cached = await this.redis.get<ModelSelectionResult>(cacheKey);
        if (cached) {
            this.logger.debug(`Cache hit for model selection: ${params.symbol}`);
            return cached;
        }

        // Try gRPC first (PRIMARY)
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const result = await this.grpcClient.selectModel({
                    symbol: params.symbol,
                    regime: params.regime,
                    volatility: params.volatility,
                    liquidity: params.liquidity || 0,
                    context: params.context,
                });

                const model = this.models.get(result.modelId) || this.getDefaultModel();

                const selection: ModelSelectionResult = {
                    model: {
                        ...model,
                        performance: result.performance,
                    },
                    confidence: result.confidence,
                    reasoning: result.reasoning,
                    fallback: false,
                    performance: result.performance,
                };

                await this.redis.set(cacheKey, selection, this.CACHE_TTL);
                this.logger.log(`[gRPC] Model selected: ${result.modelId} for ${params.symbol} (${MarketRegime[params.regime]})`);
                return selection;
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) {
                    throw error;
                }
                this.logger.warn(`gRPC model selection failed, using local: ${error.message}`);
            }
        }

        // Local fallback
        return this.localSelectModel(params, cacheKey);
    }

    /**
     * Local model selection logic (fallback)
     */
    private async localSelectModel(
        params: {
            symbol: string;
            regime: MarketRegime;
            volatility: number;
        },
        cacheKey: string,
    ): Promise<ModelSelectionResult> {
        // Find models matching regime
        const candidates: ModelInfo[] = [];

        for (const model of this.models.values()) {
            if (model.isActive && model.regimes.includes(params.regime)) {
                candidates.push(model);
            }
        }

        if (candidates.length === 0) {
            const defaultModel = this.getDefaultModel();
            return {
                model: defaultModel,
                confidence: 0.5,
                reasoning: `No model matched regime ${MarketRegime[params.regime]}, using default ensemble`,
                fallback: true,
            };
        }

        // Score candidates based on regime and volatility
        const scored = candidates.map(model => {
            let score = model.accuracy;

            // Regime-specific adjustments
            switch (params.regime) {
                case MarketRegime.TRENDING_UP:
                case MarketRegime.TRENDING_DOWN:
                    if (model.type === ModelType.LSTM) score += 0.05;
                    if (model.type === ModelType.DRL) score += 0.03;
                    break;
                case MarketRegime.RANGING:
                    if (model.type === ModelType.ENSEMBLE) score += 0.05;
                    if (model.type === ModelType.LIGHTGBM) score += 0.03;
                    break;
                case MarketRegime.VOLATILE:
                    if (model.type === ModelType.TRANSFORMER) score += 0.05;
                    if (model.type === ModelType.DRL) score += 0.03;
                    break;
                case MarketRegime.CRISIS:
                    if (model.type === ModelType.ENSEMBLE) score += 0.08;
                    break;
            }

            // Volatility adjustments
            if (params.volatility > 0.03) {
                if (model.type === ModelType.TRANSFORMER) score += 0.03;
                if (model.type === ModelType.ENSEMBLE) score += 0.02;
            } else if (params.volatility < 0.01) {
                if (model.type === ModelType.LIGHTGBM) score += 0.03;
                if (model.type === ModelType.XGB) score += 0.02;
            }

            return { model, score };
        });

        // Sort by score descending
        scored.sort((a, b) => b.score - a.score);
        const best = scored[0];

        const result: ModelSelectionResult = {
            model: best.model,
            confidence: Math.min(best.score, 0.95),
            reasoning: `Selected ${best.model.name} for ${MarketRegime[params.regime]} regime` +
                `(accuracy: ${(best.model.accuracy * 100).toFixed(0)}%, volatility: ${(params.volatility * 100).toFixed(2)}%)`,
            fallback: true,
        };

        // Cache result
        await this.redis.set(cacheKey, result, this.CACHE_TTL);
        this.logger.log(`[Local] Model selected: ${best.model.id} for ${params.symbol}`);

        return result;
    }

    /**
     * Get all available models from AI Engine
     */
    async listModels(activeOnly: boolean = true): Promise<ModelInfo[]> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const response = await this.grpcClient.listModels({ activeOnly });
                return response.models.map(m => this.mapGrpcModelToLocal(m));
            } catch (error) {
                this.logger.warn(`Failed to list models from gRPC: ${error.message}`);
            }
        }

        // Return local models
        return Array.from(this.models.values()).filter(m => !activeOnly || m.isActive);
    }

    /**
     * Get model metrics from AI Engine
     */
    async getModelMetrics(modelId: string, startTime?: number, endTime?: number): Promise<ModelPerformance | null> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const response = await this.grpcClient.getModelMetrics({
                    modelId,
                    startTime,
                    endTime,
                });
                return response.performance;
            } catch (error) {
                this.logger.warn(`Failed to get model metrics: ${error.message}`);
            }
        }
        return null;
    }

    /**
     * Get default model (ensemble for safety)
     */
    private getDefaultModel(): ModelInfo {
        return this.models.get('ensemble_conservative')!;
    }

    /**
     * Map gRPC model info to local format
     */
    private mapGrpcModelToLocal(grpcModel: GrpcModelInfo): ModelInfo {
        return {
            id: grpcModel.id,
            name: grpcModel.name,
            type: grpcModel.type,
            version: grpcModel.version,
            accuracy: grpcModel.accuracy,
            regimes: grpcModel.supportedSymbols as unknown as MarketRegime[], // Map appropriately
            isActive: grpcModel.isActive,
            lastTrained: grpcModel.lastTrained ? new Date(grpcModel.lastTrained) : undefined,
            performance: grpcModel.performance,
        };
    }

    /**
     * Get model by ID
     */
    getModel(id: string): ModelInfo | undefined {
        return this.models.get(id);
    }

    /**
     * Check if a model is available and active
     */
    isModelActive(id: string): boolean {
        const model = this.models.get(id);
        return model?.isActive ?? false;
    }

    /**
     * Get models suitable for a specific regime
     */
    getModelsForRegime(regime: MarketRegime): ModelInfo[] {
        return Array.from(this.models.values()).filter(
            m => m.isActive && m.regimes.includes(regime)
        );
    }
}