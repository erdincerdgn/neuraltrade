import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom, catchError, timeout } from 'rxjs';
import { AxiosError } from 'axios';
import {
    GrpcClientService,
    GrpcConnectionError,
    SignalResponse,
    ModelSelectionResponse,
    ListModelsResponse,
    ModelMetricsResponse,
    StrategyResponse,
    ListStrategiesResponse,
    BacktestResponse,
    VolSurfaceResponse,
    ImpliedVolResponse,
    LocalVolResponse,
    SkewMetricsResponse,
    ArbitrageCheckResponse,
    GreeksResponse,
    OptionPricingResponse,
    OptionsChainResponse,
    RiskMetricsResponse,
    VaRResponse,
    CVaRResponse,
    StressTestResponse,
    PortfolioOptResponse,
    HRPResponse,
    BlackLittermanResponse,
    RegimeDetectionResponse,
    SentimentResponse,
    MicrostructureResponse,
    HealthResponse,
    MetricsResponse,
    SignalAction,
} from './grpc-client.service';
import {
    SignalRequestDto,
    ModelSelectionRequestDto,
    ListModelsRequestDto,
    ModelMetricsRequestDto,
    StrategyRequestDto,
    ListStrategiesRequestDto,
    BacktestRequestDto,
    VolSurfaceRequestDto,
    ImpliedVolRequestDto,
    LocalVolRequestDto,
    SkewMetricsRequestDto,
    ArbitrageCheckRequestDto,
    GreeksRequestDto,
    OptionPricingRequestDto,
    OptionsChainRequestDto,
    RiskMetricsRequestDto,
    VaRRequestDto,
    CVaRRequestDto,
    StressTestRequestDto,
    PortfolioOptRequestDto,
    HRPRequestDto,
    BlackLittermanRequestDto,
    RegimeDetectionRequestDto,
    SentimentRequestDto,
    MicrostructureRequestDto,
    MetricsRequestDto,
    GenerateSignalDto,
    BatchSignalsDto,
    PortfolioOptimizeDto,
    RiskAssessmentDto,
    RAGQueryDto,
    PipelineRunDto,
    RiskProfile,
} from './dto';

export interface AISignalResponse {
    symbol: string;
    action: SignalAction | string;
    confidence: number;
    targetPrice?: number;
    stopLoss?: number;
    takeProfit?: number;
    reasoning: string;
    modelUsed?: string;
    regime?: string;
    volatility?: number;
    riskRewardRatio?: number;
    expectedReturn?: number;
    maxDrawdownEstimate?: number;
    contributors?: SignalContributor[];
    timestamp: string;
    metadata?: Record<string, any>;
}

export interface SignalContributor {
    name: string;
    weight: number;
    signalValue: number;
}

@Injectable()
export class AIProxyService implements OnModuleInit {
    private readonly logger = new Logger(AIProxyService.name);
    private readonly httpBaseUrl: string;
    private httpConnected = false;

    constructor(
        private readonly httpService: HttpService,
        private readonly configService: ConfigService,private readonly grpcClient: GrpcClientService,
    ) {
        this.httpBaseUrl = this.configService.get<string>('AI_BOT_URL') || 'http://localhost:8000';
    }

    async onModuleInit(): Promise<void> {
        await this.checkHealth();
    }

    async checkHealth(): Promise<BotHealthStatus> {
        const grpcStatus = this.grpcClient.getConnectionStatus();
        let grpcHealth: Partial<HealthResponse> = { healthy: false, version: 'N/A', uptimeSeconds: 0 };
        let httpHealth = { connected: false, latency: 0 };

        if (grpcStatus.connected) {
            try {
                grpcHealth = await this.grpcClient.healthCheck();
                this.logger.log('✅ gRPC PRIMARY: Connected');
            } catch (e) {
                this.logger.warn('⚠️ gRPC health check failed');
            }
        }

        const start = Date.now();
        try {
            await firstValueFrom(
                this.httpService.get(`${this.httpBaseUrl}/health`).pipe(timeout(5000)),
            );
            this.httpConnected = true;
            httpHealth = { connected: true, latency: Date.now() - start };
            this.logger.log('✅ HTTP FALLBACK: Available');
        } catch (error) {
            this.httpConnected = false;
            this.logger.warn('⚠️ HTTP fallback not available');
        }

        return {
            grpc: {
                connected: grpcStatus.connected,
                healthy: grpcHealth.healthy || false,
                version: grpcHealth.version || 'N/A',
                uptime: grpcHealth.uptimeSeconds || 0,
                services: grpcHealth.services || {},resources: grpcHealth.resources,},
            http: httpHealth,
            primaryProtocol: grpcStatus.connected ? 'gRPC' : 'HTTP',overall: grpcStatus.connected || this.httpConnected,
        };
    }

    async getServiceMetrics(request: MetricsRequestDto = {}): Promise<MetricsResponse | null> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getServiceMetrics(request);
            } catch (error) {
                this.logger.warn(`gRPC metrics failed: ${error.message}`);}
        }
        return null;
    }

    async predictSignal(request: SignalRequestDto): Promise<SignalResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const response = await this.grpcClient.predictSignal(request);
                this.logger.log(`[gRPC] Signal ${request.symbol}: Action ${response.action} (${(response.confidence * 100).toFixed(1)}%)`);
                return response;
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
                this.logger.warn(`gRPC signal failed, falling back to HTTP: ${error.message}`);
            }
        }

        return this.httpPost<SignalResponse>('/api/v2/signals/predict', {
            symbol: request.symbol,
            timeframe: request.timeframe || '1h',
            prices: request.prices,
            volumes: request.volumes,
            high_prices: request.highPrices,
            low_prices: request.lowPrices,
            open_prices: request.openPrices,
            indicators: request.indicators,
            model_id: request.modelId,
            current_regime: request.currentRegime,
            current_volatility: request.currentVolatility,
        },30000);
    }

    async generateSignal(dto: GenerateSignalDto): Promise<SignalResponse> {
        return this.predictSignal({
            symbol: dto.symbol,
            timeframe: dto.timeframe,
            modelId: dto.modelId,});
    }

    async batchGenerateSignals(dto: BatchSignalsDto): Promise<SignalResponse[]> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                const promises = dto.symbols.map(symbol =>
                    this.predictSignal({ symbol, timeframe: dto.timeframe })
                );
                const results = await Promise.allSettled(promises);
                return results
                    .filter((r): r is PromiseFulfilledResult<SignalResponse> => r.status === 'fulfilled')
                    .map(r => r.value);
            } catch (error) {
                this.logger.warn(`Batch gRPC failed: ${error.message}`);
            }
        }

        return this.httpPost<SignalResponse[]>('/api/v2/signals/batch', dto, 120000);
    }

    async selectModel(request: ModelSelectionRequestDto): Promise<ModelSelectionResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.selectModel(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
                this.logger.warn('Model selection falling back to HTTP');
            }
        }
        return this.httpPost<ModelSelectionResponse>('/api/v2/models/select', request, 10000);
    }

    async listModels(request: ListModelsRequestDto = {}): Promise<ListModelsResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.listModels(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;}
        }
        return this.httpGet<ListModelsResponse>('/api/v2/models', request, 10000);
    }

    async getModelMetrics(request: ModelMetricsRequestDto): Promise<ModelMetricsResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getModelMetrics(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpGet<ModelMetricsResponse>(`/api/v2/models/${request.modelId}/metrics`, request, 15000);
    }

    async routeStrategy(request: StrategyRequestDto): Promise<StrategyResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.routeStrategy(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;this.logger.warn('Strategy routing falling back to HTTP');
            }
        }
        return this.httpPost<StrategyResponse>('/api/v2/strategies/route', request, 10000);
    }

    async listStrategies(request: ListStrategiesRequestDto = {}): Promise<ListStrategiesResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.listStrategies(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpGet<ListStrategiesResponse>('/api/v2/strategies', request, 10000);
    }

    async backtestStrategy(request: BacktestRequestDto): Promise<BacktestResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.backtestStrategy(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<BacktestResponse>('/api/v2/strategies/backtest', request, 120000);
    }

    async calibrateVolatilitySurface(request: VolSurfaceRequestDto): Promise<VolSurfaceResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calibrateVolatilitySurface(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<VolSurfaceResponse>('/api/v2/volatility/surface/calibrate', request, 60000);
    }

    async getImpliedVolatility(request: ImpliedVolRequestDto): Promise<ImpliedVolResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getImpliedVolatility(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<ImpliedVolResponse>('/api/v2/volatility/implied', request, 10000);
    }

    async getLocalVolatility(request: LocalVolRequestDto): Promise<LocalVolResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getLocalVolatility(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<LocalVolResponse>('/api/v2/volatility/local', request, 30000);
    }

    async getSkewMetrics(request: SkewMetricsRequestDto): Promise<SkewMetricsResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getSkewMetrics(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<SkewMetricsResponse>('/api/v2/volatility/skew', request, 10000);
    }

    async checkSurfaceArbitrage(request: ArbitrageCheckRequestDto): Promise<ArbitrageCheckResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.checkSurfaceArbitrage(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<ArbitrageCheckResponse>('/api/v2/volatility/arbitrage', request, 15000);
    }

    async calculateGreeks(request: GreeksRequestDto): Promise<GreeksResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calculateGreeks(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<GreeksResponse>('/api/v2/options/greeks', request, 10000);
    }

    async priceOption(request: OptionPricingRequestDto): Promise<OptionPricingResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.priceOption(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<OptionPricingResponse>('/api/v2/options/price', request, 10000);
    }

    async getOptionsChain(request: OptionsChainRequestDto): Promise<OptionsChainResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getOptionsChain(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<OptionsChainResponse>('/api/v2/options/chain', request, 30000);
    }

    async calculateRiskMetrics(request: RiskMetricsRequestDto): Promise<RiskMetricsResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calculateRiskMetrics(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<RiskMetricsResponse>('/api/v2/risk/metrics', request, 15000);
    }

    async calculateVaR(request: VaRRequestDto): Promise<VaRResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calculateVaR(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<VaRResponse>('/api/v2/risk/var', request, 15000);
    }

    async calculateCVaR(request: CVaRRequestDto): Promise<CVaRResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calculateCVaR(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<CVaRResponse>('/api/v2/risk/cvar', request, 15000);
    }

    async stressTest(request: StressTestRequestDto): Promise<StressTestResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.stressTest(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<StressTestResponse>('/api/v2/risk/stress-test', request, 30000);
    }

    async assessRisk(dto: RiskAssessmentDto): Promise<RiskMetricsResponse> {
        return this.httpPost<RiskMetricsResponse>('/api/v2/risk/assess', dto, 15000);
    }

    async optimizePortfolio(request: PortfolioOptRequestDto): Promise<PortfolioOptResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.optimizePortfolio(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<PortfolioOptResponse>('/api/v2/portfolio/optimize', request, 60000);
    }

    async calculateHRP(request: HRPRequestDto): Promise<HRPResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.calculateHRP(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<HRPResponse>('/api/v2/portfolio/hrp', request, 60000);
    }

    async blackLitterman(request: BlackLittermanRequestDto): Promise<BlackLittermanResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.blackLitterman(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<BlackLittermanResponse>('/api/v2/portfolio/black-litterman', request, 60000);
    }

    async optimizePortfolioLegacy(dto: PortfolioOptimizeDto): Promise<PortfolioOptResponse> {
        const symbols = dto.holdings.map(h => h.symbol);
        return this.optimizePortfolio({
            symbols,
            method: dto.useHRP ? 3 : dto.useBlackLitterman ? 2 : 1,
            targetReturn: dto.targetReturn,
        });
    }

    async detectRegime(request: RegimeDetectionRequestDto): Promise<RegimeDetectionResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.detectRegime(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<RegimeDetectionResponse>('/api/v2/market/regime', request, 15000);
    }

    async analyzeSentiment(request: SentimentRequestDto): Promise<SentimentResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.analyzeSentiment(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<SentimentResponse>('/api/v2/market/sentiment', request, 20000);
    }

    async getMarketMicrostructure(request: MicrostructureRequestDto): Promise<MicrostructureResponse> {
        if (this.grpcClient.isGrpcConnected()) {
            try {
                return await this.grpcClient.getMarketMicrostructure(request);
            } catch (error) {
                if (!(error instanceof GrpcConnectionError)) throw error;
            }
        }
        return this.httpPost<MicrostructureResponse>('/api/v2/market/microstructure', request, 15000);
    }

    async queryRAG(dto: RAGQueryDto): Promise<RAGQueryResponse> {
        return this.httpPost<RAGQueryResponse>('/api/v2/rag/query', {
            query: dto.query,
            context: dto.context,
            symbols: dto.symbols,
            max_sources: dto.maxSources || 5,
            similarity_threshold: dto.similarityThreshold || 0.7,
        }, 20000);
    }

    async runPipeline(dto: PipelineRunDto): Promise<PipelineResponse> {
        const startTime = Date.now();

        try {
            const [signal, regime] = await Promise.all([
                this.predictSignal({ symbol: dto.symbol, timeframe: dto.timeframe }),
                dto.useRegime !== false ? this.detectRegime({ symbol: dto.symbol }).catch(() => null) : null,
            ]);

            const model = regime
                ? await this.selectModel({
                    symbol: dto.symbol,
                    regime: regime.currentRegime,
                    volatility: 0,
                }).catch(() => null)
                : null;

            const strategy = await this.routeStrategy({
                symbol: dto.symbol,
                signalAction: signal.action,
                confidence: signal.confidence,
                userRiskProfile: dto.riskProfile || RiskProfile.MODERATE,
            }).catch(() => null);

            const sentiment = dto.useSentiment
                ? await this.analyzeSentiment({ symbol: dto.symbol }).catch(() => null)
                : null;

            this.logger.log(`Pipeline completed for ${dto.symbol} in ${Date.now() - startTime}ms`);

            return {
                symbol: dto.symbol,
                success: true,
                signal,
                regime,
                model,
                strategy,
                sentiment,
                processingTime: Date.now() - startTime,
                timestamp: new Date().toISOString(),
            };
        } catch (error) {
            this.logger.error(`Pipeline failed for ${dto.symbol}: ${error.message}`);
            return {
                symbol: dto.symbol,
                success: false,
                error: error.message,
                processingTime: Date.now() - startTime,
                timestamp: new Date().toISOString(),
            };
        }
    }

    private async httpPost<T>(path: string, data: any, timeoutMs: number): Promise<T> {
        try {
            const response = await firstValueFrom(
                this.httpService.post(`${this.httpBaseUrl}${path}`, data).pipe(
                    timeout(timeoutMs),
                    catchError((error:AxiosError) => {
                        throw new Error(`HTTP POST ${path} failed: ${error.message}`);
                    }),
                ),
            );
            this.logger.debug(`[HTTP] ${path} completed`);
            return response.data;
        } catch (error) {
            this.logger.error(`HTTP POST ${path} failed: ${error.message}`);
            throw error;
        }
    }

    private async httpGet<T>(path: string, params: any, timeoutMs: number): Promise<T> {
        try {
            const response = await firstValueFrom(
                this.httpService.get(`${this.httpBaseUrl}${path}`, { params }).pipe(
                    timeout(timeoutMs),
                    catchError((error: AxiosError) => {
                        throw new Error(`HTTP GET ${path} failed: ${error.message}`);
                    }),
                ),
            );
            return response.data;
        } catch (error) {
            this.logger.error(`HTTP GET ${path} failed: ${error.message}`);
            throw error;
        }
    }

    isBotConnected(): boolean {
        return this.grpcClient.isGrpcConnected() || this.httpConnected;
    }

    isGrpcConnected(): boolean {
        return this.grpcClient.isGrpcConnected();
    }

    getProtocol(): string {
        return this.grpcClient.isGrpcConnected() ? 'gRPC' : 'HTTP';
    }
}

export interface BotHealthStatus {
    grpc: {
        connected: boolean;
        healthy: boolean;
        version: string;
        uptime: number;
        services: Record<string, boolean>;
        resources?: {
            cpuUsage: number;
            memoryUsage: number;
            gpuUsage: number;
            activeConnections: number;
        };
    };
    http: {
        connected: boolean;
        latency: number;
    };primaryProtocol: string;
    overall: boolean;
}

export interface RAGQueryResponse {
    answer: string;
    sources: Array<{ title: string; url?: string; relevance: number; snippet: string }>;
    confidence: number;
    relatedQueries: string[];
}

export interface PipelineResponse {
    symbol: string;
    success: boolean;
    signal?: SignalResponse;
    regime?: RegimeDetectionResponse | null;
    model?: ModelSelectionResponse | null;
    strategy?: StrategyResponse | null;
    sentiment?: SentimentResponse | null;
    error?: string;
    processingTime: number;
    timestamp: string;
}