import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import * as path from 'path';

/**
 * NeuralTrade gRPC Client Service v2.0
 * 
 * PRIMARY communication channel with Python AI Engine.
 * Aligned with ai_service.proto v2.0
 * 
 * Features:
 * - Signal Prediction & Streaming
 * - Model Management
 * - Strategy Routing & Backtesting
 * - Volatility Surface (SABR, SVI, Dupire)
 * - Options & Greeks Calculation
 * - Risk Management (VaR, CVaR, Stress Testing)
 * - Portfolio Optimization (Black-Litterman, HRP)
 * - Market Analysis & Regime Detection
 * 
 * @version 2.0.0
 * @author Senior Quant Developer
 */
@Injectable()
export class GrpcClientService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(GrpcClientService.name);
    private client: any;
    private isConnected = false;
    private reconnectAttempts = 0;
    private readonly maxReconnectAttempts = 5;
    private reconnectTimer: NodeJS.Timeout | null = null;

    constructor(private readonly configService: ConfigService) {}

    async onModuleInit(): Promise<void> {
        await this.connect();
    }

    async onModuleDestroy(): Promise<void> {
        this.disconnect();
    }

    // =========================================================================
    // CONNECTION MANAGEMENT
    // =========================================================================

    private async connect(): Promise<void> {
        const grpcHost = this.configService.get<string>('AI_GRPC_HOST', 'localhost');
        const grpcPort = this.configService.get<string>('AI_GRPC_PORT', '50051');
        const protoPath = path.join(process.cwd(), 'proto', 'ai_service.proto');

        try {
            const packageDefinition = await protoLoader.load(protoPath, {
                keepCase: true,
                longs: String,
                enums: String,
                defaults: true,
                oneofs: true,
            });

            const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);
            const aiPackage = (protoDescriptor as any).neuraltrade.ai.v2;

            this.client = new aiPackage.AIService(
                `${grpcHost}:${grpcPort}`,
                grpc.credentials.createInsecure(),{
                    'grpc.keepalive_time_ms': 10000,
                    'grpc.keepalive_timeout_ms': 5000,
                    'grpc.keepalive_permit_without_calls': 1,
                    'grpc.max_receive_message_length': 50 * 1024 * 1024,
                    'grpc.max_send_message_length': 50 * 1024 * 1024,
                },
            );

            await this.healthCheck();
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.logger.log(`✅ gRPC PRIMARY connected to AI Engine v2.0 at ${grpcHost}:${grpcPort}`);
        } catch (error) {
            this.logger.warn(`⚠️ gRPC connection failed: ${error.message} - Will use HTTP fallback`);
            this.isConnected = false;
            this.scheduleReconnect();
        }
    }

    private disconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.client) {
            grpc.closeClient(this.client);
            this.client = null;
        }
        this.isConnected = false;
        this.logger.log('🔌 gRPC client disconnected');
    }

    private scheduleReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.logger.error('❌ Max gRPC reconnection attempts reached - Using HTTP fallback');
            return;
        }

        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        this.reconnectAttempts++;

        this.logger.log(`🔄 gRPC reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
    }

    // =========================================================================
    // SIGNAL & PREDICTION
    // =========================================================================

    async predictSignal(request: SignalRequest): Promise<SignalResponse> {
        return this.callGrpc<SignalResponse>('PredictSignal', {
            symbol: request.symbol,
            timeframe: request.timeframe || '1h',
            prices: request.prices || [],
            volumes: request.volumes || [],
            high_prices: request.highPrices || [],
            low_prices: request.lowPrices || [],
            open_prices: request.openPrices || [],
            indicators: request.indicators || {},model_id: request.modelId || '',
            current_regime: request.currentRegime || 0,
            current_volatility: request.currentVolatility || 0,
        },30);
    }

    streamSignals(request: SignalStreamRequest): grpc.ClientReadableStream<SignalResponse> {
        this.ensureConnected();
        return this.client.StreamSignals({
            symbols: request.symbols,
            timeframe: request.timeframe || '1h',
            model_id: request.modelId || '',
            update_interval_ms: request.updateIntervalMs || 1000,
        });
    }

    // =========================================================================
    // MODEL MANAGEMENT
    // =========================================================================

    async selectModel(request: ModelSelectionRequest): Promise<ModelSelectionResponse> {
        return this.callGrpc<ModelSelectionResponse>('SelectModel', {
            symbol: request.symbol,
            regime: request.regime || 0,
            volatility: request.volatility || 0,
            liquidity: request.liquidity || 0,
            context: request.context || {},
        }, 10);
    }

    async listModels(request: ListModelsRequest = {}): Promise<ListModelsResponse> {
        return this.callGrpc<ListModelsResponse>('ListModels', {
            filter: request.filter || '',
            type_filter: request.typeFilter || 0,
            active_only: request.activeOnly ?? true,
        }, 10);
    }

    async getModelMetrics(request: ModelMetricsRequest): Promise<ModelMetricsResponse> {
        return this.callGrpc<ModelMetricsResponse>('GetModelMetrics', {
            model_id: request.modelId,
            start_time: request.startTime || 0,
            end_time: request.endTime || 0,
        }, 15);
    }

    // =========================================================================
    // STRATEGY ROUTING
    // =========================================================================

    async routeStrategy(request: StrategyRequest): Promise<StrategyResponse> {
        return this.callGrpc<StrategyResponse>('RouteStrategy', {
            symbol: request.symbol,
            signal_action: request.signalAction || 0,
            confidence: request.confidence || 0,
            market_context: request.marketContext || {},
            user_risk_profile: request.userRiskProfile || 2, // MODERATE
            account_balance: request.accountBalance || 0,
            current_exposure: request.currentExposure || 0,
        }, 10);
    }

    async listStrategies(request: ListStrategiesRequest = {}): Promise<ListStrategiesResponse> {
        return this.callGrpc<ListStrategiesResponse>('ListStrategies', {
            filter: request.filter || '',
            regime_filter: request.regimeFilter || 0,
        }, 10);
    }

    async backtestStrategy(request: BacktestRequest): Promise<BacktestResponse> {
        return this.callGrpc<BacktestResponse>('BacktestStrategy', {
            strategy_id: request.strategyId,
            symbol: request.symbol,
            start_time: request.startTime,
            end_time: request.endTime,
            initial_capital: request.initialCapital || 100000,
            parameters: request.parameters || {},
        }, 120);
    }

    // =========================================================================
    // VOLATILITY SURFACE (NEW)
    // =========================================================================

    async calibrateVolatilitySurface(request:VolSurfaceRequest): Promise<VolSurfaceResponse> {
        return this.callGrpc<VolSurfaceResponse>('CalibrateVolatilitySurface', {
            symbol: request.symbol,
            spot_price: request.spotPrice,
            risk_free_rate: request.riskFreeRate,
            dividend_yield: request.dividendYield || 0,
            model: request.model || 1, // SABR
            market_data: request.marketData || [],beta: request.beta || 0.5,
        }, 60);
    }

    async getImpliedVolatility(request: ImpliedVolRequest): Promise<ImpliedVolResponse> {
        return this.callGrpc<ImpliedVolResponse>('GetImpliedVolatility', {
            symbol: request.symbol,
            strike: request.strike,
            expiry: request.expiry,
            spot_price: request.spotPrice,}, 10);
    }

    async getLocalVolatility(request: LocalVolRequest): Promise<LocalVolResponse> {
        return this.callGrpc<LocalVolResponse>('GetLocalVolatility', {
            symbol: request.symbol,
            strike_min: request.strikeMin,
            strike_max: request.strikeMax,
            expiry_min: request.expiryMin,
            expiry_max: request.expiryMax,
            num_strikes: request.numStrikes || 20,
            num_expiries: request.numExpiries || 10,
        }, 30);
    }

    async getSkewMetrics(request: SkewMetricsRequest): Promise<SkewMetricsResponse> {
        return this.callGrpc<SkewMetricsResponse>('GetSkewMetrics', {
            symbol: request.symbol,
            expiry: request.expiry,}, 10);
    }

    async checkSurfaceArbitrage(request: ArbitrageCheckRequest): Promise<ArbitrageCheckResponse> {
        return this.callGrpc<ArbitrageCheckResponse>('CheckSurfaceArbitrage', {
            symbol: request.symbol,}, 15);
    }

    // =========================================================================
    // OPTIONS & GREEKS (NEW)
    // =========================================================================

    async calculateGreeks(request: GreeksRequest): Promise<GreeksResponse> {
        return this.callGrpc<GreeksResponse>('CalculateGreeks', {
            symbol: request.symbol,
            option_type: request.optionType || 1, // CALL
            spot_price: request.spotPrice,
            strike: request.strike,
            expiry: request.expiry,
            risk_free_rate: request.riskFreeRate,
            dividend_yield: request.dividendYield || 0,
            volatility: request.volatility,
            pricing_model: request.pricingModel || 'black_scholes',
        }, 10);
    }

    async priceOption(request: OptionPricingRequest): Promise<OptionPricingResponse> {
        return this.callGrpc<OptionPricingResponse>('PriceOption', {
            symbol: request.symbol,
            option_type: request.optionType || 1,
            spot_price: request.spotPrice,
            strike: request.strike,
            expiry: request.expiry,
            risk_free_rate: request.riskFreeRate,
            dividend_yield: request.dividendYield || 0,
            volatility: request.volatility,
            pricing_model: request.pricingModel || 'black_scholes',
            american_style: request.americanStyle || false,
        }, 10);
    }

    async getOptionsChain(request: OptionsChainRequest): Promise<OptionsChainResponse> {
        return this.callGrpc<OptionsChainResponse>('GetOptionsChain', {
            symbol: request.symbol,
            spot_price: request.spotPrice,
            expiries: request.expiries || [],
            num_strikes: request.numStrikes || 10,
            strike_range_pct: request.strikeRangePct || 0.2,
        }, 30);
    }

    // =========================================================================
    // RISK MANAGEMENT (NEW)
    // =========================================================================

    async calculateRiskMetrics(request: RiskMetricsRequest): Promise<RiskMetricsResponse> {
        return this.callGrpc<RiskMetricsResponse>('CalculateRiskMetrics', {
            returns: request.returns,
            confidence_level: request.confidenceLevel || 0.95,
            time_horizon: request.timeHorizon || 1,
            risk_free_rate: request.riskFreeRate || 0,}, 15);
    }

    async calculateVaR(request: VaRRequest): Promise<VaRResponse> {
        return this.callGrpc<VaRResponse>('CalculateVaR', {
            returns: request.returns,
            confidence_level: request.confidenceLevel || 0.95,
            time_horizon: request.timeHorizon || 1,
            method: request.method || 'historical',
            portfolio_value: request.portfolioValue || 0,
        }, 15);
    }

    async calculateCVaR(request: CVaRRequest): Promise<CVaRResponse> {
        return this.callGrpc<CVaRResponse>('CalculateCVaR', {
            returns: request.returns,
            confidence_level: request.confidenceLevel || 0.95,
            time_horizon: request.timeHorizon || 1,
            portfolio_value: request.portfolioValue || 0,
        }, 15);
    }

    async stressTest(request: StressTestRequest): Promise<StressTestResponse> {
        return this.callGrpc<StressTestResponse>('StressTest', {
            symbols: request.symbols,
            positions: request.positions,
            scenarios: request.scenarios || [],
        }, 30);
    }

    // =========================================================================
    // PORTFOLIO OPTIMIZATION (NEW)
    // =========================================================================

    async optimizePortfolio(request: PortfolioOptRequest): Promise<PortfolioOptResponse> {
        return this.callGrpc<PortfolioOptResponse>('OptimizePortfolio', {
            symbols: request.symbols,
            expected_returns: request.expectedReturns || [],
            covariance_matrix: request.covarianceMatrix || [],
            method: request.method || 1, // MEAN_VARIANCE
            target_return: request.targetReturn || 0,
            risk_free_rate: request.riskFreeRate || 0,
            constraints: request.constraints || {},
        }, 60);
    }

    async calculateHRP(request: HRPRequest): Promise<HRPResponse> {
        return this.callGrpc<HRPResponse>('CalculateHRP', {
            symbols: request.symbols,
            returns_matrix: request.returnsMatrix,
            n_periods: request.nPeriods,
            n_assets: request.nAssets,
            linkage_method: request.linkageMethod || 'ward',
        }, 60);
    }

    async blackLitterman(request: BlackLittermanRequest): Promise<BlackLittermanResponse> {
        return this.callGrpc<BlackLittermanResponse>('BlackLitterman', {
            symbols: request.symbols,
            market_caps: request.marketCaps,
            covariance_matrix: request.covarianceMatrix,
            risk_aversion: request.riskAversion || 2.5,
            tau: request.tau || 0.05,
            views: request.views || [],
        }, 60);
    }

    // =========================================================================
    // MARKET ANALYSIS
    // =========================================================================

    async detectRegime(request: RegimeDetectionRequest): Promise<RegimeDetectionResponse> {
        return this.callGrpc<RegimeDetectionResponse>('DetectRegime', {
            symbol: request.symbol,
            prices: request.prices || [],
            volumes: request.volumes || [],
            lookback_period: request.lookbackPeriod || 100,
        }, 15);
    }

    async analyzeSentiment(request: SentimentRequest): Promise<SentimentResponse> {
        return this.callGrpc<SentimentResponse>('AnalyzeSentiment', {
            symbol: request.symbol,
            texts: request.texts || [],
            source: request.source || 'mixed',
        }, 20);
    }

    async getMarketMicrostructure(request: MicrostructureRequest): Promise<MicrostructureResponse> {
        return this.callGrpc<MicrostructureResponse>('GetMarketMicrostructure', {
            symbol: request.symbol,
            bid_prices: request.bidPrices || [],
            ask_prices: request.askPrices || [],
            bid_sizes: request.bidSizes || [],
            ask_sizes: request.askSizes || [],
            trade_prices: request.tradePrices || [],
            trade_sizes: request.tradeSizes || [],
            timestamps: request.timestamps || [],
        }, 15);
    }

    // =========================================================================
    // HEALTH & MONITORING
    // =========================================================================

    async healthCheck(): Promise<HealthResponse> {
        if (!this.client) {
            throw new GrpcConnectionError('gRPC client not initialized');
        }

        return new Promise((resolve, reject) => {
            const deadline = new Date();
            deadline.setSeconds(deadline.getSeconds() + 5);

            this.client.HealthCheck({}, { deadline }, (error: grpc.ServiceError | null, response: any) => {
                if (error) {
                    reject(new GrpcServiceError(error.message, error.code, error.details));
                } else {
                    resolve(this.transformResponse<HealthResponse>(response));
                }
            });
        });
    }

    async getServiceMetrics(request: MetricsRequest = {}): Promise<MetricsResponse> {
        return this.callGrpc<MetricsResponse>('GetServiceMetrics', {
            start_time: request.startTime || 0,
            end_time: request.endTime || 0,
        }, 10);
    }

    // =========================================================================
    // GENERIC gRPC CALL WRAPPER
    // =========================================================================

    private async callGrpc<T>(method: string, request: any, timeoutSeconds: number): Promise<T> {
        this.ensureConnected();

        return new Promise((resolve, reject) => {
            const deadline = new Date();
            deadline.setSeconds(deadline.getSeconds() + timeoutSeconds);

            this.client[method](request, { deadline }, (error: grpc.ServiceError | null, response: any) => {
                if (error) {
                    this.handleGrpcError(error, method);
                    reject(new GrpcServiceError(error.message, error.code, error.details));
                } else {
                    resolve(this.transformResponse<T>(response));
                }
            });
        });
    }

    private transformResponse<T>(response: any): T {
        const transformed: any = {};
        for (const key in response) {
            const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
            transformed[camelKey] = response[key];
        }
        return transformed as T;
    }

    private ensureConnected(): void {
        if (!this.isConnected || !this.client) {
            throw new GrpcConnectionError('gRPC client not connected - Use HTTP fallback');
        }
    }

    private handleGrpcError(error: grpc.ServiceError, method: string): void {
        this.logger.error(`gRPC ${method} error: ${error.message} (code: ${error.code})`);

        if (error.code === grpc.status.UNAVAILABLE || error.code === grpc.status.DEADLINE_EXCEEDED) {
            this.isConnected = false;
            this.scheduleReconnect();
        }
    }

    // =========================================================================
    // STATUS METHODS
    // =========================================================================

    isGrpcConnected(): boolean {
        return this.isConnected;
    }

    getConnectionStatus(): ConnectionStatus {
        return {
            connected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxAttempts: this.maxReconnectAttempts,
            protocol: 'gRPC',
        };
    }
}

// =========================================================================
// ENUMS (Matching Proto)
// =========================================================================

export enum OptionType {
    UNSPECIFIED = 0,
    CALL = 1,
    PUT = 2,
}

export enum SignalAction {
    UNSPECIFIED = 0,
    BUY = 1,
    SELL = 2,
    HOLD = 3,
    CLOSE = 4,
}

export enum MarketRegime {
    UNSPECIFIED = 0,
    TRENDING_UP = 1,
    TRENDING_DOWN = 2,
    RANGING = 3,
    VOLATILE = 4,
    CRISIS = 5,
}

export enum ModelType {
    UNSPECIFIED = 0,
    LSTM = 1,
    TRANSFORMER = 2,
    DRL = 3,
    ENSEMBLE = 4,
    XGB = 5,
    LIGHTGBM = 6,
}

export enum VolatilityModel {
    UNSPECIFIED = 0,
    SABR = 1,
    SVI = 2,
    DUPIRE = 3,
    HESTON = 4,
}

export enum RiskProfile {
    UNSPECIFIED = 0,
    CONSERVATIVE = 1,
    MODERATE = 2,
    AGGRESSIVE = 3,
}

export enum ExecutionType {
    UNSPECIFIED = 0,
    AGGRESSIVE = 1,
    NEUTRAL = 2,
    CONSERVATIVE = 3,
    TWAP = 4,
    VWAP = 5,
}

export enum OptimizationMethod {
    UNSPECIFIED = 0,
    MEAN_VARIANCE = 1,
    BLACK_LITTERMAN = 2,
    HRP = 3,
    RISK_PARITY = 4,
    MAX_SHARPE = 5,
    MIN_VARIANCE = 6,
    MAX_DIVERSIFICATION = 7,
}

// =========================================================================
// REQUEST INTERFACES
// =========================================================================

export interface SignalRequest {
    symbol: string;
    timeframe?: string;
    prices?: number[];
    volumes?: number[];
    highPrices?: number[];
    lowPrices?: number[];
    openPrices?: number[];
    indicators?: Record<string, number>;
    modelId?: string;
    currentRegime?: MarketRegime;
    currentVolatility?: number;
}

export interface SignalStreamRequest {
    symbols: string[];
    timeframe?: string;
    modelId?: string;
    updateIntervalMs?: number;
}

export interface ModelSelectionRequest {
    symbol: string;
    regime?: MarketRegime;
    volatility?: number;
    liquidity?: number;
    context?: Record<string, string>;
}

export interface ListModelsRequest {
    filter?: string;
    typeFilter?: ModelType;
    activeOnly?: boolean;
}

export interface ModelMetricsRequest {
    modelId: string;
    startTime?: number;
    endTime?: number;
}

export interface StrategyRequest {
    symbol: string;
    signalAction?: SignalAction;
    confidence?: number;
    marketContext?: Record<string, string>;userRiskProfile?: RiskProfile;
    accountBalance?: number;
    currentExposure?: number;
}

export interface ListStrategiesRequest {
    filter?: string;
    regimeFilter?: MarketRegime;
}

export interface BacktestRequest {
    strategyId: string;
    symbol: string;
    startTime: number;
    endTime: number;
    initialCapital?: number;
    parameters?: Record<string, string>;
}

export interface VolSurfaceRequest {
    symbol: string;
    spotPrice: number;
    riskFreeRate: number;
    dividendYield?: number;
    model?: VolatilityModel;
    marketData?: VolSurfacePoint[];
    beta?: number;
}

export interface VolSurfacePoint {
    strike: number;
    expiry: number;
    volatility: number;
    bidVol?: number;
    askVol?: number;
    optionType?: OptionType;
}

export interface ImpliedVolRequest {
    symbol: string;
    strike: number;
    expiry: number;
    spotPrice: number;
}

export interface LocalVolRequest {
    symbol: string;
    strikeMin: number;
    strikeMax: number;
    expiryMin: number;
    expiryMax: number;
    numStrikes?: number;
    numExpiries?: number;
}

export interface SkewMetricsRequest {
    symbol: string;
    expiry: number;
}

export interface ArbitrageCheckRequest {
    symbol: string;
}

export interface GreeksRequest {
    symbol: string;
    optionType?: OptionType;
    spotPrice: number;
    strike: number;
    expiry: number;
    riskFreeRate: number;
    dividendYield?: number;
    volatility: number;
    pricingModel?: string;
}

export interface OptionPricingRequest {
    symbol: string;
    optionType?: OptionType;
    spotPrice: number;
    strike: number;
    expiry: number;
    riskFreeRate: number;
    dividendYield?: number;
    volatility: number;
    pricingModel?: string;
    americanStyle?: boolean;
}

export interface OptionsChainRequest {
    symbol: string;
    spotPrice: number;
    expiries?: number[];
    numStrikes?: number;
    strikeRangePct?: number;
}

export interface RiskMetricsRequest {
    returns: number[];
    confidenceLevel?: number;
    timeHorizon?: number;
    riskFreeRate?: number;
}

export interface VaRRequest {
    returns: number[];
    confidenceLevel?: number;
    timeHorizon?: number;
    method?: string;
    portfolioValue?: number;
}

export interface CVaRRequest {
    returns: number[];
    confidenceLevel?: number;
    timeHorizon?: number;
    portfolioValue?: number;
}

export interface StressTestRequest {
    symbols: string[];
    positions: number[];
    scenarios?: StressScenario[];
}

export interface StressScenario {
    name: string;
    priceShocks: Record<string, number>;
    volatilityShock?: number;
    correlationShock?: number;
}

export interface PortfolioOptRequest {
    symbols: string[];
    expectedReturns?: number[];
    covarianceMatrix?: number[];
    method?: OptimizationMethod;
    targetReturn?: number;
    riskFreeRate?: number;
    constraints?: PortfolioConstraints;
}

export interface PortfolioConstraints {
    minWeight?: number;
    maxWeight?: number;
    maxSectorWeight?: number;
    sectorAssignments?: string[];
    longOnly?: boolean;
}

export interface HRPRequest {
    symbols: string[];
    returnsMatrix: number[];
    nPeriods: number;
    nAssets: number;
    linkageMethod?: string;
}

export interface BlackLittermanRequest {
    symbols: string[];
    marketCaps: number[];
    covarianceMatrix: number[];
    riskAversion?: number;
    tau?: number;
    views?: ViewInfo[];
}

export interface ViewInfo {
    symbols: string[];
    weights: number[];
    expectedReturn: number;confidence: number;
}

export interface RegimeDetectionRequest {
    symbol: string;
    prices?: number[];
    volumes?: number[];
    lookbackPeriod?: number;
}

export interface SentimentRequest {
    symbol: string;
    texts?: string[];
    source?: string;
}

export interface MicrostructureRequest {
    symbol: string;
    bidPrices?: number[];
    askPrices?: number[];
    bidSizes?: number[];
    askSizes?: number[];
    tradePrices?: number[];
    tradeSizes?: number[];
    timestamps?: number[];
}

export interface MetricsRequest {
    startTime?: number;
    endTime?: number;
}

// =========================================================================
// RESPONSE INTERFACES
// =========================================================================

export interface SignalResponse {
    action: SignalAction;
    confidence: number;
    targetPrice: number;
    stopLoss: number;
    takeProfit: number;
    reasoning: string;
    features: Record<string, number>;
    timestamp: number;riskRewardRatio: number;
    expectedReturn: number;
    maxDrawdownEstimate: number;
    contributors: SignalContributor[];
}

export interface SignalContributor {
    name: string;
    weight: number;
    signalValue: number;
}

export interface ModelSelectionResponse {
    modelId: string;
    modelName: string;
    confidence: number;
    reasoning: string;
    metadata: Record<string, string>;
    performance: ModelPerformance;
}

export interface ModelPerformance {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    sharpeRatio: number;
    sortinoRatio: number;
    totalPredictions: number;
}

export interface ListModelsResponse {
    models: ModelInfo[];
}

export interface ModelInfo {
    id: string;
    name: string;
    type: ModelType;
    version: string;
    isActive: boolean;
    accuracy: number;
    lastTrained: number;
    supportedSymbols: string[];
    supportedTimeframes: string[];
    performance: ModelPerformance;
}

export interface ModelMetricsResponse {
    modelId: string;
    performance: ModelPerformance;
    recentPredictions: PredictionRecord[];
    confusionMatrix: ConfusionMatrix;
}

export interface PredictionRecord {
    timestamp: number;
    predicted: SignalAction;
    actual: SignalAction;
    confidence: number;
    pnl: number;
}

export interface ConfusionMatrix {
    truePositive: number;
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
}

export interface StrategyResponse {
    strategyId: string;
    strategyName: string;
    executionType: ExecutionType;
    positionSize: number;
    positionSizeModifier: number;
    parameters: Record<string, string>;
    reasoning: string;
    riskLimits: RiskLimits;
}

export interface RiskLimits {
    maxPositionSize: number;
    maxDailyLoss: number;
    maxDrawdown: number;
    stopLossPct: number;
    takeProfitPct: number;
}

export interface ListStrategiesResponse {
    strategies: StrategyInfo[];
}

export interface StrategyInfo {
    id: string;
    name: string;
    description: string;
    supportedRegimes: MarketRegime[];
    isActive: boolean;
    performance: StrategyPerformance;
}

export interface StrategyPerformance {
    totalReturn: number;
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    winRate: number;
    profitFactor: number;
    totalTrades: number;
}

export interface BacktestResponse {
    performance: StrategyPerformance;
    trades: TradeRecord[];
    equityCurve: EquityCurve;
    drawdownAnalysis: DrawdownAnalysis;
}

export interface TradeRecord {
    entryTime: number;
    exitTime: number;
    action: SignalAction;
    entryPrice: number;
    exitPrice: number;
    positionSize: number;
    pnl: number;
    pnlPct: number;
}

export interface EquityCurve {
    timestamps: number[];
    values: number[];
    returns: number[];
}

export interface DrawdownAnalysis {
    maxDrawdown: number;
    avgDrawdown: number;
    maxDrawdownDuration: number;
    periods: DrawdownPeriod[];
}

export interface DrawdownPeriod {
    startTime: number;
    endTime: number;
    depth: number;
}

export interface VolSurfaceResponse {
    success: boolean;
    model: VolatilityModel;
    rmse: number;
    maxError: number;
    calibrationTimeMs: number;
    smiles: CalibratedSmile[];
    errorMessage: string;
}

export interface CalibratedSmile {
    expiry: number;
    forward: number;
    atmVol: number;
    sabrParams: SABRParams;
    sviParams: SVIParams;
}

export interface SABRParams {
    alpha: number;
    beta: number;
    rho: number;
    nu: number;
}

export interface SVIParams {
    a: number;
    b: number;
    rho: number;
    m: number;
    sigma: number;
}

export interface ImpliedVolResponse {
    impliedVolatility: number;
    forwardPrice: number;
    moneyness: number;
    logMoneyness: number;
}

export interface LocalVolResponse {
    strikes: number[];
    expiries: number[];
    localVols: number[];
    impliedVols: number[];
}

export interface SkewMetricsResponse {
    expiry: number;
    atmVol: number;
    skew25d: number;
    skew10d: number;
    riskReversal25d: number;
    butterfly25d: number;
    smileCurvature: number;
}

export interface ArbitrageCheckResponse {
    hasCalendarArbitrage: boolean;
    hasButterflyArbitrage: boolean;
    arbitragePoints: ArbitragePoint[];maxArbitrageAmount: number;
}

export interface ArbitragePoint {
    strike: number;
    expiry: number;
    arbitrageType: string;
    amount: number;
}

export interface GreeksResponse {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
    vanna: number;
    volga: number;
    charm: number;
    veta: number;
    speed: number;
    zomma: number;
    color: number;
}

export interface OptionPricingResponse {
    price: number;
    intrinsicValue: number;
    timeValue: number;
    greeks: GreeksResponse;
    impliedVolatility: number;
    bidPrice: number;
    askPrice: number;
}

export interface OptionsChainResponse {
    expiries: OptionChainExpiry[];
}

export interface OptionChainExpiry {
    expiry: number;
    strikes: OptionChainStrike[];
}

export interface OptionChainStrike {
    strike: number;
    call:OptionPricingResponse;
    put: OptionPricingResponse;
}

export interface RiskMetricsResponse {
    volatility: number;
    var95: number;
    var99: number;
    cvar95: number;
    cvar99: number;
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxDrawdown: number;
    skewness: number;
    kurtosis: number;
    omegaRatio: number;
    tailRatio: number;
}

export interface VaRResponse {
    varAmount: number;
    varPercentage: number;
    method: string;
    confidenceLevel: number;
    timeHorizon: number;
}

export interface CVaRResponse {
    cvarAmount: number;
    cvarPercentage: number;
    varAmount: number;
    expectedShortfall: number;
    tailLossMean: number;
}

export interface StressTestResponse {
    results: StressTestResult[];
    worstCaseLoss: number;
    worstScenario: string;
}

export interface StressTestResult {
    scenarioName: string;
    portfolioPnl: number;
    portfolioPnlPct: number;
    positionPnls: Record<string, number>;
}

export interface PortfolioOptResponse {
    weights: number[];
    expectedReturn: number;
    expectedVolatility: number;
    sharpeRatio: number;
    diversificationRatio: number;
    symbolWeights: Record<string, number>;
    efficientFrontier: EfficientFrontier;
}

export interface EfficientFrontier {
    returns: number[];
    volatilities: number[];
    sharpeRatios: number[];
}

export interface HRPResponse {
    weights: number[];
    symbolWeights: Record<string, number>;
    clusters: ClusterInfo[];
    portfolioVariance: number;
}

export interface ClusterInfo {
    symbols: string[];
    intraClusterCorrelation: number;
    clusterId: number;
}

export interface BlackLittermanResponse {
    posteriorReturns: number[];
    optimalWeights: number[];
    symbolWeights: Record<string, number>;
    priorReturns: number[];
    expectedReturn: number;
    expectedVolatility: number;
}

export interface RegimeDetectionResponse {
    currentRegime: MarketRegime;
    confidence: number;
    probabilities: RegimeProbability[];
    regimeStartTime: number;
    regimeDurationBars: number;
    transitionMatrix: RegimeTransitionMatrix;
}

export interface RegimeProbability {
    regime: MarketRegime;
    probability: number;
}

export interface RegimeTransitionMatrix {
    matrix: number[];
    states: MarketRegime[];
}

export interface SentimentResponse {
    overallSentiment: number;
    bullishRatio: number;
    bearishRatio: number;
    neutralRatio: number;
    items: SentimentItem[];
    trend: SentimentTrend;
}

export interface SentimentItem {
    text: string;
    score: number;
    label: string;
    confidence: number;
    keywords: string[];
}

export interface SentimentTrend {
    momentum: number;
    acceleration: number;
    bullishStreak: number;
    bearishStreak: number;
}

export interface MicrostructureResponse {
    bidAskSpread: number;
    effectiveSpread: number;
    realizedSpread: number;
    priceImpact: number;
    kyleLambda: number;
    amihudIlliquidity: number;
    orderImbalance: number;
    vpin: number;
    toxicity: ToxicityMetrics;
}

export interface ToxicityMetrics {
    informedTradingProbability: number;
    adverseSelectionCost: number;
    flowToxicity: number;
}

export interface HealthResponse {
    healthy: boolean;
    version: string;
    uptimeSeconds: number;
    services: Record<string, boolean>;
    resources: SystemResources;
}

export interface SystemResources {
    cpuUsage: number;
    memoryUsage: number;
    gpuUsage: number;
    activeConnections: number;
}

export interface MetricsResponse {
    totalRequests: number;
    avgLatencyMs: number;
    p99LatencyMs: number;
    requestsByMethod: Record<string, number>;
    errorRates: Record<string, number>;
    latencyHistogram: LatencyBucket[];
}

export interface LatencyBucket {
    lowerBoundMs: number;
    upperBoundMs: number;
    count: number;
}

export interface ConnectionStatus {
    connected: boolean;
    reconnectAttempts: number;
    maxAttempts: number;
    protocol: string;
}

// =========================================================================
// CUSTOM ERRORS
// =========================================================================

export class GrpcConnectionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'GrpcConnectionError';
    }
}

export class GrpcServiceError extends Error {
    constructor(
        message: string,
        public readonly code: number,
        public readonly details?: string,) {
        super(message);
        this.name = 'GrpcServiceError';
    }
}