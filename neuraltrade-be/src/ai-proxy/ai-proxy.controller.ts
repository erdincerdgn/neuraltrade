import {
    Controller,
    Get,
    Post,
    Body,
    Param,
    Query,
    UseGuards,
    HttpStatus,
    HttpCode,
} from '@nestjs/common';
import {
    ApiTags,
    ApiOperation,
    ApiResponse,
    ApiBearerAuth,
    ApiParam,
} from '@nestjs/swagger';
import { AIProxyService } from './ai-proxy.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { TradingGuard } from '../auth/guards/trading.guard';
import {
    // Signal & Prediction
    SignalRequestDto,
    GenerateSignalDto,
    BatchSignalsDto,
    // Model Management
    ModelSelectionRequestDto,
    ListModelsRequestDto,
    ModelMetricsRequestDto,
    // Strategy Routing
    StrategyRequestDto,
    ListStrategiesRequestDto,
    BacktestRequestDto,
    // Volatility Surface
    VolSurfaceRequestDto,
    ImpliedVolRequestDto,
    LocalVolRequestDto,
    SkewMetricsRequestDto,
    ArbitrageCheckRequestDto,
    // Options & Greeks
    GreeksRequestDto,
    OptionPricingRequestDto,
    OptionsChainRequestDto,
    // Risk Management
    RiskMetricsRequestDto,
    VaRRequestDto,
    CVaRRequestDto,
    StressTestRequestDto,
    RiskAssessmentDto,
    // Portfolio Optimization
    PortfolioOptRequestDto,
    HRPRequestDto,
    BlackLittermanRequestDto,
    PortfolioOptimizeDto,
    // Market Analysis
    RegimeDetectionRequestDto,
    SentimentRequestDto,
    MicrostructureRequestDto,
    // Other
    MetricsRequestDto,
    RAGQueryDto,
    PipelineRunDto,
} from './dto';

/**
 * NeuralTrade AI Proxy Controller v2.0
 * 
 * REST API endpoints for AI Engine communication.
 * Uses gRPC as PRIMARY protocol, HTTP as FALLBACK.
 * 
 * Aligned with ai_service.proto v2.0
 * 
 * @version 2.0.0
 * @author Senior Quant Developer
 */
@ApiTags('AI')
@Controller({ path: 'ai', version: '2' })
export class AIProxyController {
    constructor(private readonly aiProxyService: AIProxyService) {}

    // ============================================================
    // HEALTH & STATUS
    // ============================================================

    @Get('health')
    @ApiOperation({ summary: 'Check AI Engine connection status' })
    @ApiResponse({ status: 200, description: 'AI Engine health status' })
    async getHealth() {
        return this.aiProxyService.checkHealth();
    }

    @Get('metrics')
    @ApiOperation({ summary: 'Get AI Engine service metrics' })
    @ApiResponse({ status: 200, description: 'Service metrics' })
    async getMetrics(@Query() query: MetricsRequestDto) {
        return this.aiProxyService.getServiceMetrics(query);
    }

    @Get('protocol')
    @ApiOperation({ summary: 'Get current communication protocol' })
    @ApiResponse({ status: 200, description: 'Current protocol' })
    async getProtocol() {
        return {
            protocol: this.aiProxyService.getProtocol(),
            grpcConnected: this.aiProxyService.isGrpcConnected(),
        };
    }

    // ============================================================
    // SIGNAL & PREDICTION
    // ============================================================

    @Post('signals/predict')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Predict trading signal (Proto v2.0)' })
    @ApiResponse({ status: 200, description: 'Signal prediction result' })
    async predictSignal(@Body() dto: SignalRequestDto) {
        return this.aiProxyService.predictSignal(dto);
    }

    @Post('signals/generate')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Generate trading signal (Legacy)' })
    @ApiResponse({ status: 200, description: 'Signal generated' })
    async generateSignal(@Body() dto: GenerateSignalDto) {
        return this.aiProxyService.generateSignal(dto);
    }

    @Post('signals/batch')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Generate signals for multiple symbols' })
    @ApiResponse({ status: 200, description: 'Batch signals generated' })
    async batchGenerateSignals(@Body() dto: BatchSignalsDto) {
        return this.aiProxyService.batchGenerateSignals(dto);
    }

    // ============================================================
    // MODEL MANAGEMENT
    // ============================================================

    @Post('models/select')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Select optimal model' })
    @ApiResponse({ status: 200, description: 'Model selected' })
    async selectModel(@Body() dto: ModelSelectionRequestDto) {
        return this.aiProxyService.selectModel(dto);
    }

    @Get('models')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @ApiOperation({ summary: 'List available models' })
    @ApiResponse({ status: 200, description: 'Models list' })
    async listModels(@Query() query: ListModelsRequestDto) {
        return this.aiProxyService.listModels(query);
    }

    @Get('models/:modelId/metrics')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @ApiOperation({ summary: 'Get model metrics' })
    @ApiParam({ name: 'modelId', description: 'Model ID' })
    @ApiResponse({ status: 200, description: 'Model metrics' })
    async getModelMetrics(
        @Param('modelId') modelId: string,
        @Query() query: Omit<ModelMetricsRequestDto, 'modelId'>,
    ) {
        return this.aiProxyService.getModelMetrics({ modelId, ...query });
    }

    // ============================================================
    // STRATEGY ROUTING
    // ============================================================

    @Post('strategies/route')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Route to optimal strategy' })
    @ApiResponse({ status: 200, description: 'Strategy routed' })
    async routeStrategy(@Body() dto: StrategyRequestDto) {
        return this.aiProxyService.routeStrategy(dto);
    }

    @Get('strategies')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @ApiOperation({ summary: 'List available strategies' })
    @ApiResponse({ status: 200, description: 'Strategies list' })
    async listStrategies(@Query() query: ListStrategiesRequestDto) {
        return this.aiProxyService.listStrategies(query);
    }

    @Post('strategies/backtest')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Backtest a strategy' })
    @ApiResponse({ status: 200, description: 'Backtest results' })
    async backtestStrategy(@Body() dto: BacktestRequestDto) {
        return this.aiProxyService.backtestStrategy(dto);
    }

    // ============================================================
    // VOLATILITY SURFACE
    // ============================================================

    @Post('volatility/surface/calibrate')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calibrate volatility surface (SABR/SVI/Dupire)' })
    @ApiResponse({ status: 200, description: 'Calibration result' })
    async calibrateVolatilitySurface(@Body() dto: VolSurfaceRequestDto) {
        return this.aiProxyService.calibrateVolatilitySurface(dto);
    }

    @Post('volatility/implied')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Get implied volatility' })
    @ApiResponse({ status: 200, description: 'Implied volatility' })
    async getImpliedVolatility(@Body() dto: ImpliedVolRequestDto) {
        return this.aiProxyService.getImpliedVolatility(dto);
    }

    @Post('volatility/local')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Get local volatility surface' })
    @ApiResponse({ status: 200, description: 'Local volatility' })
    async getLocalVolatility(@Body() dto: LocalVolRequestDto) {
        return this.aiProxyService.getLocalVolatility(dto);
    }

    @Post('volatility/skew')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Get skew metrics' })
    @ApiResponse({ status: 200, description: 'Skew metrics' })
    async getSkewMetrics(@Body() dto: SkewMetricsRequestDto) {
        return this.aiProxyService.getSkewMetrics(dto);
    }

    @Post('volatility/arbitrage')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Check for surface arbitrage' })
    @ApiResponse({ status: 200, description: 'Arbitrage check result' })
    async checkSurfaceArbitrage(@Body() dto: ArbitrageCheckRequestDto) {
        return this.aiProxyService.checkSurfaceArbitrage(dto);
    }

    // ============================================================
    // OPTIONS & GREEKS
    // ============================================================

    @Post('options/greeks')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calculate option Greeks' })
    @ApiResponse({ status: 200, description: 'Greeks calculated' })
    async calculateGreeks(@Body() dto: GreeksRequestDto) {
        return this.aiProxyService.calculateGreeks(dto);
    }

    @Post('options/price')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Price an option' })
    @ApiResponse({ status: 200, description: 'Option priced' })
    async priceOption(@Body() dto: OptionPricingRequestDto) {
        return this.aiProxyService.priceOption(dto);
    }

    @Post('options/chain')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Get options chain' })
    @ApiResponse({ status: 200, description: 'Options chain' })
    async getOptionsChain(@Body() dto: OptionsChainRequestDto) {
        return this.aiProxyService.getOptionsChain(dto);
    }

    // ============================================================
    // RISK MANAGEMENT
    // ============================================================

    @Post('risk/metrics')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calculate risk metrics' })
    @ApiResponse({ status: 200, description: 'Risk metrics' })
    async calculateRiskMetrics(@Body() dto: RiskMetricsRequestDto) {
        return this.aiProxyService.calculateRiskMetrics(dto);
    }

    @Post('risk/var')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calculate Value at Risk' })
    @ApiResponse({ status: 200, description: 'VaR result' })
    async calculateVaR(@Body() dto: VaRRequestDto) {
        return this.aiProxyService.calculateVaR(dto);
    }

    @Post('risk/cvar')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calculate Conditional VaR (Expected Shortfall)' })
    @ApiResponse({ status: 200, description: 'CVaR result' })
    async calculateCVaR(@Body() dto: CVaRRequestDto) {
        return this.aiProxyService.calculateCVaR(dto);
    }

    @Post('risk/stress-test')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Run stress test scenarios' })
    @ApiResponse({ status: 200, description: 'Stress test results' })
    async stressTest(@Body() dto: StressTestRequestDto) {
        return this.aiProxyService.stressTest(dto);
    }

    @Post('risk/assess')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Assess position risk (Legacy)' })
    @ApiResponse({ status: 200, description: 'Risk assessment' })
    async assessRisk(@Body() dto: RiskAssessmentDto) {
        return this.aiProxyService.assessRisk(dto);
    }

    // ============================================================
    // PORTFOLIO OPTIMIZATION
    // ============================================================

    @Post('portfolio/optimize')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Optimize portfolio (Mean-Variance/Black-Litterman/HRP)' })
    @ApiResponse({ status: 200, description: 'Optimization result' })
    async optimizePortfolio(@Body() dto: PortfolioOptRequestDto) {
        return this.aiProxyService.optimizePortfolio(dto);
    }

    @Post('portfolio/hrp')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Calculate Hierarchical Risk Parity weights' })
    @ApiResponse({ status: 200, description: 'HRP result' })
    async calculateHRP(@Body() dto: HRPRequestDto) {
        return this.aiProxyService.calculateHRP(dto);
    }

    @Post('portfolio/black-litterman')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Black-Litterman optimization' })
    @ApiResponse({ status: 200, description: 'Black-Litterman result' })
    async blackLitterman(@Body() dto: BlackLittermanRequestDto) {
        return this.aiProxyService.blackLitterman(dto);
    }

    @Post('portfolio/optimize-legacy')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Optimize portfolio (Legacy)' })
    @ApiResponse({ status: 200, description: 'Optimization result' })
    async optimizePortfolioLegacy(@Body() dto: PortfolioOptimizeDto) {
        return this.aiProxyService.optimizePortfolioLegacy(dto);
    }

    // ============================================================
    // MARKET ANALYSIS
    // ============================================================

    @Post('market/regime')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Detect market regime' })
    @ApiResponse({ status: 200, description: 'Regime detection result' })
    async detectRegime(@Body() dto: RegimeDetectionRequestDto) {
        return this.aiProxyService.detectRegime(dto);
    }

    @Post('market/sentiment')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Analyze market sentiment' })
    @ApiResponse({ status: 200, description: 'Sentiment analysis result' })
    async analyzeSentiment(@Body() dto: SentimentRequestDto) {
        return this.aiProxyService.analyzeSentiment(dto);
    }

    @Post('market/microstructure')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Get market microstructure metrics' })
    @ApiResponse({ status: 200, description: 'Microstructure metrics' })
    async getMarketMicrostructure(@Body() dto: MicrostructureRequestDto) {
        return this.aiProxyService.getMarketMicrostructure(dto);
    }

    // ============================================================
    // RAG QUERY
    // ============================================================

    @Post('rag/query')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Query AI knowledge base (RAG)' })
    @ApiResponse({ status: 200, description: 'RAG response' })
    async queryRAG(@Body() dto: RAGQueryDto) {
        return this.aiProxyService.queryRAG(dto);
    }

    // ============================================================
    // FULL PIPELINE
    // ============================================================

    @Post('pipeline/run')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Run full AI trading pipeline' })
    @ApiResponse({ status: 200, description: 'Pipeline result' })
    async runPipeline(@Body() dto: PipelineRunDto) {
        return this.aiProxyService.runPipeline(dto);
    }

    @Post('pipeline/run/:symbol')
    @UseGuards(JwtAuthGuard, TradingGuard)
    @ApiBearerAuth()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Run pipeline for symbol (URL param)' })
    @ApiParam({ name: 'symbol', description: 'Trading symbol' })
    @ApiResponse({ status: 200, description: 'Pipeline result' })
    async runPipelineBySymbol(@Param('symbol') symbol: string) {
        return this.aiProxyService.runPipeline({ symbol });
    }
}