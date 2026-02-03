import {IsString,
    IsOptional,
    IsBoolean,
    IsArray,
    IsNumber,
    IsEnum,
    IsObject,
    Min,
    Max,
    ValidateNested,
} from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { Type } from 'class-transformer';

//============================================================
// ENUMS (Matching Proto v2.0)
// ============================================================

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

// ============================================================
// SIGNAL PREDICTION DTOs
// ============================================================

export class SignalRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Timeframe', default: '1h' })
    @IsOptional()
    @IsString()
    timeframe?: string;

    @ApiPropertyOptional({ description: 'Price data array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    prices?: number[];

    @ApiPropertyOptional({ description: 'Volume data array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    volumes?: number[];

    @ApiPropertyOptional({ description: 'High prices array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    highPrices?: number[];

    @ApiPropertyOptional({ description: 'Low prices array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    lowPrices?: number[];

    @ApiPropertyOptional({ description: 'Open prices array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    openPrices?: number[];

    @ApiPropertyOptional({ description: 'Technical indicators' })
    @IsOptional()
    @IsObject()
    indicators?: Record<string, number>;

    @ApiPropertyOptional({ description: 'Specific model ID to use' })
    @IsOptional()
    @IsString()
    modelId?: string;

    @ApiPropertyOptional({ enum: MarketRegime })
    @IsOptional()
    @IsEnum(MarketRegime)
    currentRegime?: MarketRegime;

    @ApiPropertyOptional({ description: 'Current volatility level' })
    @IsOptional()
    @IsNumber()
    currentVolatility?: number;
}

export class SignalStreamRequestDto {
    @ApiProperty({ description: 'Array of symbols to stream', example: ['AAPL', 'GOOGL'] })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiPropertyOptional({ description: 'Timeframe', default: '1h' })
    @IsOptional()
    @IsString()
    timeframe?: string;

    @ApiPropertyOptional({ description: 'Model ID' })
    @IsOptional()
    @IsString()
    modelId?: string;

    @ApiPropertyOptional({ description: 'Update interval in milliseconds', default: 1000 })
    @IsOptional()
    @IsNumber()
    updateIntervalMs?: number;
}

// ============================================================
// MODEL MANAGEMENT DTOs
// ============================================================

export class ModelSelectionRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ enum: MarketRegime })
    @IsOptional()
    @IsEnum(MarketRegime)
    regime?: MarketRegime;

    @ApiPropertyOptional({ description: 'Current volatility (0-1)' })
    @IsOptional()
    @IsNumber()
    @Min(0)
    @Max(1)
    volatility?: number;

    @ApiPropertyOptional({ description: 'Liquidity level' })
    @IsOptional()
    @IsNumber()
    liquidity?: number;

    @ApiPropertyOptional({ description: 'Additional context' })
    @IsOptional()
    @IsObject()
    context?: Record<string, string>;
}

export class ListModelsRequestDto {
    @ApiPropertyOptional({ description: 'Filter string' })
    @IsOptional()
    @IsString()
    filter?: string;

    @ApiPropertyOptional({ enum: ModelType })
    @IsOptional()
    @IsEnum(ModelType)
    typeFilter?: ModelType;

    @ApiPropertyOptional({ description: 'Only active models', default: true })
    @IsOptional()
    @IsBoolean()
    activeOnly?: boolean;
}

export class ModelMetricsRequestDto {
    @ApiProperty({ description: 'Model ID' })
    @IsString()
    modelId: string;

    @ApiPropertyOptional({ description: 'Start time (Unix timestamp)' })
    @IsOptional()
    @IsNumber()
    startTime?: number;

    @ApiPropertyOptional({ description: 'End time (Unix timestamp)' })
    @IsOptional()
    @IsNumber()
    endTime?: number;
}

// ============================================================
// STRATEGY ROUTING DTOs
// ============================================================

export class StrategyRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ enum: SignalAction })
    @IsOptional()
    @IsEnum(SignalAction)
    signalAction?: SignalAction;

    @ApiPropertyOptional({ description: 'Signal confidence (0-1)' })
    @IsOptional()
    @IsNumber()
    @Min(0)
    @Max(1)
    confidence?: number;

    @ApiPropertyOptional({ description: 'Market context' })
    @IsOptional()
    @IsObject()
    marketContext?: Record<string, string>;

    @ApiPropertyOptional({ enum: RiskProfile, default: RiskProfile.MODERATE })
    @IsOptional()
    @IsEnum(RiskProfile)
    userRiskProfile?: RiskProfile;

    @ApiPropertyOptional({ description: 'Account balance' })
    @IsOptional()
    @IsNumber()
    accountBalance?: number;

    @ApiPropertyOptional({ description: 'Current exposure' })
    @IsOptional()
    @IsNumber()
    currentExposure?: number;
}

export class ListStrategiesRequestDto {
    @ApiPropertyOptional({ description: 'Filter string' })
    @IsOptional()
    @IsString()
    filter?: string;

    @ApiPropertyOptional({ enum: MarketRegime })
    @IsOptional()
    @IsEnum(MarketRegime)
    regimeFilter?: MarketRegime;
}

export class BacktestRequestDto {
    @ApiProperty({ description: 'Strategy ID' })
    @IsString()
    strategyId: string;

    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Start time (Unix timestamp)' })
    @IsNumber()
    startTime: number;

    @ApiProperty({ description: 'End time (Unix timestamp)' })
    @IsNumber()
    endTime: number;

    @ApiPropertyOptional({ description: 'Initial capital', default: 100000 })
    @IsOptional()
    @IsNumber()
    initialCapital?: number;

    @ApiPropertyOptional({ description: 'Strategy parameters' })
    @IsOptional()
    @IsObject()
    parameters?: Record<string, string>;
}

// ============================================================
// VOLATILITY SURFACE DTOs
// ============================================================

export class VolSurfacePointDto {
    @ApiProperty({ description: 'Strike price' })
    @IsNumber()
    strike: number;

    @ApiProperty({ description: 'Time to expiry (years)' })
    @IsNumber()
    expiry: number;

    @ApiProperty({ description: 'Implied volatility' })
    @IsNumber()
    volatility: number;

    @ApiPropertyOptional({ description: 'Bid volatility' })
    @IsOptional()
    @IsNumber()
    bidVol?: number;

    @ApiPropertyOptional({ description: 'Ask volatility' })
    @IsOptional()
    @IsNumber()
    askVol?: number;

    @ApiPropertyOptional({ enum: OptionType })
    @IsOptional()
    @IsEnum(OptionType)
    optionType?: OptionType;
}

export class VolSurfaceRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Spot price' })
    @IsNumber()
    spotPrice: number;

    @ApiProperty({ description: 'Risk-free rate' })
    @IsNumber()
    riskFreeRate: number;

    @ApiPropertyOptional({ description: 'Dividend yield', default: 0 })
    @IsOptional()
    @IsNumber()
    dividendYield?: number;

    @ApiPropertyOptional({ enum: VolatilityModel, default: VolatilityModel.SABR })
    @IsOptional()
    @IsEnum(VolatilityModel)
    model?: VolatilityModel;

    @ApiPropertyOptional({ type: [VolSurfacePointDto], description: 'Market data points' })
    @IsOptional()
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => VolSurfacePointDto)
    marketData?: VolSurfacePointDto[];

    @ApiPropertyOptional({ description: 'SABR beta parameter', default: 0.5 })
    @IsOptional()
    @IsNumber()
    beta?: number;
}

export class ImpliedVolRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Strike price' })
    @IsNumber()
    strike: number;

    @ApiProperty({ description: 'Time to expiry (years)' })
    @IsNumber()
    expiry: number;

    @ApiProperty({ description: 'Spot price' })
    @IsNumber()
    spotPrice: number;
}

export class LocalVolRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Minimum strike' })
    @IsNumber()
    strikeMin: number;

    @ApiProperty({ description: 'Maximum strike' })
    @IsNumber()
    strikeMax: number;

    @ApiProperty({ description: 'Minimum expiry' })
    @IsNumber()
    expiryMin: number;

    @ApiProperty({ description: 'Maximum expiry' })
    @IsNumber()
    expiryMax: number;

    @ApiPropertyOptional({ description: 'Number of strikes', default: 20 })
    @IsOptional()
    @IsNumber()
    numStrikes?: number;

    @ApiPropertyOptional({ description: 'Number of expiries', default: 10 })
    @IsOptional()
    @IsNumber()
    numExpiries?: number;
}

export class SkewMetricsRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Time to expiry (years)' })
    @IsNumber()
    expiry: number;
}

export class ArbitrageCheckRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;
}

// ============================================================
// OPTIONS & GREEKS DTOs
// ============================================================

export class GreeksRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ enum: OptionType, default: OptionType.CALL })
    @IsOptional()
    @IsEnum(OptionType)
    optionType?: OptionType;

    @ApiProperty({ description: 'Spot price' })
    @IsNumber()
    spotPrice: number;

    @ApiProperty({ description: 'Strike price' })
    @IsNumber()
    strike: number;

    @ApiProperty({ description: 'Time to expiry (years)' })
    @IsNumber()
    expiry: number;

    @ApiProperty({ description: 'Risk-free rate' })
    @IsNumber()
    riskFreeRate: number;

    @ApiPropertyOptional({ description: 'Dividend yield', default: 0 })
    @IsOptional()
    @IsNumber()
    dividendYield?: number;

    @ApiProperty({ description: 'Volatility' })
    @IsNumber()
    volatility: number;

    @ApiPropertyOptional({ description: 'Pricing model', default: 'black_scholes' })
    @IsOptional()
    @IsString()
    pricingModel?: string;
}

export class OptionPricingRequestDto extends GreeksRequestDto {
    @ApiPropertyOptional({ description: 'American style option', default: false })
    @IsOptional()
    @IsBoolean()
    americanStyle?: boolean;
}

export class OptionsChainRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Spot price' })
    @IsNumber()
    spotPrice: number;

    @ApiPropertyOptional({ description: 'Expiries array (years)' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    expiries?: number[];

    @ApiPropertyOptional({ description: 'Number of strikes', default: 10 })
    @IsOptional()
    @IsNumber()
    numStrikes?: number;

    @ApiPropertyOptional({ description: 'Strike range percentage', default: 0.2 })
    @IsOptional()
    @IsNumber()
    strikeRangePct?: number;
}

// ============================================================
// RISK MANAGEMENT DTOs
// ============================================================

export class RiskMetricsRequestDto {
    @ApiProperty({ description: 'Returns array' })
    @IsArray()
    @IsNumber({}, { each: true })
    returns: number[];

    @ApiPropertyOptional({ description: 'Confidence level', default: 0.95 })
    @IsOptional()
    @IsNumber()
    @Min(0.9)
    @Max(0.99)
    confidenceLevel?: number;

    @ApiPropertyOptional({ description: 'Time horizon (days)', default: 1 })
    @IsOptional()
    @IsNumber()
    timeHorizon?: number;

    @ApiPropertyOptional({ description: 'Risk-free rate', default: 0 })
    @IsOptional()
    @IsNumber()
    riskFreeRate?: number;
}

export class VaRRequestDto {
    @ApiProperty({ description: 'Returns array' })
    @IsArray()
    @IsNumber({}, { each: true })
    returns: number[];

    @ApiPropertyOptional({ description: 'Confidence level', default: 0.95 })
    @IsOptional()
    @IsNumber()
    confidenceLevel?: number;

    @ApiPropertyOptional({ description: 'Time horizon (days)', default: 1 })
    @IsOptional()
    @IsNumber()
    timeHorizon?: number;

    @ApiPropertyOptional({ description: 'VaR method', default: 'historical', example: 'historical' })
    @IsOptional()
    @IsString()
    method?: string;

    @ApiPropertyOptional({ description: 'Portfolio value' })
    @IsOptional()
    @IsNumber()
    portfolioValue?: number;
}

export class CVaRRequestDto {
    @ApiProperty({ description: 'Returns array' })
    @IsArray()
    @IsNumber({}, { each: true })
    returns: number[];

    @ApiPropertyOptional({ description: 'Confidence level', default: 0.95 })
    @IsOptional()
    @IsNumber()
    confidenceLevel?: number;

    @ApiPropertyOptional({ description: 'Time horizon (days)', default: 1 })
    @IsOptional()
    @IsNumber()
    timeHorizon?: number;

    @ApiPropertyOptional({ description: 'Portfolio value' })
    @IsOptional()
    @IsNumber()
    portfolioValue?: number;
}

export class StressScenarioDto {
    @ApiProperty({ description: 'Scenario name' })
    @IsString()
    name: string;

    @ApiProperty({ description: 'Price shocks by symbol' })
    @IsObject()
    priceShocks: Record<string, number>;

    @ApiPropertyOptional({ description: 'Volatility shock' })
    @IsOptional()
    @IsNumber()
    volatilityShock?: number;

    @ApiPropertyOptional({ description: 'Correlation shock' })
    @IsOptional()
    @IsNumber()
    correlationShock?: number;
}

export class StressTestRequestDto {
    @ApiProperty({ description: 'Symbols array' })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiProperty({ description: 'Positions array' })
    @IsArray()
    @IsNumber({}, { each: true })
    positions: number[];

    @ApiPropertyOptional({ type: [StressScenarioDto], description: 'Stress scenarios' })
    @IsOptional()
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => StressScenarioDto)
    scenarios?: StressScenarioDto[];
}

// ============================================================
// PORTFOLIO OPTIMIZATION DTOs
// ============================================================

export class PortfolioConstraintsDto {
    @ApiPropertyOptional({ description: 'Minimum weight', default: 0 })
    @IsOptional()
    @IsNumber()
    minWeight?: number;

    @ApiPropertyOptional({ description: 'Maximum weight', default: 1})
    @IsOptional()
    @IsNumber()
    maxWeight?: number;

    @ApiPropertyOptional({ description: 'Maximum sector weight' })
    @IsOptional()
    @IsNumber()
    maxSectorWeight?: number;

    @ApiPropertyOptional({ description: 'Sector assignments' })
    @IsOptional()
    @IsArray()
    @IsString({ each: true })
    sectorAssignments?: string[];

    @ApiPropertyOptional({ description: 'Long only constraint', default: true })
    @IsOptional()
    @IsBoolean()
    longOnly?: boolean;
}

export class PortfolioOptRequestDto {
    @ApiProperty({ description: 'Symbols array' })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiPropertyOptional({ description: 'Expected returns array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    expectedReturns?: number[];

    @ApiPropertyOptional({ description: 'Covariance matrix (flattened)' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    covarianceMatrix?: number[];

    @ApiPropertyOptional({ enum: OptimizationMethod, default: OptimizationMethod.MEAN_VARIANCE })
    @IsOptional()
    @IsEnum(OptimizationMethod)
    method?: OptimizationMethod;

    @ApiPropertyOptional({ description: 'Target return' })
    @IsOptional()
    @IsNumber()
    targetReturn?: number;

    @ApiPropertyOptional({ description: 'Risk-free rate', default: 0 })
    @IsOptional()
    @IsNumber()
    riskFreeRate?: number;

    @ApiPropertyOptional({ type: PortfolioConstraintsDto })
    @IsOptional()
    @ValidateNested()
    @Type(() => PortfolioConstraintsDto)
    constraints?: PortfolioConstraintsDto;
}

export class HRPRequestDto {
    @ApiProperty({ description: 'Symbols array' })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiProperty({ description: 'Returns matrix (flattened)' })
    @IsArray()
    @IsNumber({}, { each: true })
    returnsMatrix: number[];

    @ApiProperty({ description: 'Number of periods' })
    @IsNumber()
    nPeriods: number;

    @ApiProperty({ description: 'Number of assets' })
    @IsNumber()
    nAssets: number;

    @ApiPropertyOptional({ description: 'Linkage method', default: 'ward' })
    @IsOptional()
    @IsString()
    linkageMethod?: string;
}

export class ViewInfoDto {
    @ApiProperty({ description: 'Symbols in view' })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiProperty({ description: 'View weights' })
    @IsArray()
    @IsNumber({}, { each: true })
    weights: number[];

    @ApiProperty({ description: 'Expected return' })
    @IsNumber()
    expectedReturn: number;

    @ApiProperty({ description: 'View confidence' })
    @IsNumber()
    @Min(0)
    @Max(1)
    confidence: number;
}

export class BlackLittermanRequestDto {
    @ApiProperty({ description: 'Symbols array' })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiProperty({ description: 'Market capitalizations' })
    @IsArray()
    @IsNumber({}, { each: true })
    marketCaps: number[];

    @ApiProperty({ description: 'Covariance matrix (flattened)' })
    @IsArray()
    @IsNumber({}, { each: true })
    covarianceMatrix: number[];

    @ApiPropertyOptional({ description: 'Risk aversion', default: 2.5 })
    @IsOptional()
    @IsNumber()
    riskAversion?: number;

    @ApiPropertyOptional({ description: 'Tau parameter', default: 0.05 })
    @IsOptional()
    @IsNumber()
    tau?: number;

    @ApiPropertyOptional({ type: [ViewInfoDto], description: 'Investor views' })
    @IsOptional()
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => ViewInfoDto)
    views?: ViewInfoDto[];
}

// ============================================================
// MARKET ANALYSIS DTOs
// ============================================================

export class RegimeDetectionRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Price data array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    prices?: number[];

    @ApiPropertyOptional({ description: 'Volume data array' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    volumes?: number[];

    @ApiPropertyOptional({ description: 'Lookback period', default: 100 })
    @IsOptional()
    @IsNumber()
    @Min(20)
    @Max(500)
    lookbackPeriod?: number;
}

export class SentimentRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Texts to analyze' })
    @IsOptional()
    @IsArray()
    @IsString({ each: true })
    texts?: string[];

    @ApiPropertyOptional({ description: 'Data source', default: 'mixed' })
    @IsOptional()
    @IsString()
    source?: string;
}

export class MicrostructureRequestDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Bid prices' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    bidPrices?: number[];

    @ApiPropertyOptional({ description: 'Ask prices' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    askPrices?: number[];

    @ApiPropertyOptional({ description: 'Bid sizes' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    bidSizes?: number[];

    @ApiPropertyOptional({ description: 'Ask sizes' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    askSizes?: number[];

    @ApiPropertyOptional({ description: 'Trade prices' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    tradePrices?: number[];

    @ApiPropertyOptional({ description: 'Trade sizes' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    tradeSizes?: number[];

    @ApiPropertyOptional({ description: 'Timestamps' })
    @IsOptional()
    @IsArray()
    @IsNumber({}, { each: true })
    timestamps?: number[];
}

// ============================================================
// HEALTH & MONITORING DTOs
// ============================================================

export class MetricsRequestDto {
    @ApiPropertyOptional({ description: 'Start time (Unix timestamp)' })
    @IsOptional()
    @IsNumber()
    startTime?: number;

    @ApiPropertyOptional({ description: 'End time (Unix timestamp)' })
    @IsOptional()
    @IsNumber()
    endTime?: number;
}

// ============================================================
// LEGACY COMPATIBILITY DTOs
// ============================================================

export class GenerateSignalDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Timeframe', default: '1h' })
    @IsOptional()
    @IsString()
    timeframe?: string;

    @ApiPropertyOptional({ description: 'Use ensemble models', default: true })
    @IsOptional()
    @IsBoolean()
    useEnsemble?: boolean;

    @ApiPropertyOptional({ description: 'Use regime detection', default: true })
    @IsOptional()
    @IsBoolean()
    useRegime?: boolean;

    @ApiPropertyOptional({ description: 'Use sentiment analysis' })
    @IsOptional()
    @IsBoolean()
    useSentiment?: boolean;

    @ApiPropertyOptional({ enum: RiskProfile, default: RiskProfile.MODERATE })
    @IsOptional()
    @IsEnum(RiskProfile)
    riskProfile?: RiskProfile;

    @ApiPropertyOptional({ description: 'Specific model ID' })
    @IsOptional()
    @IsString()
    modelId?: string;
}

export class BatchSignalsDto {
    @ApiProperty({ description: 'Array of symbols', example: ['AAPL', 'GOOGL', 'MSFT'] })
    @IsArray()
    @IsString({ each: true })
    symbols: string[];

    @ApiPropertyOptional({ description: 'Timeframe', default: '1h' })
    @IsOptional()
    @IsString()
    timeframe?: string;

    @ApiPropertyOptional({ description: 'Use ensemble models', default: true })
    @IsOptional()
    @IsBoolean()
    useEnsemble?: boolean;
}

export class PortfolioHoldingDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Quantity held' })
    @IsNumber()
    @Min(0)
    quantity: number;

    @ApiPropertyOptional({ description: 'Current price' })
    @IsOptional()
    @IsNumber()
    currentPrice?: number;

    @ApiPropertyOptional({ description: 'Entry price' })
    @IsOptional()
    @IsNumber()
    entryPrice?: number;

    @ApiPropertyOptional({ description: 'Weight (0-1)' })
    @IsOptional()
    @IsNumber()
    @Min(0)
    @Max(1)
    weight?: number;
}

export class PortfolioOptimizeDto {
    @ApiProperty({ type: [PortfolioHoldingDto] })
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => PortfolioHoldingDto)
    holdings: PortfolioHoldingDto[];

    @ApiPropertyOptional({ enum: RiskProfile, default: RiskProfile.MODERATE })
    @IsOptional()
    @IsEnum(RiskProfile)
    riskProfile?: RiskProfile;

    @ApiPropertyOptional({ description: 'Target return' })
    @IsOptional()
    @IsNumber()
    targetReturn?: number;

    @ApiPropertyOptional({ description: 'Use Black-Litterman', default: true })
    @IsOptional()
    @IsBoolean()
    useBlackLitterman?: boolean;

    @ApiPropertyOptional({ description: 'Use HRP', default: false })
    @IsOptional()
    @IsBoolean()
    useHRP?: boolean;
}

export class RiskAssessmentDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ description: 'Position size' })
    @IsNumber()
    @Min(0)
    positionSize: number;

    @ApiProperty({ description: 'Entry price' })
    @IsNumber()
    entryPrice: number;

    @ApiPropertyOptional({ description: 'Current price' })
    @IsOptional()
    @IsNumber()
    currentPrice?: number;

    @ApiPropertyOptional({ description: 'Portfolio value' })
    @IsOptional()
    @IsNumber()
    portfolioValue?: number;
}

export class RAGQueryDto {
    @ApiProperty({ description: 'Query string' })
    @IsString()
    query: string;

    @ApiPropertyOptional({ description: 'Additional context' })
    @IsOptional()
    @IsString()
    context?: string;

    @ApiPropertyOptional({ description: 'Symbols to focus on' })
    @IsOptional()
    @IsArray()
    @IsString({ each: true })
    symbols?: string[];

    @ApiPropertyOptional({ description: 'Max sources', default: 5 })
    @IsOptional()
    @IsNumber()
    @Min(1)
    @Max(20)
    maxSources?: number;

    @ApiPropertyOptional({ description: 'Similarity threshold', default: 0.7 })
    @IsOptional()
    @IsNumber()
    @Min(0)
    @Max(1)
    similarityThreshold?: number;
}

export class PipelineRunDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiPropertyOptional({ description: 'Timeframe', default: '1h' })
    @IsOptional()
    @IsString()
    timeframe?: string;

    @ApiPropertyOptional({ description: 'Use ensemble', default: true })
    @IsOptional()
    @IsBoolean()
    useEnsemble?: boolean;

    @ApiPropertyOptional({ description: 'Use regime detection', default: true })
    @IsOptional()
    @IsBoolean()
    useRegime?: boolean;

    @ApiPropertyOptional({ description: 'Use sentiment' })
    @IsOptional()
    @IsBoolean()
    useSentiment?: boolean;

    @ApiPropertyOptional({ description: 'Include prediction', default: true })
    @IsOptional()
    @IsBoolean()
    includePrediction?: boolean;

    @ApiPropertyOptional({ description: 'Include optimization' })
    @IsOptional()
    @IsBoolean()
    includeOptimization?: boolean;

    @ApiPropertyOptional({ enum: RiskProfile, default: RiskProfile.MODERATE })
    @IsOptional()
    @IsEnum(RiskProfile)
    riskProfile?: RiskProfile;
}