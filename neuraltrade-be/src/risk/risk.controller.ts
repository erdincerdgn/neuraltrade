import { Controller, Get, Post, Body, Param, UseGuards } from '@nestjs/common';
import { RiskEngineService } from './risk-engine.service';
import { CircuitBreakerService } from './circuit-breaker.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
    PositionSizeRequest,
    RiskTolerance,
    DEFAULT_RISK_PROFILES,
    ExposureCheck,
} from './risk.types';

/**
 * Risk Controller
 * 
 * API endpoints for risk management operations.
 */
@Controller('api/v1/risk')
@UseGuards(JwtAuthGuard)
export class RiskController {
    constructor(
        private readonly riskEngine: RiskEngineService,
        private readonly circuitBreaker: CircuitBreakerService,
    ) { }

    // ==========================================
    // POSITION SIZING
    // ==========================================

    @Post('position-size')
    async calculatePositionSize(@Body() request: PositionSizeRequest) {
        // Use moderate profile as default
        const profile = DEFAULT_RISK_PROFILES[RiskTolerance.MODERATE];
        const result = this.riskEngine.calculatePositionSize(request, profile);
        return result;
    }

    @Post('position-size/:tolerance')
    async calculatePositionSizeWithProfile(
        @Param('tolerance') tolerance: string,
        @Body() request: PositionSizeRequest,
    ) {
        const riskTolerance = tolerance.toUpperCase() as RiskTolerance;
        const profile = DEFAULT_RISK_PROFILES[riskTolerance] || DEFAULT_RISK_PROFILES[RiskTolerance.MODERATE];
        const result = this.riskEngine.calculatePositionSize(request, profile);
        return result;
    }

    // ==========================================
    // EXPOSURE
    // ==========================================

    @Post('exposure-check/:userId/:portfolioId')
    async checkExposure(
        @Param('userId') userId: string,
        @Param('portfolioId') portfolioId: string,
        @Body() check: ExposureCheck,
    ) {
        const profile = DEFAULT_RISK_PROFILES[RiskTolerance.MODERATE];
        return this.riskEngine.checkExposure(
            parseInt(userId),
            parseInt(portfolioId),
            check,
            profile,
        );
    }

    // ==========================================
    // RISK METRICS
    // ==========================================

    @Get('metrics/:userId/:portfolioId')
    async getRiskMetrics(
        @Param('userId') userId: string,
        @Param('portfolioId') portfolioId: string,
    ) {
        return this.riskEngine.getPortfolioRiskMetrics(
            parseInt(userId),
            parseInt(portfolioId),
        );
    }

    // ==========================================
    // CIRCUIT BREAKER
    // ==========================================

    @Get('circuit/:userId/:portfolioId')
    async getCircuitStatus(
        @Param('userId') userId: string,
        @Param('portfolioId') portfolioId: string,
    ) {
        return this.circuitBreaker.getCircuitStatus(
            parseInt(userId),
            parseInt(portfolioId),
        );
    }

    @Post('circuit/:userId/:portfolioId/reset')
    async resetCircuit(
        @Param('userId') userId: string,
        @Param('portfolioId') portfolioId: string,
    ) {
        await this.circuitBreaker.resetCircuit(
            parseInt(userId),
            parseInt(portfolioId),
        );
        return { message: 'Circuit breaker reset', state: 'CLOSED' };
    }

    @Get('can-trade/:userId/:portfolioId')
    async canTrade(
        @Param('userId') userId: string,
        @Param('portfolioId') portfolioId: string,
    ) {
        return this.circuitBreaker.canTrade(
            parseInt(userId),
            parseInt(portfolioId),
        );
    }

    // ==========================================
    // PROFILES
    // ==========================================

    @Get('profiles')
    getAvailableProfiles() {
        return {
            profiles: Object.keys(DEFAULT_RISK_PROFILES),
            details: DEFAULT_RISK_PROFILES,
        };
    }

    @Get('profile/:userId')
    async getUserProfile(@Param('userId') userId: string) {
        return this.riskEngine.getUserRiskProfile(parseInt(userId));
    }

    @Post('profile/:userId')
    async setUserProfile(
        @Param('userId') userId: string,
        @Body() body: { tolerance: RiskTolerance },
    ) {
        const profile = DEFAULT_RISK_PROFILES[body.tolerance];
        await this.riskEngine.setUserRiskProfile(parseInt(userId), profile);
        return { message: 'Profile updated', profile };
    }
}
