import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { EventEmitter2, OnEvent } from '@nestjs/event-emitter';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import Decimal from 'decimal.js';

/**
 * Circuit Breaker Types
 */
export enum CircuitBreakerType {
    DAILY_LOSS = 'DAILY_LOSS',
    DRAWDOWN = 'DRAWDOWN',
    VOLATILITY = 'VOLATILITY',
    ERROR_RATE = 'ERROR_RATE',
    MANUAL = 'MANUAL',
}

export enum CircuitState {
    CLOSED = 'CLOSED',       // Normal operation
    OPEN = 'OPEN',           // Trading blocked
    HALF_OPEN = 'HALF_OPEN', // Testing recovery
}

export interface CircuitStatus {
    state: CircuitState;
    type?: CircuitBreakerType;
    triggeredAt?: Date;
    reason?: string;
    recoveryAt?: Date;
    metrics: {
        dailyLoss: string;
        dailyLossPercent: string;
        currentDrawdown: string;
        volatility?: string;
    };
}

/**
 * Circuit Breaker Service
 * 
 * Protects portfolio from excessive losses:
 * - Daily loss limit
 * - Drawdown protection
 * - Volatility spike detection
 * - Cooldown periods
 */
@Injectable()
export class CircuitBreakerService implements OnModuleInit {
    private readonly logger = new Logger(CircuitBreakerService.name);

    // In-memory circuit states (backed by Redis)
    private circuitStates: Map<number, CircuitStatus> = new Map();

    constructor(
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
        private readonly eventEmitter: EventEmitter2,
    ) { }

    async onModuleInit() {
        this.logger.log('⚡ Circuit Breaker Service initialized');
    }

    // ==========================================
    // CIRCUIT STATE MANAGEMENT
    // ==========================================

    /**
     * Check if trading is allowed for a user
     */
    async canTrade(userId: number, portfolioId: number): Promise<{ allowed: boolean; reason?: string }> {
        const status = await this.getCircuitStatus(userId, portfolioId);

        if (status.state === CircuitState.OPEN) {
            return {
                allowed: false,
                reason: `Circuit breaker active: ${status.reason}. Recovery at ${status.recoveryAt?.toISOString() || 'manual reset required'}`,
            };
        }

        if (status.state === CircuitState.HALF_OPEN) {
            return {
                allowed: true,
                reason: 'Circuit in recovery mode - trading with caution',
            };
        }

        return { allowed: true };
    }

    /**
     * Get circuit status for a user/portfolio
     */
    async getCircuitStatus(userId: number, portfolioId: number): Promise<CircuitStatus> {
        const cacheKey = `circuit:${userId}:${portfolioId}`;
        const cached = await this.redis.get<CircuitStatus>(cacheKey);

        if (cached) {
            // Check if recovery time has passed
            if (cached.state === CircuitState.OPEN && cached.recoveryAt) {
                if (new Date() > new Date(cached.recoveryAt)) {
                    await this.transitionToHalfOpen(userId, portfolioId);
                    return this.getCircuitStatus(userId, portfolioId);
                }
            }
            return cached;
        }

        // Calculate fresh metrics
        const metrics = await this.calculateMetrics(userId, portfolioId);

        return {
            state: CircuitState.CLOSED,
            metrics,
        };
    }

    /**
     * Trip the circuit breaker
     */
    async tripCircuit(
        userId: number,
        portfolioId: number,
        type: CircuitBreakerType,
        reason: string,
        cooldownMinutes: number = 60,
    ): Promise<void> {
        const cacheKey = `circuit:${userId}:${portfolioId}`;
        const metrics = await this.calculateMetrics(userId, portfolioId);

        const status: CircuitStatus = {
            state: CircuitState.OPEN,
            type,
            triggeredAt: new Date(),
            reason,
            recoveryAt: new Date(Date.now() + cooldownMinutes * 60 * 1000),
            metrics,
        };

        await this.redis.set(cacheKey, status, cooldownMinutes * 60 + 3600); // Extra hour for safety
        this.circuitStates.set(portfolioId, status);

        this.logger.warn(`🔴 Circuit OPEN: User ${userId}, Portfolio ${portfolioId} - ${reason}`);

        // Emit event for notifications
        this.eventEmitter.emit('circuit.tripped', {
            userId,
            portfolioId,
            type,
            reason,
            recoveryAt: status.recoveryAt,
        });
    }

    /**
     * Transition to half-open state
     */
    private async transitionToHalfOpen(userId: number, portfolioId: number): Promise<void> {
        const cacheKey = `circuit:${userId}:${portfolioId}`;
        const metrics = await this.calculateMetrics(userId, portfolioId);

        const status: CircuitStatus = {
            state: CircuitState.HALF_OPEN,
            metrics,
        };

        await this.redis.set(cacheKey, status, 3600); // 1 hour
        this.logger.log(`🟡 Circuit HALF-OPEN: User ${userId}, Portfolio ${portfolioId}`);
    }

    /**
     * Reset circuit to closed state
     */
    async resetCircuit(userId: number, portfolioId: number): Promise<void> {
        const cacheKey = `circuit:${userId}:${portfolioId}`;
        await this.redis.delete(cacheKey);
        this.circuitStates.delete(portfolioId);

        this.logger.log(`🟢 Circuit CLOSED: User ${userId}, Portfolio ${portfolioId}`);

        this.eventEmitter.emit('circuit.reset', { userId, portfolioId });
    }

    // ==========================================
    // DAILY LOSS CHECK
    // ==========================================

    /**
     * Check daily loss limit and trip if exceeded
     */
    async checkDailyLoss(
        userId: number,
        portfolioId: number,
        maxDailyLossPercent: number,
    ): Promise<boolean> {
        const metrics = await this.calculateMetrics(userId, portfolioId);
        const dailyLossPercent = new Decimal(metrics.dailyLossPercent);

        if (dailyLossPercent.abs().gte(maxDailyLossPercent)) {
            await this.tripCircuit(
                userId,
                portfolioId,
                CircuitBreakerType.DAILY_LOSS,
                `Daily loss ${dailyLossPercent.toFixed(2)}% exceeds ${maxDailyLossPercent}% limit`,
                60, // 1 hour cooldown
            );
            return false; // Trading not allowed
        }

        return true; // Trading allowed
    }

    /**
     * Update daily PnL after a trade
     */
    async updateDailyPnL(portfolioId: number, pnlChange: string): Promise<void> {
        const today = new Date().toISOString().split('T')[0];
        const cacheKey = `daily_pnl:${portfolioId}:${today}`;

        const current = await this.redis.get<string>(cacheKey) || '0';
        const updated = new Decimal(current).plus(pnlChange);

        await this.redis.set(cacheKey, updated.toString(), 86400); // 24 hours
    }

    // ==========================================
    // DRAWDOWN CHECK
    // ==========================================

    /**
     * Check drawdown limit
     */
    async checkDrawdown(
        userId: number,
        portfolioId: number,
        maxDrawdownPercent: number,
    ): Promise<boolean> {
        const metrics = await this.calculateMetrics(userId, portfolioId);
        const currentDrawdown = new Decimal(metrics.currentDrawdown);

        if (currentDrawdown.abs().gte(maxDrawdownPercent)) {
            await this.tripCircuit(
                userId,
                portfolioId,
                CircuitBreakerType.DRAWDOWN,
                `Drawdown ${currentDrawdown.toFixed(2)}% exceeds ${maxDrawdownPercent}% limit`,
                240, // 4 hour cooldown
            );
            return false;
        }

        return true;
    }

    /**
     * Update peak portfolio value for drawdown calculation
     */
    async updatePeakValue(portfolioId: number, currentValue: string): Promise<void> {
        const cacheKey = `peak_value:${portfolioId}`;
        const peak = await this.redis.get<string>(cacheKey);

        if (!peak || new Decimal(currentValue).gt(peak)) {
            await this.redis.set(cacheKey, currentValue, 86400 * 30); // 30 days
        }
    }

    // ==========================================
    // VOLATILITY CHECK
    // ==========================================

    /**
     * Check for volatility spike
     */
    async checkVolatility(
        userId: number,
        portfolioId: number,
        currentVolatility: number,
        maxVolatility: number = 5, // 5% = extreme
    ): Promise<boolean> {
        if (currentVolatility >= maxVolatility) {
            await this.tripCircuit(
                userId,
                portfolioId,
                CircuitBreakerType.VOLATILITY,
                `Market volatility ${currentVolatility.toFixed(2)}% exceeds safe threshold`,
                30, // 30 minute cooldown
            );
            return false;
        }

        return true;
    }

    // ==========================================
    // METRICS CALCULATION
    // ==========================================

    /**
     * Calculate current risk metrics
     */
    private async calculateMetrics(_userId: number, portfolioId: number): Promise<CircuitStatus['metrics']> {
        const today = new Date().toISOString().split('T')[0];

        // Get daily PnL
        const dailyPnLKey = `daily_pnl:${portfolioId}:${today}`;
        const dailyPnL = await this.redis.get<string>(dailyPnLKey) || '0';

        // Get portfolio value from totalValue field
        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
        });

        const portfolioValue = portfolio?.totalValue
            ? new Decimal(portfolio.totalValue.toString())
            : new Decimal(0);

        // Get peak value for drawdown
        const peakKey = `peak_value:${portfolioId}`;
        const peakValue = await this.redis.get<string>(peakKey) || portfolioValue.toString();

        const drawdown = portfolioValue.gt(0)
            ? new Decimal(peakValue).minus(portfolioValue).div(peakValue).mul(100)
            : new Decimal(0);

        const dailyLossPercent = portfolioValue.gt(0)
            ? new Decimal(dailyPnL).div(portfolioValue).mul(100)
            : new Decimal(0);

        return {
            dailyLoss: dailyPnL,
            dailyLossPercent: dailyLossPercent.toFixed(2),
            currentDrawdown: drawdown.toFixed(2),
        };
    }

    // ==========================================
    // EVENT HANDLERS
    // ==========================================

    /**
     * Handle order filled event - check circuit breakers
     */
    @OnEvent('order.filled')
    async handleOrderFilled(payload: { userId: number; portfolioId: number; pnl: string }): Promise<void> {
        await this.updateDailyPnL(payload.portfolioId, payload.pnl);

        // Check if we need to trip circuit
        // Using default limits - should come from user's risk profile
        await this.checkDailyLoss(payload.userId, payload.portfolioId, 5);
        await this.checkDrawdown(payload.userId, payload.portfolioId, 15);
    }
}
