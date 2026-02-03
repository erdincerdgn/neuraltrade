import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { RedisService } from '../core/redis/redis.service';
import { PrismaService } from '../core/prisma/prisma.service';
import { AISignalResponse } from './ai-proxy.service';
import { SignalAction } from './grpc-client.service';

export interface SignalValidation {
    isValid: boolean;
    reason?: string;
    warnings?: string[];
}

@Injectable()
export class SignalProcessorService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(SignalProcessorService.name);
    private isSubscribed = false;

    private readonly MIN_CONFIDENCE = 0.6;
    private readonly MAX_RISK_REWARD_RATIO = 5.0;
    private readonly MIN_RISK_REWARD_RATIO = 0.5;
    private readonly SIGNAL_CHANNEL = 'ai:signals:v2';
    private readonly SIGNAL_HISTORY_CHANNEL = 'ai:signals:history';

    private signalsProcessed = 0;
    private signalsRejected = 0;
    private lastSignalTime: Date | null = null;

    constructor(
        private readonly redis: RedisService,
        private readonly eventEmitter: EventEmitter2,
        private readonly prisma: PrismaService,
    ) {}

    async onModuleInit(): Promise<void> {
        await this.subscribeToSignals();
        this.logger.log('🚀 Signal Processor Service v2.0 initialized');
    }

    async onModuleDestroy(): Promise<void> {
        await this.unsubscribe();
    }

    async subscribeToSignals(): Promise<void> {
        if (this.isSubscribed) return;

        try {
            await this.redis.subscribe(this.SIGNAL_CHANNEL, async (message: string) => {
                await this.handleIncomingSignal(message);
            });

            this.isSubscribed = true;
            this.logger.log(`📡 Subscribed to Redis channel: ${this.SIGNAL_CHANNEL}`);
        } catch (error) {
            this.logger.error(`Failed to subscribe to signals: ${error.message}`);
        }
    }

    async unsubscribe(): Promise<void> {
        if (!this.isSubscribed) return;

        try {
            await this.redis.unsubscribe(this.SIGNAL_CHANNEL);
            this.isSubscribed = false;
            this.logger.log(`👋 Unsubscribed from ${this.SIGNAL_CHANNEL}`);
        } catch (error) {
            this.logger.error(`Failed to unsubscribe: ${error.message}`);
        }
    }

    private async handleIncomingSignal(message: string): Promise<void> {
        try {
            const signal = JSON.parse(message) as AISignalResponse;

            this.logger.log(
                `📨 Received signal: ${signal.symbol} - ${this.getActionString(signal.action)} ` +
                `(${(signal.confidence * 100).toFixed(1)}%) [${signal.modelUsed || 'unknown'}]`
            );

            const validation = this.validateSignal(signal);
            if (!validation.isValid) {
                this.signalsRejected++;
                this.logger.warn(`⚠️ Signal rejected: ${validation.reason}`);
                this.eventEmitter.emit('ai.signal.rejected', { signal, reason: validation.reason });
                return;
            }

            if (validation.warnings?.length) {
                validation.warnings.forEach(w => this.logger.warn(`⚠️ ${w}`));
            }

            await this.processSignal(signal);
        } catch (error) {
            this.logger.error(`Failed to parse signal: ${error.message}`);
        }
    }

    validateSignal(signal: AISignalResponse): SignalValidation {
        const warnings: string[] = [];

        if (!signal.symbol || signal.action === undefined) {
            return { isValid: false, reason: 'Missing required fields (symbol or action)' };
        }

        const validActions = ['BUY', 'SELL', 'HOLD', 'CLOSE', 1, 2, 3, 4];
        if (!validActions.includes(signal.action as any)) {
            return { isValid: false, reason: `Invalid action: ${signal.action}` };
        }

        if (signal.confidence < this.MIN_CONFIDENCE) {
            return {
                isValid: false,
                reason: `Confidence ${(signal.confidence * 100).toFixed(1)}% below threshold ${this.MIN_CONFIDENCE * 100}%`,
            };
        }

        if (signal.riskRewardRatio !== undefined) {
            if (signal.riskRewardRatio < this.MIN_RISK_REWARD_RATIO) {
                return {
                    isValid: false,
                    reason: `Risk/Reward ratio ${signal.riskRewardRatio.toFixed(2)} below minimum ${this.MIN_RISK_REWARD_RATIO}`,
                };
            }if (signal.riskRewardRatio > this.MAX_RISK_REWARD_RATIO) {
                warnings.push(`High risk/reward ratio: ${signal.riskRewardRatio.toFixed(2)}`);
            }
        }

        if (signal.targetPrice && signal.stopLoss) {
            const actionStr = this.getActionString(signal.action);
            if (actionStr === 'BUY' && signal.stopLoss >= signal.targetPrice) {
                return { isValid: false, reason: 'Stop loss must be below target price for BUY signals' };
            }
            if (actionStr === 'SELL' && signal.stopLoss <= signal.targetPrice) {
                return { isValid: false, reason: 'Stop loss must be above target price for SELL signals' };
            }
        }

        if (signal.volatility && signal.volatility > 0.05) {
            warnings.push(`High volatility detected: ${(signal.volatility * 100).toFixed(2)}%`);
        }

        if (signal.regime === 'CRISIS') {
            warnings.push('Signal generated during CRISIS regime - exercise caution');
        }

        return { isValid: true, warnings };
    }

    private async processSignal(signal: AISignalResponse): Promise<void> {
        this.signalsProcessed++;
        this.lastSignalTime = new Date();

        this.eventEmitter.emit('ai.signal.received', signal);
        this.eventEmitter.emit('ai.signal.broadcast', signal);

        const actionStr = this.getActionString(signal.action);
        if (actionStr !== 'HOLD') {
            this.eventEmitter.emit('ai.signal.actionable', signal);
            this.eventEmitter.emit(`ai.signal.${actionStr.toLowerCase()}`, signal);
        }

        this.persistSignal(signal).catch(err =>
            this.logger.error(`Failed to persist signal: ${err.message}`)
        );

        await this.redis.publish(this.SIGNAL_HISTORY_CHANNEL, JSON.stringify({
            ...signal,
            processedAt: new Date().toISOString(),
        }));

        this.logger.log(`✅ Signal processed: ${signal.symbol} ${actionStr} (${this.signalsProcessed} total)`);
    }

    private async persistSignal(signal: AISignalResponse): Promise<void> {
        try {
            await this.prisma.signalHistory.create({
                data: {
                    symbol: signal.symbol,
                    action: this.getActionString(signal.action),
                    confidence: signal.confidence,
                    targetPrice: signal.targetPrice,
                    stopLoss: signal.stopLoss,
                    takeProfit: signal.takeProfit,
                    reasoning: signal.reasoning,
                    modelUsed: signal.modelUsed,
                    regime: signal.regime?.toString(),
                    volatility: signal.volatility,
                    riskRewardRatio: signal.riskRewardRatio,
                    expectedReturn: signal.expectedReturn,
                    maxDrawdownEstimate: signal.maxDrawdownEstimate,
                    contributors: signal.contributors as any,
                    metadata: signal.metadata ? signal.metadata : undefined,
                    timestamp: signal.timestamp ? new Date(signal.timestamp) : new Date(),
                },
            });

            this.logger.debug(`Signal persisted to database: ${signal.symbol}`);
        } catch (error) {
            this.logger.error(`Database persistence failed: ${error.message}`);
        }
    }

    private getActionString(action: SignalAction | string): string {
        if (typeof action === 'string') return action;

        switch (action) {
            case SignalAction.BUY:
            case 1:
                return 'BUY';
            case SignalAction.SELL:
            case 2:
                return 'SELL';
            case SignalAction.HOLD:
            case 3:
                return 'HOLD';
            case SignalAction.CLOSE:
            case 4:
                return 'CLOSE';
            default:
                return 'UNKNOWN';
        }
    }

    async injectTestSignal(signal: Partial<AISignalResponse>): Promise<void> {
        const testSignal: AISignalResponse = {
            symbol: signal.symbol || 'BTC/USDT',
            action: signal.action || SignalAction.BUY,
            confidence: signal.confidence || 0.85,
            targetPrice: signal.targetPrice,
            stopLoss: signal.stopLoss,
            takeProfit: signal.takeProfit,
            reasoning: signal.reasoning || 'Test signal injection',
            modelUsed: signal.modelUsed || 'test_model',
            regime: signal.regime || 'RANGING',
            volatility: signal.volatility || 0.02,
            riskRewardRatio: signal.riskRewardRatio || 2.0,
            timestamp: new Date().toISOString(),
            metadata: { ...signal.metadata, isTest: true },
        };

        await this.processSignal(testSignal);
    }

    async publishSignal(signal: AISignalResponse): Promise<void> {
        await this.redis.publish(this.SIGNAL_CHANNEL, JSON.stringify(signal));
    }

    isActive(): boolean {
        return this.isSubscribed;
    }

    getMinConfidence(): number {
        return this.MIN_CONFIDENCE;
    }

    getMetrics(): {
        signalsProcessed: number;
        signalsRejected: number;
        lastSignalTime: Date | null;
        isActive: boolean;
    } {
        return {
            signalsProcessed: this.signalsProcessed,
            signalsRejected: this.signalsRejected,
            lastSignalTime: this.lastSignalTime,
            isActive: this.isSubscribed,
        };
    }

    resetMetrics(): void {
        this.signalsProcessed = 0;
        this.signalsRejected = 0;
        this.lastSignalTime = null;
    }
}