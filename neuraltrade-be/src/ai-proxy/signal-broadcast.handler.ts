import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { EventsGateway, AISignalBroadcast, SignalAlertPayload } from '../websocket/events.gateway';
import { AISignalResponse } from './ai-proxy.service';
import { SignalAction } from './grpc-client.service';

export interface WebSocketSignalPayload {
    symbol: string;
    action: string;
    confidence: number;
    targetPrice?: number;
    stopLoss?: number;
    takeProfit?: number;
    reasoning: string;
    modelUsed?: string;
    regime?: string;volatility?: number;
    riskRewardRatio?: number;
    expectedReturn?: number;
    timestamp: string;
}

@Injectable()
export class SignalBroadcastHandler implements OnModuleInit {
    private readonly logger = new Logger(SignalBroadcastHandler.name);

    private broadcastCount = 0;
    private alertCount = 0;

    constructor(private readonly eventsGateway: EventsGateway) {}

    async onModuleInit(): Promise<void> {
        this.logger.log('📡 Signal Broadcast Handler v2.0 initialized');
    }

    @OnEvent('ai.signal.broadcast')
    async handleSignalBroadcast(signal: AISignalResponse): Promise<void> {
        try {
            const payload = this.formatSignalPayload(signal);
            const broadcastPayload: AISignalBroadcast = {
                symbol: signal.symbol,
                action: this.getActionString(signal.action),
                confidence: signal.confidence,
                reasoning: signal.reasoning,
                targetPrice: signal.targetPrice,
                stopLoss: signal.stopLoss,
                modelUsed: signal.modelUsed,
                regime: signal.regime,
            };

            this.eventsGateway.broadcastSignal(broadcastPayload);
            this.eventsGateway.broadcastToRoom(`signal:${signal.symbol}`, 'signal:update', payload);

            this.broadcastCount++;
            this.logger.debug(`📡 Signal broadcast: ${signal.symbol} ${this.getActionString(signal.action)} (${this.broadcastCount} total)`);
        } catch (error) {
            this.logger.error(`Failed to broadcast signal: ${error.message}`);
        }
    }

    @OnEvent('ai.signal.actionable')
    async handleActionableSignal(signal: AISignalResponse): Promise<void> {
        try {
            const actionStr = this.getActionString(signal.action);
            const priority = this.calculateAlertPriority(signal);

            const alert: SignalAlertPayload = {
                type: actionStr,
                symbol: signal.symbol,
                action: actionStr,
                confidence: signal.confidence,
                message: this.formatAlertMessage(signal),
                priority,
            };

            this.eventsGateway.broadcastAlert(alert);

            if (priority === 'CRITICAL' || priority === 'HIGH') {
                this.eventsGateway.broadcastToRoom('alerts:high-priority', 'alert:critical', alert);
            }

            this.alertCount++;
            this.logger.log(
                `🎯 Actionable signal: ${signal.symbol} ${actionStr} ` +
                `(${(signal.confidence * 100).toFixed(1)}%) [${priority}]`
            );
        } catch (error) {
            this.logger.error(`Failed to handle actionable signal: ${error.message}`);
        }
    }

    @OnEvent('ai.signal.rejected')
    async handleRejectedSignal(data: { signal: AISignalResponse; reason: string }): Promise<void> {
        try {
            this.eventsGateway.broadcastToRoom('admin:signals', 'signal:rejected', {
                signal: this.formatSignalPayload(data.signal),
                reason: data.reason,
            });

            this.logger.debug(`📡 Rejected signal broadcast: ${data.signal.symbol} - ${data.reason}`);
        } catch (error) {
            this.logger.error(`Failed to broadcast rejected signal: ${error.message}`);
        }
    }

    @OnEvent('ai.signal.buy')
    async handleBuySignal(signal: AISignalResponse): Promise<void> {
        this.eventsGateway.broadcastToRoom('signals:buy', 'signal:buy', this.formatSignalPayload(signal));
        this.logger.debug(`📈 BUY signal broadcast: ${signal.symbol}`);
    }

    @OnEvent('ai.signal.sell')
    async handleSellSignal(signal: AISignalResponse): Promise<void> {
        this.eventsGateway.broadcastToRoom('signals:sell', 'signal:sell', this.formatSignalPayload(signal));
        this.logger.debug(`📉 SELL signal broadcast: ${signal.symbol}`);
    }

    @OnEvent('ai.signal.close')
    async handleCloseSignal(signal: AISignalResponse): Promise<void> {
        this.eventsGateway.broadcastToRoom('signals:close', 'signal:close', this.formatSignalPayload(signal));
        this.logger.debug(`🔒 CLOSE signal broadcast: ${signal.symbol}`);
    }

    private formatSignalPayload(signal: AISignalResponse): WebSocketSignalPayload {
        return {
            symbol: signal.symbol,
            action: this.getActionString(signal.action),
            confidence: signal.confidence,
            targetPrice: signal.targetPrice,
            stopLoss: signal.stopLoss,
            takeProfit: signal.takeProfit,
            reasoning: signal.reasoning,
            modelUsed: signal.modelUsed,
            regime: signal.regime,
            volatility: signal.volatility,
            riskRewardRatio: signal.riskRewardRatio,
            expectedReturn: signal.expectedReturn,
            timestamp: signal.timestamp || new Date().toISOString(),
        };
    }

    private formatAlertMessage(signal: AISignalResponse): string {
        const actionStr = this.getActionString(signal.action);
        const confidencePct = (signal.confidence * 100).toFixed(1);

        let message = `${actionStr} ${signal.symbol} with ${confidencePct}% confidence`;

        if (signal.targetPrice) {
            message += ` | Target: $${signal.targetPrice.toFixed(2)}`;
        }
        if (signal.stopLoss) {
            message += ` | SL: $${signal.stopLoss.toFixed(2)}`;
        }
        if (signal.riskRewardRatio) {
            message += ` | R:R ${signal.riskRewardRatio.toFixed(2)}`;
        }

        return message;
    }

    private calculateAlertPriority(signal: AISignalResponse): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
        if (signal.regime === 'CRISIS') {
            return 'CRITICAL';
        }

        if (signal.confidence >= 0.9&& (signal.expectedReturn ||0) > 0.05) {
            return 'HIGH';
        }

        if (signal.confidence >= 0.85) {
            return 'HIGH';
        }

        if (signal.confidence >= 0.75) {
            return 'MEDIUM';
        }

        return 'LOW';
    }

    private getActionString(action: SignalAction | string): string {
        if (typeof action === 'string') return action;

        switch (action) {
            case SignalAction.BUY:case 1:
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

    getMetrics(): { broadcastCount: number; alertCount: number } {
        return {
            broadcastCount: this.broadcastCount,
            alertCount: this.alertCount,
        };
    }

    resetMetrics(): void {
        this.broadcastCount = 0;
        this.alertCount = 0;
    }
}