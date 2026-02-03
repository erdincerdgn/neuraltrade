import { Injectable, Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { PrismaService } from '../core/prisma/prisma.service';

/**
 * Audit Action Types
 */
export enum AuditAction {
    // Order lifecycle
    ORDER_PLACED = 'ORDER_PLACED',
    ORDER_SUBMITTED = 'ORDER_SUBMITTED',
    ORDER_FILLED = 'ORDER_FILLED',
    ORDER_PARTIAL = 'ORDER_PARTIAL',
    ORDER_CANCELLED = 'ORDER_CANCELLED',
    ORDER_REJECTED = 'ORDER_REJECTED',

    // Risk events
    CIRCUIT_TRIPPED = 'CIRCUIT_TRIPPED',
    CIRCUIT_RESET = 'CIRCUIT_RESET',
    RISK_LIMIT_BREACH = 'RISK_LIMIT_BREACH',

    // AI signals
    SIGNAL_RECEIVED = 'SIGNAL_RECEIVED',
    SIGNAL_EXECUTED = 'SIGNAL_EXECUTED',
    SIGNAL_REJECTED = 'SIGNAL_REJECTED',

    // Account events
    BALANCE_LOCKED = 'BALANCE_LOCKED',
    BALANCE_UNLOCKED = 'BALANCE_UNLOCKED',
    DEPOSIT = 'DEPOSIT',
    WITHDRAWAL = 'WITHDRAWAL',

    // System events
    SYSTEM_ERROR = 'SYSTEM_ERROR',
    CONFIG_CHANGED = 'CONFIG_CHANGED',
}

export interface AuditEntry {
    userId: number;
    portfolioId?: number;
    action: AuditAction;
    details: Record<string, any>;
    aiReasoning?: string;
    ipAddress?: string;
    userAgent?: string;
}

export interface AuditLogRecord {
    id: number;
    timestamp: Date;
    userId: number;
    portfolioId?: number;
    action: string;
    details: any;
    aiReasoning?: string;
    ipAddress?: string;
}

/**
 * Audit Log Service
 * 
 * Immutable audit trail for:
 * - All trading decisions
 * - AI signal processing
 * - Risk management events
 * - Regulatory compliance
 */
@Injectable()
export class AuditLogService {
    private readonly logger = new Logger(AuditLogService.name);

    constructor(private readonly prisma: PrismaService) { }

    // ==========================================
    // CORE LOGGING
    // ==========================================

    /**
     * Log an audit entry (immutable)
     */
    async log(entry: AuditEntry): Promise<number> {
        try {
            // Using raw SQL for audit_log table (not in Prisma schema yet)
            const result = await this.prisma.$executeRaw`
                INSERT INTO audit_log (timestamp, user_id, portfolio_id, action, details, ai_reasoning, ip_address)
                VALUES (
                    NOW(),
                    ${entry.userId},
                    ${entry.portfolioId || null},
                    ${entry.action},
                    ${JSON.stringify(entry.details)}::jsonb,
                    ${entry.aiReasoning || null},
                    ${entry.ipAddress || null}
                )
            `;

            this.logger.debug(`Audit: ${entry.action} for user ${entry.userId}`);
            return result;
        } catch (error) {
            // Fallback to console logging if table doesn't exist yet
            this.logger.warn(`Audit (fallback): ${entry.action} - ${JSON.stringify(entry.details)}`);
            return 0;
        }
    }

    /**
     * Log order event
     */
    async logOrder(
        userId: number,
        portfolioId: number,
        action: AuditAction,
        orderId: string,
        details: {
            symbol: string;
            side: string;
            quantity: string;
            price?: string;
            status?: string;
            filledQuantity?: string;
            pnl?: string;
        },
        aiReasoning?: string,
    ): Promise<void> {
        await this.log({
            userId,
            portfolioId,
            action,
            details: {
                orderId,
                ...details,
                timestamp: new Date().toISOString(),
            },
            aiReasoning,
        });
    }

    /**
     * Log risk event
     */
    async logRiskEvent(
        userId: number,
        portfolioId: number,
        action: AuditAction,
        details: {
            type: string;
            threshold?: string;
            actual?: string;
            reason?: string;
        },
    ): Promise<void> {
        await this.log({
            userId,
            portfolioId,
            action,
            details: {
                ...details,
                timestamp: new Date().toISOString(),
            },
        });
    }

    /**
     * Log AI signal
     */
    async logSignal(
        userId: number,
        action: AuditAction,
        signal: {
            symbol: string;
            signalAction: string;
            confidence: number;
            models?: string[];
        },
        aiReasoning?: string,
    ): Promise<void> {
        await this.log({
            userId,
            action,
            details: {
                ...signal,
                timestamp: new Date().toISOString(),
            },
            aiReasoning,
        });
    }

    // ==========================================
    // QUERY / EXPORT
    // ==========================================

    /**
     * Get audit history for a user
     */
    async getUserAuditHistory(
        userId: number,
        options?: {
            portfolioId?: number;
            action?: AuditAction;
            startDate?: Date;
            endDate?: Date;
            limit?: number;
        },
    ): Promise<AuditLogRecord[]> {
        const limit = options?.limit || 100;
        const startDate = options?.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
        const endDate = options?.endDate || new Date();

        try {
            const logs = await this.prisma.$queryRaw<AuditLogRecord[]>`
                SELECT id, timestamp, user_id, portfolio_id, action, details, ai_reasoning, ip_address
                FROM audit_log
                WHERE user_id = ${userId}
                    AND timestamp BETWEEN ${startDate} AND ${endDate}
                    ${options?.portfolioId ? this.prisma.$queryRaw`AND portfolio_id = ${options.portfolioId}` : this.prisma.$queryRaw``}
                    ${options?.action ? this.prisma.$queryRaw`AND action = ${options.action}` : this.prisma.$queryRaw``}
                ORDER BY timestamp DESC
                LIMIT ${limit}
            `;
            return logs;
        } catch (error) {
            this.logger.warn('Audit table query failed (table may not exist yet)');
            return [];
        }
    }

    /**
     * Export audit logs for regulatory reporting
     */
    async exportForCompliance(
        userId: number,
        startDate: Date,
        endDate: Date,
    ): Promise<{
        user: { id: number };
        period: { start: string; end: string };
        totalTrades: number;
        logs: AuditLogRecord[];
    }> {
        const logs = await this.getUserAuditHistory(userId, {
            startDate,
            endDate,
            limit: 10000,
        });

        const tradeLogs = logs.filter(l =>
            l.action.startsWith('ORDER_') &&
            ['ORDER_FILLED', 'ORDER_PARTIAL'].includes(l.action)
        );

        return {
            user: { id: userId },
            period: {
                start: startDate.toISOString(),
                end: endDate.toISOString(),
            },
            totalTrades: tradeLogs.length,
            logs,
        };
    }

    /**
     * Get AI decision explainability report
     */
    async getAIDecisionReport(
        userId: number,
        startDate: Date,
        endDate: Date,
    ): Promise<{
        totalSignals: number;
        executedSignals: number;
        rejectedSignals: number;
        decisions: Array<{
            timestamp: Date;
            symbol: string;
            action: string;
            confidence: number;
            reasoning?: string;
            outcome?: string;
        }>;
    }> {
        const logs = await this.getUserAuditHistory(userId, {
            startDate,
            endDate,
            limit: 10000,
        });

        const signalLogs = logs.filter(l => l.action.startsWith('SIGNAL_'));
        const executed = signalLogs.filter(l => l.action === 'SIGNAL_EXECUTED');
        const rejected = signalLogs.filter(l => l.action === 'SIGNAL_REJECTED');

        const decisions = signalLogs.map(log => ({
            timestamp: log.timestamp,
            symbol: log.details?.symbol || 'Unknown',
            action: log.details?.signalAction || log.action,
            confidence: log.details?.confidence || 0,
            reasoning: log.aiReasoning,
            outcome: log.action === 'SIGNAL_EXECUTED' ? 'EXECUTED' :
                log.action === 'SIGNAL_REJECTED' ? 'REJECTED' : 'PENDING',
        }));

        return {
            totalSignals: signalLogs.length,
            executedSignals: executed.length,
            rejectedSignals: rejected.length,
            decisions,
        };
    }

    // ==========================================
    // EVENT HANDLERS
    // ==========================================

    @OnEvent('order.placed')
    async handleOrderPlaced(payload: any): Promise<void> {
        await this.logOrder(
            payload.userId,
            payload.portfolioId,
            AuditAction.ORDER_PLACED,
            payload.orderId,
            {
                symbol: payload.symbol,
                side: payload.side,
                quantity: payload.quantity,
                price: payload.price,
            },
        );
    }

    @OnEvent('order.filled')
    async handleOrderFilled(payload: any): Promise<void> {
        await this.logOrder(
            payload.userId,
            payload.portfolioId,
            AuditAction.ORDER_FILLED,
            payload.orderId,
            {
                symbol: payload.symbol,
                side: payload.side,
                quantity: payload.quantity,
                price: payload.price,
                filledQuantity: payload.filledQuantity,
                pnl: payload.pnl,
            },
        );
    }

    @OnEvent('circuit.tripped')
    async handleCircuitTripped(payload: any): Promise<void> {
        await this.logRiskEvent(
            payload.userId,
            payload.portfolioId,
            AuditAction.CIRCUIT_TRIPPED,
            {
                type: payload.type,
                reason: payload.reason,
            },
        );
    }

    @OnEvent('circuit.reset')
    async handleCircuitReset(payload: any): Promise<void> {
        await this.logRiskEvent(
            payload.userId,
            payload.portfolioId,
            AuditAction.CIRCUIT_RESET,
            {
                type: 'manual_reset',
            },
        );
    }

    @OnEvent('ai.signal.received')
    async handleSignalReceived(payload: any): Promise<void> {
        await this.logSignal(
            0, // System-wide signal, no specific user
            AuditAction.SIGNAL_RECEIVED,
            {
                symbol: payload.symbol,
                signalAction: payload.action,
                confidence: payload.confidence,
                models: payload.models,
            },
            payload.reasoning,
        );
    }
}
