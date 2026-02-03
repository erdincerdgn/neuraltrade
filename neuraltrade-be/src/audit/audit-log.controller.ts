import { Controller, Get, Query, Param, UseGuards } from '@nestjs/common';
import { AuditLogService, AuditAction } from './audit-log.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

/**
 * Audit Log Controller
 * 
 * API endpoints for audit log queries and exports.
 */
@Controller('api/v1/audit')
@UseGuards(JwtAuthGuard)
export class AuditLogController {
    constructor(private readonly auditService: AuditLogService) { }

    /**
     * Get user's audit history
     */
    @Get('history/:userId')
    async getHistory(
        @Param('userId') userId: string,
        @Query('portfolioId') portfolioId?: string,
        @Query('action') action?: string,
        @Query('days') days: string = '30',
        @Query('limit') limit: string = '100',
    ) {
        const startDate = new Date(Date.now() - parseInt(days) * 24 * 60 * 60 * 1000);

        const logs = await this.auditService.getUserAuditHistory(parseInt(userId), {
            portfolioId: portfolioId ? parseInt(portfolioId) : undefined,
            action: action as AuditAction,
            startDate,
            limit: parseInt(limit),
        });

        return { userId: parseInt(userId), count: logs.length, logs };
    }

    /**
     * Export for compliance/regulatory reporting
     */
    @Get('export/:userId')
    async exportCompliance(
        @Param('userId') userId: string,
        @Query('startDate') startDate?: string,
        @Query('endDate') endDate?: string,
    ) {
        const start = startDate ? new Date(startDate) : new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);
        const end = endDate ? new Date(endDate) : new Date();

        return this.auditService.exportForCompliance(parseInt(userId), start, end);
    }

    /**
     * Get AI decision explainability report
     */
    @Get('ai-decisions/:userId')
    async getAIDecisions(
        @Param('userId') userId: string,
        @Query('days') days: string = '30',
    ) {
        const startDate = new Date(Date.now() - parseInt(days) * 24 * 60 * 60 * 1000);

        return this.auditService.getAIDecisionReport(
            parseInt(userId),
            startDate,
            new Date(),
        );
    }
}
