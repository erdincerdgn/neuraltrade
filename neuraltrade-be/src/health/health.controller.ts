import { Controller, Get } from '@nestjs/common';
import {
    HealthCheck,
    HealthCheckService,
    HealthCheckResult,
    HealthIndicatorResult,
    HealthIndicator,
} from '@nestjs/terminus';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';

/**
 * Custom Database Health Indicator
 */
class DatabaseHealthIndicator extends HealthIndicator {
    constructor(private readonly prismaService: PrismaService) {
        super();
    }

    async isHealthy(key: string): Promise<HealthIndicatorResult> {
        try {
            await this.prismaService.$queryRaw`SELECT 1`;
            return this.getStatus(key, true);
        } catch (error) {
            return this.getStatus(key, false, { error: (error as Error).message });
        }
    }
}

/**
 * Custom Redis Health Indicator
 */
class RedisHealthIndicator extends HealthIndicator {
    constructor(private readonly redisService: RedisService) {
        super();
    }

    async isHealthy(key: string): Promise<HealthIndicatorResult> {
        try {
            const pong = await this.redisService.ping();
            return this.getStatus(key, pong === 'PONG');
        } catch (error) {
            return this.getStatus(key, false, { error: (error as Error).message });
        }
    }
}

/**
 * Health Controller
 * 
 * Kubernetes-compatible health check endpoints.
 */
@Controller('health')
export class HealthController {
    private readonly dbIndicator: DatabaseHealthIndicator;
    private readonly redisIndicator: RedisHealthIndicator;

    constructor(
        private readonly health: HealthCheckService,
        prisma: PrismaService,
        redis: RedisService,
    ) {
        this.dbIndicator = new DatabaseHealthIndicator(prisma);
        this.redisIndicator = new RedisHealthIndicator(redis);
    }

    @Get()
    @HealthCheck()
    async check(): Promise<HealthCheckResult> {
        return this.health.check([
            () => this.dbIndicator.isHealthy('database'),
            () => this.redisIndicator.isHealthy('redis'),
            () => this.checkMemory(),
        ]);
    }

    @Get('live')
    liveness(): { status: string; timestamp: string } {
        return {
            status: 'ok',
            timestamp: new Date().toISOString(),
        };
    }

    @Get('ready')
    @HealthCheck()
    async readiness(): Promise<HealthCheckResult> {
        return this.health.check([
            () => this.dbIndicator.isHealthy('database'),
            () => this.redisIndicator.isHealthy('redis'),
        ]);
    }

    private async checkMemory(): Promise<HealthIndicatorResult> {
        const used = process.memoryUsage();
        const heapUsedMB = Math.round(used.heapUsed / 1024 / 1024);
        const heapTotalMB = Math.round(used.heapTotal / 1024 / 1024);
        const heapPercent = Math.round((used.heapUsed / used.heapTotal) * 100);
        const isHealthy = heapPercent < 85;

        return {
            memory: {
                status: isHealthy ? 'up' : 'down',
                heapUsedMB,
                heapTotalMB,
                heapPercent,
            },
        };
    }
}
