import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from './core/prisma/prisma.service';
import { RedisService } from './core/redis/redis.service';
import { QueueService } from './core/bullmq/queue.service';

/**
 * App Service - Root level application services
 * 
 * Provides:
 * - Health checks for all services
 * - System information
 * - Version info
 */
@Injectable()
export class AppService {
    private readonly logger = new Logger(AppService.name);

    // App version from package.json or env
    private readonly version = process.env.APP_VERSION || '1.0.0';
    private readonly environment = process.env.NODE_ENV || 'development';

    constructor(
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
        private readonly queue: QueueService,
    ) { }

    /**
     * Simple app info
     */
    getAppInfo() {
        return {
            name: 'NeuralTrade API',
            version: this.version,
            environment: this.environment,
            timestamp: new Date().toISOString(),
        };
    }

    /**
     * Comprehensive health check for all services
     */
    async healthCheck() {
        const startTime = Date.now();

        const [dbHealth, redisHealth, queueHealth] = await Promise.all([
            this.checkDatabase(),
            this.checkRedis(),
            this.checkQueues(),
        ]);

        const isHealthy = dbHealth.status === 'healthy'
            && redisHealth.status === 'healthy'
            && queueHealth.status === 'healthy';

        return {
            status: isHealthy ? 'healthy' : 'degraded',
            version: this.version,
            environment: this.environment,
            uptime: process.uptime(),
            timestamp: new Date().toISOString(),
            responseTime: Date.now() - startTime,
            services: {
                database: dbHealth,
                redis: redisHealth,
                queues: queueHealth,
            },
        };
    }

    /**
     * Database health check
     */
    private async checkDatabase(): Promise<{ status: string; latency: number }> {
        try {
            return await this.prisma.healthCheck();
        } catch (error) {
            this.logger.error('Database health check failed', error);
            return { status: 'unhealthy', latency: -1 };
        }
    }

    /**
     * Redis health check
     */
    private async checkRedis(): Promise<{ status: string; latency: number }> {
        try {
            return await this.redis.healthCheck();
        } catch (error) {
            this.logger.error('Redis health check failed', error);
            return { status: 'unhealthy', latency: -1 };
        }
    }

    /**
     * Queue health check
     */
    private async checkQueues(): Promise<{ status: string; queues: number }> {
        try {
            return await this.queue.healthCheck();
        } catch (error) {
            this.logger.error('Queue health check failed', error);
            return { status: 'unhealthy', queues: 0 };
        }
    }

    /**
     * Get system statistics
     */
    async getSystemStats() {
        const dbStats = await this.prisma.getDatabaseStats();
        const queueStats = await this.queue.getAllQueueStats();

        return {
            database: dbStats,
            queues: queueStats,
            memory: {
                heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
                heapTotal: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
                rss: Math.round(process.memoryUsage().rss / 1024 / 1024),
            },
            uptime: process.uptime(),
        };
    }

    /**
     * Readiness check - is the app ready to serve requests?
     */
    async readinessCheck(): Promise<{ ready: boolean; reason?: string }> {
        try {
            const dbHealth = await this.prisma.healthCheck();

            if (dbHealth.status !== 'healthy') {
                return { ready: false, reason: 'Database not ready' };
            }

            return { ready: true };
        } catch (error) {
            return { ready: false, reason: 'Health check failed' };
        }
    }

    /**
     * Liveness check - is the app alive?
     */
    livenessCheck(): { alive: boolean } {
        return { alive: true };
    }
}