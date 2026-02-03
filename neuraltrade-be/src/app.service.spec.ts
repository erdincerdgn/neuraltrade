import { Test, TestingModule } from '@nestjs/testing';
import { AppService } from './app.service';
import { PrismaService } from './core/prisma/prisma.service';
import { RedisService } from './core/redis/redis.service';
import { QueueService } from './core/bullmq/queue.service';

describe('AppService', () => {
    let service: AppService;

    const mockPrismaService = {
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', latency: 5 }),
        getDatabaseStats: jest.fn().mockResolvedValue({
            users: 100,
            portfolios: 50,
            positions: 200,
            alerts: 30,
            aiSignals: 500,
        }),
    };

    const mockRedisService = {
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', latency: 2 }),
    };

    const mockQueueService = {
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', queues: 7 }),
        getAllQueueStats: jest.fn().mockResolvedValue({
            trading: { waiting: 0, active: 0, completed: 100, failed: 0 },
            signals: { waiting: 5, active: 2, completed: 500, failed: 10 },
        }),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AppService,
                { provide: PrismaService, useValue: mockPrismaService },
                { provide: RedisService, useValue: mockRedisService },
                { provide: QueueService, useValue: mockQueueService },
            ],
        }).compile();

        service = module.get<AppService>(AppService);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    // ==========================================
    // APP INFO TESTS
    // ==========================================

    describe('getAppInfo', () => {
        it('should return app info', () => {
            const result = service.getAppInfo();

            expect(result).toHaveProperty('name', 'NeuralTrade API');
            expect(result).toHaveProperty('version');
            expect(result).toHaveProperty('environment');
            expect(result).toHaveProperty('timestamp');
        });
    });

    // ==========================================
    // HEALTH CHECK TESTS
    // ==========================================

    describe('healthCheck', () => {
        it('should return healthy status when all services are healthy', async () => {
            const result = await service.healthCheck();

            expect(result.status).toBe('healthy');
            expect(result.services.database.status).toBe('healthy');
            expect(result.services.redis.status).toBe('healthy');
            expect(result.services.queues.status).toBe('healthy');
        });

        it('should return degraded status when any service is unhealthy', async () => {
            mockRedisService.healthCheck.mockResolvedValue({ status: 'unhealthy', latency: -1 });

            const result = await service.healthCheck();

            expect(result.status).toBe('degraded');
        });

        it('should include response time', async () => {
            const result = await service.healthCheck();

            expect(result).toHaveProperty('responseTime');
            expect(typeof result.responseTime).toBe('number');
        });

        it('should include uptime', async () => {
            const result = await service.healthCheck();

            expect(result).toHaveProperty('uptime');
            expect(typeof result.uptime).toBe('number');
        });
    });

    // ==========================================
    // READINESS/LIVENESS TESTS
    // ==========================================

    describe('readinessCheck', () => {
        it('should return ready when database is healthy', async () => {
            const result = await service.readinessCheck();

            expect(result.ready).toBe(true);
        });

        it('should return not ready when database is unhealthy', async () => {
            mockPrismaService.healthCheck.mockResolvedValue({ status: 'unhealthy', latency: -1 });

            const result = await service.readinessCheck();

            expect(result.ready).toBe(false);
            expect(result.reason).toBe('Database not ready');
        });
    });

    describe('livenessCheck', () => {
        it('should always return alive', () => {
            const result = service.livenessCheck();

            expect(result.alive).toBe(true);
        });
    });

    // ==========================================
    // SYSTEM STATS TESTS
    // ==========================================

    describe('getSystemStats', () => {
        it('should return database and queue stats', async () => {
            const result = await service.getSystemStats();

            expect(result).toHaveProperty('database');
            expect(result).toHaveProperty('queues');
            expect(result).toHaveProperty('memory');
            expect(result).toHaveProperty('uptime');
        });

        it('should include memory usage', async () => {
            const result = await service.getSystemStats();

            expect(result.memory).toHaveProperty('heapUsed');
            expect(result.memory).toHaveProperty('heapTotal');
            expect(result.memory).toHaveProperty('rss');
        });

        it('should include database stats', async () => {
            const result = await service.getSystemStats();

            expect(result.database.users).toBe(100);
            expect(result.database.portfolios).toBe(50);
        });
    });
});
