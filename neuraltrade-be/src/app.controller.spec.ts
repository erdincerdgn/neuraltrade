import { Test, TestingModule } from '@nestjs/testing';
import { AppController } from './app.controller';
import { AppService } from './app.service';

describe('AppController', () => {
    let controller: AppController;
    let appService: AppService;

    const mockAppInfo = {
        name: 'NeuralTrade API',
        version: '1.0.0',
        environment: 'test',
        timestamp: new Date().toISOString(),
    };

    const mockHealthCheck = {
        status: 'healthy',
        version: '1.0.0',
        environment: 'test',
        uptime: 1000,
        timestamp: new Date().toISOString(),
        responseTime: 10,
        services: {
            database: { status: 'healthy', latency: 5 },
            redis: { status: 'healthy', latency: 2 },
            queues: { status: 'healthy', queues: 7 },
        },
    };

    const mockSystemStats = {
        database: { users: 100, portfolios: 50 },
        queues: { trading: { waiting: 0 } },
        memory: { heapUsed: 50, heapTotal: 100, rss: 150 },
        uptime: 1000,
    };

    const mockAppService = {
        getAppInfo: jest.fn().mockReturnValue(mockAppInfo),
        healthCheck: jest.fn().mockResolvedValue(mockHealthCheck),
        readinessCheck: jest.fn().mockResolvedValue({ ready: true }),
        livenessCheck: jest.fn().mockReturnValue({ alive: true }),
        getSystemStats: jest.fn().mockResolvedValue(mockSystemStats),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [AppController],
            providers: [
                { provide: AppService, useValue: mockAppService },
            ],
        }).compile();

        controller = module.get<AppController>(AppController);
        appService = module.get<AppService>(AppService);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    describe('getAppInfo', () => {
        it('should return app info', () => {
            const result = controller.getAppInfo();

            expect(result).toEqual(mockAppInfo);
            expect(appService.getAppInfo).toHaveBeenCalled();
        });
    });

    describe('healthCheck', () => {
        it('should return health status', async () => {
            const result = await controller.healthCheck();

            expect(result.status).toBe('healthy');
            expect(result.services).toBeDefined();
            expect(appService.healthCheck).toHaveBeenCalled();
        });
    });

    describe('readinessCheck', () => {
        it('should return readiness status', async () => {
            const result = await controller.readinessCheck();

            expect(result.ready).toBe(true);
            expect(appService.readinessCheck).toHaveBeenCalled();
        });
    });

    describe('livenessCheck', () => {
        it('should return liveness status', () => {
            const result = controller.livenessCheck();

            expect(result.alive).toBe(true);
            expect(appService.livenessCheck).toHaveBeenCalled();
        });
    });

    describe('getSystemStats', () => {
        it('should return system stats', async () => {
            const result = await controller.getSystemStats();

            expect(result).toEqual(mockSystemStats);
            expect(appService.getSystemStats).toHaveBeenCalled();
        });
    });
});