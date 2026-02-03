import { Test, TestingModule } from '@nestjs/testing';
import { ValidationPipe } from '@nestjs/common';
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify';
import request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/core/prisma/prisma.service';
import { RedisService } from '../src/core/redis/redis.service';
import { QueueService } from '../src/core/bullmq/queue.service';

describe('NeuralTrade API (e2e)', () => {
    let app: NestFastifyApplication;

    const mockPrismaService = {
        user: {
            findUnique: jest.fn(),
            findFirst: jest.fn(),
            create: jest.fn(),
            update: jest.fn(),
            count: jest.fn().mockResolvedValue(0),
        },
        portfolio: { count: jest.fn().mockResolvedValue(0), findMany: jest.fn().mockResolvedValue([]) },
        position: { count: jest.fn().mockResolvedValue(0) },
        alert: { count: jest.fn().mockResolvedValue(0) },
        aISignal: { count: jest.fn().mockResolvedValue(0) },
        watchlist: { count: jest.fn().mockResolvedValue(0) },
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', latency: 5 }),
        getDatabaseStats: jest.fn().mockResolvedValue({ users: 0, portfolios: 0 }),
        $connect: jest.fn(),
        $disconnect: jest.fn(),
    };

    const mockRedisService = {
        get: jest.fn().mockResolvedValue(null),
        set: jest.fn(),
        delete: jest.fn(),
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', latency: 2 }),
    };

    const mockQueueService = {
        addJob: jest.fn(),
        healthCheck: jest.fn().mockResolvedValue({ status: 'healthy', queues: 7 }),
        getAllQueueStats: jest.fn().mockResolvedValue({}),
    };

    beforeAll(async () => {
        const moduleFixture: TestingModule = await Test.createTestingModule({
            imports: [AppModule],
        })
            .overrideProvider(PrismaService)
            .useValue(mockPrismaService)
            .overrideProvider(RedisService)
            .useValue(mockRedisService)
            .overrideProvider(QueueService)
            .useValue(mockQueueService)
            .compile();

        app = moduleFixture.createNestApplication<NestFastifyApplication>(
            new FastifyAdapter(),
        );

        app.useGlobalPipes(
            new ValidationPipe({
                transform: true,
                whitelist: true,
                forbidNonWhitelisted: true,
            }),
        );

        await app.init();
        await app.getHttpAdapter().getInstance().ready();
    });

    afterAll(async () => {
        await app.close();
    });

    // ==========================================
    // HEALTH CHECK TESTS
    // ==========================================

    describe('System Endpoints', () => {
        it('GET / - should return app info', () => {
            return request(app.getHttpServer())
                .get('/')
                .expect(200)
                .expect((res) => {
                    expect(res.body).toHaveProperty('name', 'NeuralTrade API');
                    expect(res.body).toHaveProperty('version');
                    expect(res.body).toHaveProperty('environment');
                });
        });

        it('GET /health - should return health status', () => {
            return request(app.getHttpServer())
                .get('/health')
                .expect(200)
                .expect((res) => {
                    expect(res.body).toHaveProperty('status');
                    expect(res.body).toHaveProperty('services');
                });
        });

        it('GET /health/ready - should return readiness status', () => {
            return request(app.getHttpServer())
                .get('/health/ready')
                .expect(200)
                .expect((res) => {
                    expect(res.body).toHaveProperty('ready');
                });
        });

        it('GET /health/live - should return liveness status', () => {
            return request(app.getHttpServer())
                .get('/health/live')
                .expect(200)
                .expect((res) => {
                    expect(res.body).toHaveProperty('alive', true);
                });
        });
    });

    // ==========================================
    // AUTH ENDPOINTS TESTS
    // ==========================================

    describe('Auth Endpoints', () => {
        describe('POST /api/v1/auth/register', () => {
            it('should validate required fields', () => {
                return request(app.getHttpServer())
                    .post('/api/v1/auth/register')
                    .send({})
                    .expect(400)
                    .expect((res) => {
                        expect(res.body.message).toBeDefined();
                    });
            });

            it('should validate email format', () => {
                return request(app.getHttpServer())
                    .post('/api/v1/auth/register')
                    .send({
                        email: 'invalid-email',
                        password: 'Password123!',
                        name: 'Test',
                        surname: 'User',
                    })
                    .expect(400);
            });

            it('should validate password strength', () => {
                return request(app.getHttpServer())
                    .post('/api/v1/auth/register')
                    .send({
                        email: 'test@example.com',
                        password: '123',
                        name: 'Test',
                        surname: 'User',
                    })
                    .expect(400);
            });
        });

        describe('POST /api/v1/auth/login', () => {
            it('should validate required fields', () => {
                return request(app.getHttpServer())
                    .post('/api/v1/auth/login')
                    .send({})
                    .expect(400);
            });
        });

        describe('POST /api/v1/auth/forgot-password', () => {
            it('should validate email', () => {
                return request(app.getHttpServer())
                    .post('/api/v1/auth/forgot-password')
                    .send({ email: 'invalid' })
                    .expect(400);
            });
        });
    });

    // ==========================================
    // PROTECTED ENDPOINTS TESTS
    // ==========================================

    describe('Protected Endpoints', () => {
        it('GET /api/v1/auth/me - should return 401 without token', () => {
            return request(app.getHttpServer())
                .get('/api/v1/auth/me')
                .expect(401);
        });

        it('GET /api/v1/user/me/stats - should return 401 without token', () => {
            return request(app.getHttpServer())
                .get('/api/v1/user/me/stats')
                .expect(401);
        });

        it('PATCH /api/v1/user/me/profile - should return 401 without token', () => {
            return request(app.getHttpServer())
                .patch('/api/v1/user/me/profile')
                .send({ name: 'Test' })
                .expect(401);
        });
    });

    // ==========================================
    // ADMIN ENDPOINTS TESTS
    // ==========================================

    describe('Admin Endpoints', () => {
        it('GET /api/v1/user/admin/list - should return 401 without token', () => {
            return request(app.getHttpServer())
                .get('/api/v1/user/admin/list')
                .expect(401);
        });

        it('PATCH /api/v1/user/admin/1 - should return 401 without token', () => {
            return request(app.getHttpServer())
                .patch('/api/v1/user/admin/1')
                .send({ status: 'SUSPENDED' })
                .expect(401);
        });
    });
});
