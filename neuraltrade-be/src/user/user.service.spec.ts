import { Test, TestingModule } from '@nestjs/testing';
import { UserService } from './user.service';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import { QueueService } from '../core/bullmq/queue.service';
import { PaginationService } from '../common/pagination/services/pagination.service';
import { NotFoundException, ConflictException } from '@nestjs/common';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';

describe('UserService', () => {
    let service: UserService;

    const mockUser = {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
        name: 'Test',
        surname: 'User',
        role: UserRole.USER,
        status: UserStatus.ACTIVE,
        phoneNumber: '5551234567',
        profilePhoto: null,
        profileDescription: null,
        gender: GenderType.Unspecified,
        dateOfBirth: null,
        age: null,
        riskProfile: RiskProfile.MODERATE,
        maxDailyLoss: null,
        maxPositionSize: null,
        maxLeverage: null,
        tradingEnabled: true,
        circuitBreakerUntil: null,
        emailVerified: true,
        lastLoginAt: null,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockPrismaService = {
        user: {
            findUnique: jest.fn(),
            findFirst: jest.fn(),
            findMany: jest.fn(),
            count: jest.fn(),
            update: jest.fn(),
        },
        portfolio: {
            findMany: jest.fn(),
        },
        position: {
            count: jest.fn(),
        },
        alert: {
            count: jest.fn(),
        },
        aISignal: {
            count: jest.fn(),
        },
        watchlist: {
            count: jest.fn(),
        },
    };

    const mockRedisService = {
        get: jest.fn(),
        set: jest.fn(),
        delete: jest.fn(),
    };

    const mockQueueService = {
        addAnalyticsJob: jest.fn(),
        addNotificationJob: jest.fn(),
    };

    const mockPaginationService = {
        paginate: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                UserService,
                { provide: PrismaService, useValue: mockPrismaService },
                { provide: RedisService, useValue: mockRedisService },
                { provide: QueueService, useValue: mockQueueService },
                { provide: PaginationService, useValue: mockPaginationService },
            ],
        }).compile();

        service = module.get<UserService>(UserService);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    // ==========================================
    // FIND USER TESTS
    // ==========================================

    describe('findOneById', () => {
        it('should return cached user if exists', async () => {
            mockRedisService.get.mockResolvedValue(mockUser);

            const result = await service.findOneById(1);

            expect(result).toEqual(mockUser);
            expect(mockRedisService.get).toHaveBeenCalled();
            expect(mockPrismaService.user.findUnique).not.toHaveBeenCalled();
        });

        it('should fetch from database if not cached', async () => {
            mockRedisService.get.mockResolvedValue(null);
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);

            const result = await service.findOneById(1);

            expect(result).toEqual(mockUser);
            expect(mockPrismaService.user.findUnique).toHaveBeenCalled();
            expect(mockRedisService.set).toHaveBeenCalled();
        });

        it('should throw NotFoundException if user not found', async () => {
            mockRedisService.get.mockResolvedValue(null);
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            await expect(service.findOneById(999)).rejects.toThrow(NotFoundException);
        });
    });

    describe('findByUsername', () => {
        it('should return cached user by username', async () => {
            mockRedisService.get.mockResolvedValue(mockUser);

            const result = await service.findByUsername('testuser');

            expect(result).toEqual(mockUser);
        });

        it('should throw NotFoundException if not found', async () => {
            mockRedisService.get.mockResolvedValue(null);
            mockPrismaService.user.findFirst.mockResolvedValue(null);

            await expect(service.findByUsername('unknown')).rejects.toThrow(NotFoundException);
        });
    });

    // ==========================================
    // UPDATE PROFILE TESTS
    // ==========================================

    describe('updateProfile', () => {
        const updateDto = {
            name: 'Updated',
            surname: 'Name',
        };

        it('should update user profile successfully', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            mockPrismaService.user.update.mockResolvedValue({ ...mockUser, ...updateDto });
            mockRedisService.delete.mockResolvedValue(undefined);

            const result = await service.updateProfile(1, updateDto);

            expect(result.message).toBe('Profile updated successfully');
            expect(result.user.name).toBe('Updated');
            expect(mockRedisService.delete).toHaveBeenCalled();
        });

        it('should throw NotFoundException if user not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            await expect(service.updateProfile(999, updateDto)).rejects.toThrow(NotFoundException);
        });

        it('should throw ConflictException if email already exists', async () => {
            mockPrismaService.user.findUnique
                .mockResolvedValueOnce(mockUser)
                .mockResolvedValueOnce({ id: 2, email: 'other@example.com' });

            await expect(service.updateProfile(1, { email: 'other@example.com' })).rejects.toThrow(ConflictException);
        });

        it('should throw ConflictException if username already taken', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            mockPrismaService.user.findFirst.mockResolvedValue({ id: 2, username: 'taken' });

            await expect(service.updateProfile(1, { username: 'taken' })).rejects.toThrow(ConflictException);
        });
    });

    // ==========================================
    // USER STATS TESTS
    // ==========================================

    describe('getUserStats', () => {
        it('should return cached stats if exists', async () => {
            const cachedStats = {
                totalPortfolios: 2,
                totalPortfolioValue: '10000.00',
                totalRealizedPnL: '500.00',
                totalUnrealizedPnL: '200.00',
                openPositions: 5,
                activeAlerts: 3,
                totalAISignals: 10,
                watchlistCount: 2,
            };
            mockRedisService.get.mockResolvedValue(cachedStats);

            const result = await service.getUserStats(1);

            expect(result).toEqual(cachedStats);
        });

        it('should calculate stats from database if not cached', async () => {
            mockRedisService.get.mockResolvedValue(null);
            mockPrismaService.portfolio.findMany.mockResolvedValue([
                { totalValue: 5000, totalPnL: 100 },
                { totalValue: 5000, totalPnL: 100 },
            ]);
            mockPrismaService.position.count.mockResolvedValue(5);
            mockPrismaService.alert.count.mockResolvedValue(3);
            mockPrismaService.aISignal.count.mockResolvedValue(10);
            mockPrismaService.watchlist.count.mockResolvedValue(2);

            const result = await service.getUserStats(1);

            expect(result.totalPortfolios).toBe(2);
            expect(result.totalPortfolioValue).toBe('10000.00');
            expect(result.openPositions).toBe(5);
            expect(mockRedisService.set).toHaveBeenCalled();
        });
    });

    // ==========================================
    // ADMIN TESTS
    // ==========================================

    describe('adminUpdateUser', () => {
        it('should update user by admin', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            mockPrismaService.user.update.mockResolvedValue({ ...mockUser, status: UserStatus.SUSPENDED });
            mockRedisService.delete.mockResolvedValue(undefined);
            mockQueueService.addAnalyticsJob.mockResolvedValue({});

            const result = await service.adminUpdateUser(1, { status: UserStatus.SUSPENDED });

            expect(result.message).toBe('User updated successfully');
            expect(mockQueueService.addAnalyticsJob).toHaveBeenCalled();
        });

        it('should throw NotFoundException if user not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            await expect(service.adminUpdateUser(999, {})).rejects.toThrow(NotFoundException);
        });
    });

    describe('toggleUserTrading', () => {
        it('should enable trading', async () => {
            mockPrismaService.user.update.mockResolvedValue({ ...mockUser, tradingEnabled: true });
            mockRedisService.delete.mockResolvedValue(undefined);
            mockQueueService.addNotificationJob.mockResolvedValue({});

            const result = await service.toggleUserTrading(1, true);

            expect(result.tradingEnabled).toBe(true);
            expect(mockQueueService.addNotificationJob).toHaveBeenCalled();
        });

        it('should disable trading', async () => {
            mockPrismaService.user.update.mockResolvedValue({ ...mockUser, tradingEnabled: false });
            mockRedisService.delete.mockResolvedValue(undefined);
            mockQueueService.addNotificationJob.mockResolvedValue({});

            const result = await service.toggleUserTrading(1, false);

            expect(result.tradingEnabled).toBe(false);
        });
    });

    // ==========================================
    // SEARCH TESTS
    // ==========================================

    describe('searchUsers', () => {
        it('should search users with filters', async () => {
            const users = [mockUser];
            mockPrismaService.user.findMany.mockResolvedValue(users);
            mockPrismaService.user.count.mockResolvedValue(1);

            const result = await service.searchUsers({
                page: 1,
                limit: 10,
                searchTerm: 'test',
            });

            expect(result.data).toEqual(users);
            expect(result.total).toBe(1);
        });

        it('should filter by status', async () => {
            mockPrismaService.user.findMany.mockResolvedValue([]);
            mockPrismaService.user.count.mockResolvedValue(0);

            await service.searchUsers({ status: UserStatus.ACTIVE });

            expect(mockPrismaService.user.findMany).toHaveBeenCalled();
        });

        it('should filter by role', async () => {
            mockPrismaService.user.findMany.mockResolvedValue([]);
            mockPrismaService.user.count.mockResolvedValue(0);

            await service.searchUsers({ role: UserRole.NEURALTRADE });

            expect(mockPrismaService.user.findMany).toHaveBeenCalled();
        });
    });
});
