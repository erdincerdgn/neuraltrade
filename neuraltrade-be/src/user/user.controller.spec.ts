import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import { JwtService } from '@nestjs/jwt';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminAuthGuard } from '../auth/guards/admin-auth.guard';
import { RolesGuard } from '../auth/guards/roles.guard';
import { UserRole, UserStatus, RiskProfile } from '@prisma/client';

describe('UserController', () => {
    let controller: UserController;

    const mockUser = {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
        name: 'Test',
        surname: 'User',
        role: UserRole.USER,
        status: UserStatus.ACTIVE,
        riskProfile: RiskProfile.MODERATE,
        tradingEnabled: true,
        emailVerified: true,
    };

    const mockStats = {
        totalPortfolios: 2,
        totalPortfolioValue: '10000.00',
        totalRealizedPnL: '500.00',
        totalUnrealizedPnL: '200.00',
        openPositions: 5,
        activeAlerts: 3,
        totalAISignals: 10,
        watchlistCount: 2,
    };

    const mockUserService = {
        findOneById: jest.fn().mockResolvedValue(mockUser),
        findByUsername: jest.fn().mockResolvedValue(mockUser),
        findAll: jest.fn().mockResolvedValue({ items: [mockUser], meta: { total: 1 } }),
        updateProfile: jest.fn().mockResolvedValue({ message: 'Updated', user: mockUser }),
        getUserStats: jest.fn().mockResolvedValue(mockStats),
        searchUsers: jest.fn().mockResolvedValue({ data: [mockUser], total: 1 }),
        adminUpdateUser: jest.fn().mockResolvedValue({ message: 'Updated', user: mockUser }),
        toggleUserTrading: jest.fn().mockResolvedValue({ message: 'Success', tradingEnabled: true }),
    };

    const mockJwtService = {
        sign: jest.fn().mockReturnValue('mock_token'),
        verify: jest.fn().mockReturnValue({ sub: 1, email: 'test@example.com' }),
    };

    // Mock guards that always allow
    const mockGuard = { canActivate: jest.fn().mockReturnValue(true) };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [UserController],
            providers: [
                { provide: UserService, useValue: mockUserService },
                { provide: JwtService, useValue: mockJwtService },
            ],
        })
            .overrideGuard(JwtAuthGuard)
            .useValue(mockGuard)
            .overrideGuard(AdminAuthGuard)
            .useValue(mockGuard)
            .overrideGuard(RolesGuard)
            .useValue(mockGuard)
            .compile();

        controller = module.get<UserController>(UserController);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    // ==========================================
    // AUTHENTICATED USER ENDPOINTS
    // ==========================================

    describe('getMyStats', () => {
        it('should return current user stats', async () => {
            const result = await controller.getMyStats(1);

            expect(result).toEqual(mockStats);
            expect(mockUserService.getUserStats).toHaveBeenCalledWith(1);
        });
    });

    describe('updateMyProfile', () => {
        it('should update current user profile', async () => {
            const updateDto = { name: 'Updated' };

            const result = await controller.updateMyProfile(1, updateDto);

            expect(result.message).toBe('Updated');
            expect(mockUserService.updateProfile).toHaveBeenCalledWith(1, updateDto);
        });
    });

    // ==========================================
    // PUBLIC ENDPOINTS
    // ==========================================

    describe('getByUsername', () => {
        it('should return user by username', async () => {
            const result = await controller.getByUsername('testuser');

            expect(result).toEqual(mockUser);
            expect(mockUserService.findByUsername).toHaveBeenCalledWith('testuser');
        });
    });

    // ==========================================
    // ADMIN ENDPOINTS
    // ==========================================

    describe('getUsers', () => {
        it('should return paginated user list', async () => {
            const paginationDto = { page: 1, perPage: 10 };

            const result = await controller.getUsers(paginationDto);

            expect(result.items).toHaveLength(1);
            expect(mockUserService.findAll).toHaveBeenCalledWith(paginationDto);
        });
    });

    describe('searchUsers', () => {
        it('should search users with filters', async () => {
            const searchDto = { searchTerm: 'test', status: UserStatus.ACTIVE };

            const result = await controller.searchUsers(searchDto);

            expect(result.data).toHaveLength(1);
            expect(mockUserService.searchUsers).toHaveBeenCalledWith(searchDto);
        });
    });

    describe('getUserById', () => {
        it('should return user by ID', async () => {
            const result = await controller.getUserById(1);

            expect(result).toEqual(mockUser);
            expect(mockUserService.findOneById).toHaveBeenCalledWith(1);
        });
    });

    describe('adminUpdateUser', () => {
        it('should update user by admin', async () => {
            const updateDto = { status: UserStatus.SUSPENDED };

            const result = await controller.adminUpdateUser(1, updateDto);

            expect(result.message).toBe('Updated');
            expect(mockUserService.adminUpdateUser).toHaveBeenCalledWith(1, updateDto);
        });
    });

    describe('enableTrading', () => {
        it('should enable user trading', async () => {
            const result = await controller.enableTrading(1);

            expect(result.tradingEnabled).toBe(true);
            expect(mockUserService.toggleUserTrading).toHaveBeenCalledWith(1, true);
        });
    });

    describe('disableTrading', () => {
        it('should disable user trading', async () => {
            mockUserService.toggleUserTrading.mockResolvedValue({ message: 'Disabled', tradingEnabled: false });

            const result = await controller.disableTrading(1);

            expect(result.tradingEnabled).toBe(false);
            expect(mockUserService.toggleUserTrading).toHaveBeenCalledWith(1, false);
        });
    });
});
