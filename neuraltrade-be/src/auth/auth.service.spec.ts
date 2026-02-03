import { Test, TestingModule } from '@nestjs/testing';
import { AuthService } from './auth.service';
import { PrismaService } from '../core/prisma/prisma.service';
import { JwtService } from '@nestjs/jwt';
import { UnauthorizedException, ConflictException, ForbiddenException, BadRequestException } from '@nestjs/common';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';
import * as bcrypt from 'bcrypt';

// Mock bcrypt
jest.mock('bcrypt', () => ({
    hash: jest.fn().mockResolvedValue('hashed_password'),
    compare: jest.fn(),
}));

describe('AuthService', () => {
    let service: AuthService;

    const mockUser = {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
        name: 'Test',
        surname: 'User',
        password: 'hashed_password',
        role: UserRole.USER,
        status: UserStatus.ACTIVE,
        phoneNumber: '5551234567',
        profilePhoto: null,
        profileDescription: null,
        profileBanner: null,
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
        failedLoginAttempts: 0,
        lastLoginAt: null,
        lastPasswordChangeAt: null,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockPrismaService = {
        user: {
            findUnique: jest.fn(),
            findFirst: jest.fn(),
            create: jest.fn(),
            update: jest.fn(),
        },
        subscription: {
            findUnique: jest.fn(),
        },
        session: {
            findUnique: jest.fn(),
        },
    };

    const mockJwtService = {
        sign: jest.fn().mockReturnValue('mock_jwt_token'),
        verify: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AuthService,
                { provide: PrismaService, useValue: mockPrismaService },
                { provide: JwtService, useValue: mockJwtService },
            ],
        }).compile();

        service = module.get<AuthService>(AuthService);

        // Reset mocks
        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    // ==========================================
    // REGISTER TESTS
    // ==========================================

    describe('register', () => {
        const registerDto = {
            email: 'new@example.com',
            password: 'Password123!',
            name: 'New',
            surname: 'User',
            phoneNumber: '5559876543',
        };

        it('should successfully register a new user', async () => {
            mockPrismaService.user.findFirst.mockResolvedValue(null);
            mockPrismaService.user.create.mockResolvedValue({
                ...mockUser,
                email: registerDto.email,
            });

            const result = await service.register(registerDto);

            expect(result).toHaveProperty('accessToken');
            expect(result).toHaveProperty('user');
            expect(result.user.email).toBe(registerDto.email);
            expect(mockPrismaService.user.create).toHaveBeenCalled();
        });

        it('should throw ConflictException if email already exists', async () => {
            // Return a user with matching email to trigger conflict
            mockPrismaService.user.findFirst.mockResolvedValue({
                ...mockUser,
                email: registerDto.email, // Same email as registerDto
            });

            await expect(service.register(registerDto)).rejects.toThrow(ConflictException);
        });

        it('should throw ConflictException if username already exists', async () => {
            mockPrismaService.user.findFirst.mockResolvedValue({ ...mockUser, email: 'other@example.com' });

            await expect(service.register({ ...registerDto, username: 'testuser' })).rejects.toThrow(ConflictException);
        });
    });

    // ==========================================
    // LOGIN TESTS
    // ==========================================

    describe('loginFrontend', () => {
        const loginDto = {
            email: 'test@example.com',
            password: 'Password123!',
        };

        it('should successfully login a user', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            mockPrismaService.user.update.mockResolvedValue(mockUser);
            mockPrismaService.subscription.findUnique.mockResolvedValue(null);

            const result = await service.loginFrontend(loginDto);

            expect(result).toHaveProperty('accessToken');
            expect(result).toHaveProperty('user');
            expect(result.accessToken).toBe('mock_jwt_token');
        });

        it('should throw UnauthorizedException if user not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            await expect(service.loginFrontend(loginDto)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if password is wrong', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(false);

            await expect(service.loginFrontend(loginDto)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw ForbiddenException if user is suspended', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue({
                ...mockUser,
                status: UserStatus.SUSPENDED,
            });

            await expect(service.loginFrontend(loginDto)).rejects.toThrow(ForbiddenException);
        });

        it('should throw ForbiddenException if user has too many failed attempts', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue({
                ...mockUser,
                failedLoginAttempts: 6,
            });

            await expect(service.loginFrontend(loginDto)).rejects.toThrow(ForbiddenException);
        });
    });

    describe('loginAdmin', () => {
        const loginDto = {
            email: 'admin@example.com',
            password: 'AdminPassword123!',
        };

        it('should successfully login an admin user', async () => {
            const adminUser = {
                ...mockUser,
                role: UserRole.NEURALTRADE,
                emailVerified: true,
            };
            mockPrismaService.user.findUnique.mockResolvedValue(adminUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            mockPrismaService.user.update.mockResolvedValue(adminUser);

            const result = await service.loginAdmin(loginDto);

            expect(result).toHaveProperty('accessToken');
            expect(result.user.role).toBe(UserRole.NEURALTRADE);
        });

        it('should throw UnauthorizedException if not admin role', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            mockPrismaService.user.update.mockResolvedValue(mockUser);

            await expect(service.loginAdmin(loginDto)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw ForbiddenException if email not verified', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue({
                ...mockUser,
                role: UserRole.NEURALTRADE,
                emailVerified: false,
            });
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            mockPrismaService.user.update.mockResolvedValue(mockUser);

            await expect(service.loginAdmin(loginDto)).rejects.toThrow(ForbiddenException);
        });
    });

    // ==========================================
    // VALIDATE USER TESTS
    // ==========================================

    describe('validateUser', () => {
        it('should return user if found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);

            const result = await service.validateUser(1);

            expect(result).toEqual(mockUser);
        });

        it('should return null if user not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            const result = await service.validateUser(999);

            expect(result).toBeNull();
        });
    });

    // ==========================================
    // PASSWORD TESTS
    // ==========================================

    describe('changePassword', () => {
        const changePasswordDto = {
            currentPassword: 'OldPassword123!',
            newPassword: 'NewPassword456!',
            confirmPassword: 'NewPassword456!',
        };

        it('should successfully change password', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            mockPrismaService.user.update.mockResolvedValue(mockUser);

            const result = await service.changePassword(1, changePasswordDto);

            expect(result.message).toBe('Password changed successfully');
        });

        it('should throw BadRequestException if passwords do not match', async () => {
            await expect(service.changePassword(1, {
                ...changePasswordDto,
                confirmPassword: 'DifferentPassword!',
            })).rejects.toThrow(BadRequestException);
        });

        it('should throw UnauthorizedException if current password is wrong', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);
            (bcrypt.compare as jest.Mock).mockResolvedValue(false);

            await expect(service.changePassword(1, changePasswordDto)).rejects.toThrow(UnauthorizedException);
        });
    });

    describe('forgotPassword', () => {
        it('should return success message even if user not found (security)', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            const result = await service.forgotPassword({ email: 'notfound@example.com' });

            expect(result.message).toContain('If an account exists');
        });

        it('should generate reset token for existing user', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(mockUser);

            const result = await service.forgotPassword({ email: mockUser.email });

            expect(result.message).toContain('If an account exists');
            expect(mockJwtService.sign).toHaveBeenCalled();
        });
    });
});
