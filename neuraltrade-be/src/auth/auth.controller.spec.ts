import { Test, TestingModule } from '@nestjs/testing';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtService } from '@nestjs/jwt';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';

describe('AuthController', () => {
    let controller: AuthController;

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
        gender: GenderType.Unspecified,
        riskProfile: RiskProfile.MODERATE,
        tradingEnabled: true,
        emailVerified: true,
    };

    const mockAuthResponse = {
        accessToken: 'mock_jwt_token',
        expiresIn: 2592000,
        user: mockUser,
    };

    const mockAuthService = {
        register: jest.fn().mockResolvedValue(mockAuthResponse),
        loginFrontend: jest.fn().mockResolvedValue(mockAuthResponse),
        loginAdmin: jest.fn().mockResolvedValue(mockAuthResponse),
        forgotPassword: jest.fn().mockResolvedValue({ message: 'Email sent' }),
        resetPassword: jest.fn().mockResolvedValue({ message: 'Password reset' }),
        getCurrentUser: jest.fn().mockResolvedValue(mockUser),
        updateProfile: jest.fn().mockResolvedValue(mockUser),
        updateRiskSettings: jest.fn().mockResolvedValue(mockUser),
        changePassword: jest.fn().mockResolvedValue({ message: 'Password changed' }),
        validateUser: jest.fn().mockResolvedValue(mockUser),
    };

    const mockJwtService = {
        sign: jest.fn().mockReturnValue('mock_token'),
        verify: jest.fn().mockReturnValue({ sub: 1, email: 'test@example.com' }),
    };

    // Mock guard that always allows
    const mockJwtAuthGuard = {
        canActivate: jest.fn().mockReturnValue(true),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [AuthController],
            providers: [
                { provide: AuthService, useValue: mockAuthService },
                { provide: JwtService, useValue: mockJwtService },
            ],
        })
            .overrideGuard(JwtAuthGuard)
            .useValue(mockJwtAuthGuard)
            .compile();

        controller = module.get<AuthController>(AuthController);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
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

        it('should register a new user', async () => {
            const result = await controller.register(registerDto);

            expect(result).toEqual(mockAuthResponse);
            expect(mockAuthService.register).toHaveBeenCalledWith(registerDto);
        });
    });

    // ==========================================
    // LOGIN TESTS
    // ==========================================

    describe('login', () => {
        const loginDto = {
            email: 'test@example.com',
            password: 'Password123!',
        };

        it('should login a user', async () => {
            const result = await controller.login(loginDto);

            expect(result).toEqual(mockAuthResponse);
            expect(mockAuthService.loginFrontend).toHaveBeenCalledWith(loginDto);
        });
    });

    describe('loginAdmin', () => {
        const loginDto = {
            email: 'admin@example.com',
            password: 'AdminPassword123!',
        };

        it('should login an admin user', async () => {
            const result = await controller.loginAdmin(loginDto);

            expect(result).toEqual(mockAuthResponse);
            expect(mockAuthService.loginAdmin).toHaveBeenCalledWith(loginDto);
        });
    });

    // ==========================================
    // PASSWORD RESET TESTS
    // ==========================================

    describe('forgotPassword', () => {
        it('should request password reset', async () => {
            const dto = { email: 'test@example.com' };

            const result = await controller.forgotPassword(dto);

            expect(result.message).toBe('Email sent');
            expect(mockAuthService.forgotPassword).toHaveBeenCalledWith(dto);
        });
    });

    describe('resetPassword', () => {
        it('should reset password with token', async () => {
            const dto = {
                token: 'reset_token',
                newPassword: 'NewPassword123!',
                confirmPassword: 'NewPassword123!',
            };

            const result = await controller.resetPassword(dto);

            expect(result.message).toBe('Password reset');
            expect(mockAuthService.resetPassword).toHaveBeenCalledWith(dto);
        });
    });

    // ==========================================
    // AUTHENTICATED ENDPOINTS TESTS
    // ==========================================

    describe('getCurrentUser', () => {
        it('should return current user', async () => {
            const mockReq = { user: { id: 1 } };

            const result = await controller.getCurrentUser(mockReq);

            expect(result).toEqual(mockUser);
            expect(mockAuthService.getCurrentUser).toHaveBeenCalledWith(1);
        });
    });

    describe('updateProfile', () => {
        it('should update user profile', async () => {
            const mockReq = { user: { id: 1 } };
            const updateDto = { name: 'Updated', surname: 'Name' };

            const result = await controller.updateProfile(mockReq, updateDto);

            expect(result).toEqual(mockUser);
            expect(mockAuthService.updateProfile).toHaveBeenCalledWith(1, updateDto);
        });
    });

    describe('changePassword', () => {
        it('should change password', async () => {
            const mockReq = { user: { id: 1 } };
            const dto = {
                currentPassword: 'OldPassword123!',
                newPassword: 'NewPassword456!',
                confirmPassword: 'NewPassword456!',
            };

            const result = await controller.changePassword(mockReq, dto);

            expect(result.message).toBe('Password changed');
            expect(mockAuthService.changePassword).toHaveBeenCalledWith(1, dto);
        });
    });
});
