import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext, UnauthorizedException, ForbiddenException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { JwtAuthGuard } from './jwt-auth.guard';
import { AuthService } from '../auth.service';
import { UserStatus } from '@prisma/client';

describe('JwtAuthGuard', () => {
    let guard: JwtAuthGuard;

    const mockUser = {
        id: 1,
        email: 'test@example.com',
        status: UserStatus.ACTIVE,
        tradingEnabled: true,
    };

    const mockJwtService = {
        verify: jest.fn(),
    };

    const mockAuthService = {
        validateUser: jest.fn(),
        validateSession: jest.fn(),
    };

    const createMockExecutionContext = (headers: Record<string, string> = {}): ExecutionContext => {
        return {
            switchToHttp: () => ({
                getRequest: () => ({
                    headers: {
                        authorization: headers.authorization || '',
                        ...headers,
                    },
                }),
            }),
            getHandler: () => jest.fn(),
            getClass: () => jest.fn(),
        } as unknown as ExecutionContext;
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                JwtAuthGuard,
                { provide: JwtService, useValue: mockJwtService },
                { provide: AuthService, useValue: mockAuthService },
            ],
        }).compile();

        guard = module.get<JwtAuthGuard>(JwtAuthGuard);

        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(guard).toBeDefined();
    });

    describe('canActivate', () => {
        it('should throw UnauthorizedException if no token provided', async () => {
            const context = createMockExecutionContext();

            await expect(guard.canActivate(context)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if token is invalid', async () => {
            const context = createMockExecutionContext({ authorization: 'Bearer invalid_token' });
            mockJwtService.verify.mockImplementation(() => {
                throw new Error('Invalid token');
            });

            await expect(guard.canActivate(context)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if user not found', async () => {
            const context = createMockExecutionContext({ authorization: 'Bearer valid_token' });
            mockJwtService.verify.mockReturnValue({ sub: 1, email: 'test@example.com' });
            mockAuthService.validateUser.mockResolvedValue(null);

            await expect(guard.canActivate(context)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw ForbiddenException if user is not active', async () => {
            const context = createMockExecutionContext({ authorization: 'Bearer valid_token' });
            mockJwtService.verify.mockReturnValue({ sub: 1, email: 'test@example.com' });
            mockAuthService.validateUser.mockResolvedValue({
                ...mockUser,
                status: UserStatus.SUSPENDED,
            });

            await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
        });

        it('should return true and attach user to request if valid', async () => {
            const mockRequest = {
                headers: { authorization: 'Bearer valid_token' },
                user: null,
            };
            const context = {
                switchToHttp: () => ({
                    getRequest: () => mockRequest,
                }),
                getHandler: () => jest.fn(),
                getClass: () => jest.fn(),
            } as unknown as ExecutionContext;

            mockJwtService.verify.mockReturnValue({ sub: 1, email: 'test@example.com' });
            mockAuthService.validateUser.mockResolvedValue(mockUser);

            const result = await guard.canActivate(context);

            expect(result).toBe(true);
            expect(mockRequest.user).toEqual(mockUser);
        });
    });
});
