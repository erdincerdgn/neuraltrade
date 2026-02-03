import {
  Injectable,
  CanActivate,
  ExecutionContext,
  UnauthorizedException,
  ForbiddenException,
  Logger,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { AuthService } from '../auth.service';
import { UserStatus } from '@prisma/client';

/**
 * JWT Authentication Guard - Strong Security
 * 
 * Security checks:
 * 1. Valid JWT token
 * 2. User exists in database
 * 3. User status is ACTIVE
 * 4. Session is valid (if session-based)
 */
@Injectable()
export class JwtAuthGuard implements CanActivate {
  private readonly logger = new Logger(JwtAuthGuard.name);

  constructor(
    private jwtService: JwtService,
    private authService: AuthService,
  ) { }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const authHeader = request.headers.authorization;

    // 1. Check authorization header
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      this.logger.warn(`Missing or invalid auth header from IP: ${request.ip}`);
      throw new UnauthorizedException('Missing or invalid authorization token');
    }

    const token = authHeader.split(' ')[1];

    if (!token || token.length < 10) {
      throw new UnauthorizedException('Invalid token format');
    }

    try {
      // 2. Verify JWT token
      const payload = this.jwtService.verify(token, {
        ignoreExpiration: false,
      });

      // 3. Validate required payload fields
      if (!payload.sub || !payload.email) {
        this.logger.warn(`Invalid token payload: missing sub or email`);
        throw new UnauthorizedException('Invalid token payload');
      }

      // 4. Get full user from database
      const user = await this.authService.validateUser(payload.sub);

      if (!user) {
        this.logger.warn(`User not found for token sub: ${payload.sub}`);
        throw new UnauthorizedException('User not found');
      }

      // 5. Check user status - CRITICAL SECURITY CHECK
      if (user.status !== UserStatus.ACTIVE) {
        this.logger.warn(`Access denied for user ${user.id} with status: ${user.status}`);

        switch (user.status) {
          case UserStatus.SUSPENDED:
            throw new ForbiddenException('Your account has been suspended. Please contact support.');
          case UserStatus.INACTIVE:
            throw new ForbiddenException('Your account is inactive. Please reactivate your account.');
          case UserStatus.PENDING:
            throw new ForbiddenException('Your account is pending approval.');
          case UserStatus.INVITED:
            throw new ForbiddenException('Please complete your registration first.');
          default:
            throw new ForbiddenException('Account access denied');
        }
      }

      // 6. Validate session if session-based auth is enabled
      if (payload.sessionId) {
        const sessionValid = await this.authService.validateSession(payload.sessionId);
        if (!sessionValid) {
          this.logger.warn(`Invalid session ${payload.sessionId} for user ${user.id}`);
          throw new UnauthorizedException('Session expired or revoked');
        }
      }

      // 7. Attach user to request
      request.user = user;
      request.tokenPayload = payload;

      return true;
    } catch (error) {
      // Handle specific JWT errors
      if (error.name === 'TokenExpiredError') {
        this.logger.debug(`Token expired for request from IP: ${request.ip}`);
        throw new UnauthorizedException('Token has expired');
      }

      if (error.name === 'JsonWebTokenError') {
        this.logger.warn(`Invalid JWT from IP: ${request.ip} - ${error.message}`);
        throw new UnauthorizedException('Invalid token');
      }

      // Re-throw our custom exceptions
      if (error instanceof UnauthorizedException || error instanceof ForbiddenException) {
        throw error;
      }

      // Log unexpected errors
      this.logger.error(`Unexpected auth error: ${error.message}`, error.stack);
      throw new UnauthorizedException('Authentication failed');
    }
  }
}
