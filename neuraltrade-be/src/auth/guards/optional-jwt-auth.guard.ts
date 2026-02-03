import {
  Injectable,
  CanActivate,
  ExecutionContext,
  Logger,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { AuthService } from '../auth.service';
import { UserStatus } from '@prisma/client';

/**
 * Optional JWT Authentication Guard
 * 
 * For endpoints that work with or without authentication.
 * If token is present and valid: attaches user to request
 * If token is missing or invalid: request continues without user
 * 
 * Use case: Public endpoints that show extra data for logged-in users
 */
@Injectable()
export class OptionalJwtAuthGuard implements CanActivate {
  private readonly logger = new Logger(OptionalJwtAuthGuard.name);

  constructor(
    private jwtService: JwtService,
    private authService: AuthService,
  ) { }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const authHeader = request.headers.authorization;

    // Initialize user as null
    request.user = null;
    request.isAuthenticated = false;

    // No auth header - continue without user
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return true;
    }

    const token = authHeader.split(' ')[1];

    // Invalid token format - continue without user
    if (!token || token.length < 10) {
      return true;
    }

    try {
      // Verify token
      const payload = this.jwtService.verify(token, {
        ignoreExpiration: false,
      });

      // Validate payload
      if (!payload.sub || !payload.email) {
        return true;
      }

      // Get full user from database
      const user = await this.authService.validateUser(payload.sub);

      // User not found - continue without user
      if (!user) {
        return true;
      }

      // Only attach user if account is ACTIVE
      if (user.status === UserStatus.ACTIVE) {
        request.user = user;
        request.isAuthenticated = true;
        request.tokenPayload = payload;
      }

      return true;
    } catch (error) {
      // Log only for debugging, don't block request
      this.logger.debug(`Optional auth - token validation failed: ${error.message}`);

      // Continue without user for any error
      return true;
    }
  }
}
