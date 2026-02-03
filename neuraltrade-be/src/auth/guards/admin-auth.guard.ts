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
import { UserRole, UserStatus } from '@prisma/client';

/**
 * Admin Panel Authentication Guard - Maximum Security
 * 
 * Security checks:
 * 1. Valid JWT token with 'admin' tokenType
 * 2. User exists and is ACTIVE
 * 3. User has NEURALTRADE or SUPER_ADMIN role
 * 4. Email is verified
 * 5. Session is valid
 * 6. IP whitelist (optional)
 */
@Injectable()
export class AdminAuthGuard implements CanActivate {
  private readonly logger = new Logger(AdminAuthGuard.name);

  // Optional: IP whitelist for admin access
  private readonly ADMIN_IP_WHITELIST: string[] = [];
  private readonly ENABLE_IP_WHITELIST = false;

  constructor(
    private jwtService: JwtService,
    private authService: AuthService,
  ) { }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const authHeader = request.headers.authorization;
    const clientIp = request.ip || request.connection?.remoteAddress;

    // 1. IP Whitelist check (if enabled)
    if (this.ENABLE_IP_WHITELIST && this.ADMIN_IP_WHITELIST.length > 0) {
      if (!this.ADMIN_IP_WHITELIST.includes(clientIp)) {
        this.logger.warn(`Admin access attempt from non-whitelisted IP: ${clientIp}`);
        throw new ForbiddenException('Admin access not allowed from this location');
      }
    }

    // 2. Check authorization header
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      this.logger.warn(`Admin panel: Missing auth header from IP: ${clientIp}`);
      throw new UnauthorizedException('Missing or invalid authorization token');
    }

    const token = authHeader.split(' ')[1];

    if (!token || token.length < 10) {
      throw new UnauthorizedException('Invalid token format');
    }

    try {
      // 3. Verify JWT token
      const payload = this.jwtService.verify(token, {
        ignoreExpiration: false,
      });

      // 4. CRITICAL: Check admin token type
      if (payload.tokenType !== 'admin') {
        this.logger.warn(`Non-admin token used for admin panel access. User: ${payload.sub}`);
        throw new UnauthorizedException('Admin panel access requires admin authentication');
      }

      // 5. Validate required payload fields
      if (!payload.sub || !payload.email) {
        throw new UnauthorizedException('Invalid token payload');
      }

      // 6. Get full user from database
      const user = await this.authService.validateUser(payload.sub);

      if (!user) {
        this.logger.warn(`Admin panel: User not found for ID: ${payload.sub}`);
        throw new UnauthorizedException('User not found');
      }

      // 7. Check user status - Must be ACTIVE
      if (user.status !== UserStatus.ACTIVE) {
        this.logger.warn(`Admin panel access denied - User ${user.id} status: ${user.status}`);
        throw new ForbiddenException(`Account is ${user.status.toLowerCase()}. Admin access denied.`);
      }

      // 8. CRITICAL: Check admin role
      const allowedRoles: UserRole[] = [UserRole.NEURALTRADE, UserRole.SUPER_ADMIN];

      if (!allowedRoles.includes(user.role)) {
        this.logger.warn(`Admin panel access denied - User ${user.id} role: ${user.role} from IP: ${clientIp}`);
        throw new ForbiddenException('You do not have permission to access the admin panel');
      }

      // 9. Check email verification for admin
      if (!user.emailVerified) {
        this.logger.warn(`Admin panel: Unverified email for user ${user.id}`);
        throw new ForbiddenException('Email verification required for admin access');
      }

      // 10. Validate session
      if (payload.sessionId) {
        const sessionValid = await this.authService.validateSession(payload.sessionId);
        if (!sessionValid) {
          this.logger.warn(`Admin panel: Invalid session ${payload.sessionId} for user ${user.id}`);
          throw new UnauthorizedException('Admin session expired or revoked');
        }
      }

      // 11. Log successful admin access
      this.logger.log(`Admin panel access granted - User: ${user.email}, Role: ${user.role}, IP: ${clientIp}`);

      // 12. Attach user to request
      request.user = user;
      request.tokenPayload = payload;
      request.isAdmin = true;

      return true;
    } catch (error) {
      // Handle specific JWT errors
      if (error.name === 'TokenExpiredError') {
        this.logger.debug(`Admin token expired from IP: ${clientIp}`);
        throw new UnauthorizedException('Admin token has expired');
      }

      if (error.name === 'JsonWebTokenError') {
        this.logger.warn(`Invalid admin JWT from IP: ${clientIp}`);
        throw new UnauthorizedException('Invalid admin token');
      }

      // Re-throw our custom exceptions
      if (error instanceof UnauthorizedException || error instanceof ForbiddenException) {
        throw error;
      }

      // Log unexpected errors
      this.logger.error(`Admin auth error: ${error.message}`, error.stack);
      throw new UnauthorizedException('Admin authentication failed');
    }
  }
}
