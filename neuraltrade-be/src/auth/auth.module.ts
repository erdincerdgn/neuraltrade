import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { AuthService } from './auth.service';
import { AuthController } from './auth.controller';
import { PrismaModule } from '../core/prisma/prisma.module';

// Guards
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { AdminAuthGuard } from './guards/admin-auth.guard';
import { OptionalJwtAuthGuard } from './guards/optional-jwt-auth.guard';
import { TradingGuard } from './guards/trading.guard';
import { RolesGuard, SubscriptionGuard } from './guards/roles.guard';

// Phase 8: RBAC
import { RolesGuard as RbacRolesGuard, PermissionService } from './rbac/roles.guard';

// Phase 8: OAuth Strategies
import { GoogleStrategy } from './strategies/google.strategy';
import { GithubStrategy } from './strategies/github.strategy';

// Phase 8: OAuth Controller
import { OAuthController } from './oauth/oauth.controller';

// JWT secret from env with fallback
const jwtSecret = process.env.JWT_SECRET || 'neuraltrade-super-secret-key';

@Module({
  imports: [
    PrismaModule,
    PassportModule.register({ defaultStrategy: 'jwt' }),
    JwtModule.register({
      secret: jwtSecret,
      signOptions: {
        expiresIn: '30d',
        issuer: 'neuraltrade',
        audience: 'neuraltrade-api',
      },
    }),
  ],
  controllers: [AuthController, OAuthController],
  providers: [
    AuthService,
    // Guards
    JwtAuthGuard,
    AdminAuthGuard,
    OptionalJwtAuthGuard,
    TradingGuard,
    RolesGuard,
    SubscriptionGuard,
    // Phase 8: RBAC
    RbacRolesGuard,
    PermissionService,
    // Phase 8: OAuth Strategies
    GoogleStrategy,
    GithubStrategy,
  ],
  exports: [
    AuthService,
    JwtModule,
    PassportModule,
    // Export guards for use in other modules
    JwtAuthGuard,
    AdminAuthGuard,
    OptionalJwtAuthGuard,
    TradingGuard,
    RolesGuard,
    SubscriptionGuard,
    // Phase 8 exports
    RbacRolesGuard,
    PermissionService,
  ],
})
export class AuthModule { }

