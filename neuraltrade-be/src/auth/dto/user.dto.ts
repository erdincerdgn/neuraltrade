import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';

/**
 * Base User DTO - matches Prisma User model for internal use
 */
export class UserDto {
  @ApiProperty()
  id: number;

  @ApiProperty()
  email: string;

  @ApiPropertyOptional()
  username: string | null;

  @ApiPropertyOptional()
  name: string | null;

  @ApiPropertyOptional()
  surname: string | null;

  @ApiProperty({ enum: UserRole })
  role: UserRole;

  @ApiProperty({ enum: UserStatus })
  status: UserStatus;

  @ApiPropertyOptional()
  phoneNumber: string | null;

  @ApiPropertyOptional()
  profilePhoto: string | null;

  @ApiPropertyOptional()
  profileDescription: string | null;

  @ApiPropertyOptional()
  dateOfBirth: Date | null;

  @ApiPropertyOptional()
  age: number | null;

  @ApiProperty({ enum: GenderType })
  gender: GenderType;

  // Risk Management Fields
  @ApiProperty({ enum: RiskProfile })
  riskProfile: RiskProfile;

  @ApiPropertyOptional({ description: 'Maximum daily loss limit (Circuit Breaker)' })
  maxDailyLoss: string | null; // Decimal as string

  @ApiPropertyOptional({ description: 'Maximum position size allowed' })
  maxPositionSize: string | null;

  @ApiPropertyOptional({ description: 'Maximum leverage allowed' })
  maxLeverage: string | null;

  @ApiProperty({ description: 'Whether trading is enabled (Circuit Breaker state)' })
  tradingEnabled: boolean;

  @ApiPropertyOptional({ description: 'Trading disabled until this time (Circuit Breaker)' })
  circuitBreakerUntil: Date | null;

  // Account Status
  @ApiProperty()
  emailVerified: boolean;

  @ApiProperty()
  failedLoginAttempts: number;

  @ApiPropertyOptional()
  lastLoginAt: Date | null;

  @ApiPropertyOptional()
  lastPasswordChangeAt: Date | null;

  @ApiProperty()
  createdAt: Date;

  @ApiProperty()
  updatedAt: Date;
}

/**
 * Minimal User DTO - for lists and references
 */
export class UserMinimalDto {
  @ApiProperty()
  id: number;

  @ApiProperty()
  email: string;

  @ApiPropertyOptional()
  username: string | null;

  @ApiPropertyOptional()
  name: string | null;

  @ApiPropertyOptional()
  surname: string | null;

  @ApiPropertyOptional()
  profilePhoto: string | null;
}

/**
 * User with subscription info
 */
export class UserWithSubscriptionDto extends UserDto {
  @ApiPropertyOptional()
  subscription?: {
    planName: string;
    status: string;
    currentPeriodEnd: Date;
    aiSignalsEnabled: boolean;
    quantumEnabled: boolean;
    swarmEnabled: boolean;
    ragEnabled: boolean;
    apiAccessEnabled: boolean;
  } | null;
}

/**
 * Current user response (for /me endpoint)
 */
export class CurrentUserDto {
  @ApiProperty()
  id: number;

  @ApiProperty()
  email: string;

  @ApiPropertyOptional()
  username: string | null;

  @ApiPropertyOptional()
  name: string | null;

  @ApiPropertyOptional()
  surname: string | null;

  @ApiProperty({ enum: UserRole })
  role: UserRole;

  @ApiProperty({ enum: UserStatus })
  status: UserStatus;

  @ApiPropertyOptional()
  profilePhoto: string | null;

  @ApiProperty({ enum: RiskProfile })
  riskProfile: RiskProfile;

  @ApiProperty()
  tradingEnabled: boolean;

  @ApiProperty()
  emailVerified: boolean;

  @ApiPropertyOptional()
  subscription?: {
    planName: string;
    status: string;
    currentPeriodEnd: Date;
    features: {
      aiSignals: boolean;
      quantum: boolean;
      swarm: boolean;
      rag: boolean;
      apiAccess: boolean;
      maxPortfolios: number;
      maxWatchlists: number;
      maxAlerts: number;
    };
  } | null;
}

/**
 * Update user risk settings DTO
 */
export class UpdateRiskSettingsDto {
  @ApiPropertyOptional({ enum: RiskProfile })
  riskProfile?: RiskProfile;

  @ApiPropertyOptional({ description: 'Maximum daily loss limit (e.g., "1000.00")' })
  maxDailyLoss?: string;

  @ApiPropertyOptional({ description: 'Maximum position size (e.g., "10000.00")' })
  maxPositionSize?: string;

  @ApiPropertyOptional({ description: 'Maximum leverage (e.g., "5.00")' })
  maxLeverage?: string;
}
