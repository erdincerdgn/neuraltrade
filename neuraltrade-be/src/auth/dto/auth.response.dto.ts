import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';

/**
 * User response DTO - matches Prisma User model
 */
export class UserResponseDto {
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

  // Risk Management
  @ApiProperty({ enum: RiskProfile })
  riskProfile: RiskProfile;

  @ApiPropertyOptional()
  maxDailyLoss: string | null; // Decimal as string

  @ApiPropertyOptional()
  maxPositionSize: string | null; // Decimal as string

  @ApiPropertyOptional()
  maxLeverage: string | null; // Decimal as string

  @ApiProperty()
  tradingEnabled: boolean;

  @ApiPropertyOptional()
  circuitBreakerUntil: Date | null;

  @ApiProperty()
  emailVerified: boolean;

  @ApiPropertyOptional()
  lastLoginAt: Date | null;

  @ApiProperty()
  createdAt: Date;

  @ApiProperty()
  updatedAt: Date;
}

/**
 * Subscription info for auth response
 */
export class SubscriptionInfoDto {
  @ApiProperty()
  planName: string;

  @ApiProperty()
  status: string;

  @ApiProperty()
  billingCycle: string;

  @ApiProperty()
  currentPeriodEnd: Date;

  @ApiPropertyOptional()
  trialEndsAt: Date | null;

  // Feature flags
  @ApiProperty()
  aiSignalsEnabled: boolean;

  @ApiProperty()
  quantumEnabled: boolean;

  @ApiProperty()
  swarmEnabled: boolean;

  @ApiProperty()
  ragEnabled: boolean;

  @ApiProperty()
  apiAccessEnabled: boolean;
}

/**
 * Auth response after login/register
 */
export class AuthResponseDto {
  @ApiProperty()
  accessToken: string;

  @ApiProperty()
  refreshToken?: string;

  @ApiProperty()
  expiresIn: number; // seconds

  @ApiProperty({ type: UserResponseDto })
  user: {
    id: number;
    email: string;
    username: string | null;
    name: string | null;
    surname: string | null;
    role: UserRole;
    status: UserStatus;
    phoneNumber: string | null;
    profilePhoto: string | null;
    gender: GenderType;
    riskProfile: RiskProfile;
    tradingEnabled: boolean;
    emailVerified: boolean;
  };

  @ApiPropertyOptional({ type: SubscriptionInfoDto })
  subscription?: SubscriptionInfoDto | null;
}

/**
 * Token payload for JWT
 */
export class TokenPayloadDto {
  @ApiProperty()
  sub: number; // userId

  @ApiProperty()
  email: string;

  @ApiProperty({ enum: UserRole })
  role: UserRole;

  @ApiProperty()
  iat?: number;

  @ApiProperty()
  exp?: number;
}
