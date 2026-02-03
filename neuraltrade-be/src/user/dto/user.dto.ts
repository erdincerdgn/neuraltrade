import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { UserRole, UserStatus, RiskProfile, GenderType } from '@prisma/client';

/**
 * User Response DTO - for user list/detail responses
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

    @ApiProperty({ enum: GenderType })
    gender: GenderType;

    @ApiProperty({ enum: RiskProfile })
    riskProfile: RiskProfile;

    @ApiProperty()
    tradingEnabled: boolean;

    @ApiProperty()
    emailVerified: boolean;

    @ApiPropertyOptional()
    lastLoginAt: Date | null;

    @ApiProperty()
    createdAt: Date;
}

/**
 * User Profile Stats DTO - Trading platform stats
 */
export class UserProfileStatsDto {
    @ApiProperty({ description: 'Total number of portfolios' })
    totalPortfolios: number;

    @ApiProperty({ description: 'Total portfolio value in USD' })
    totalPortfolioValue: string;

    @ApiProperty({ description: 'Total realized P&L' })
    totalRealizedPnL: string;

    @ApiProperty({ description: 'Total unrealized P&L' })
    totalUnrealizedPnL: string;

    @ApiProperty({ description: 'Number of open positions' })
    openPositions: number;

    @ApiProperty({ description: 'Number of active alerts' })
    activeAlerts: number;

    @ApiProperty({ description: 'Total AI signals received' })
    totalAISignals: number;

    @ApiProperty({ description: 'Number of watchlists' })
    watchlistCount: number;
}

/**
 * User with Trading Stats DTO
 */
export class UserWithStatsDto extends UserResponseDto {
    @ApiProperty({ type: UserProfileStatsDto })
    stats: UserProfileStatsDto;
}

/**
 * Minimal User DTO - for references and lists
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
    profilePhoto: string | null;
}

/**
 * Admin Update User DTO - for admin user management
 */
export class AdminUpdateUserDto {
    @ApiPropertyOptional({ enum: UserStatus })
    status?: UserStatus;

    @ApiPropertyOptional({ enum: UserRole })
    role?: UserRole;

    @ApiPropertyOptional()
    emailVerified?: boolean;

    @ApiPropertyOptional()
    tradingEnabled?: boolean;

    @ApiPropertyOptional({ description: 'Set circuit breaker until time' })
    circuitBreakerUntil?: Date | null;
}