import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsString, IsEmail, IsEnum, IsDateString, MinLength, MaxLength, Matches } from 'class-validator';
import { Transform } from 'class-transformer';
import { GenderType, RiskProfile } from '@prisma/client';

/**
 * Update User Profile DTO - matches Prisma User model
 */
export class UpdateUserProfileDto {
    @ApiPropertyOptional({ example: 'user@example.com' })
    @IsOptional()
    @IsEmail({}, { message: 'Please provide a valid email address' })
    email?: string;

    @ApiPropertyOptional({ example: 'johndoe' })
    @IsOptional()
    @IsString()
    @MinLength(3, { message: 'Username must be at least 3 characters' })
    @MaxLength(30, { message: 'Username cannot exceed 30 characters' })
    @Matches(/^[a-zA-Z0-9_]+$/, {
        message: 'Username can only contain letters, numbers, and underscores',
    })
    username?: string;

    @ApiPropertyOptional({ example: 'John' })
    @IsOptional()
    @IsString()
    @MaxLength(50)
    name?: string;

    @ApiPropertyOptional({ example: 'Doe' })
    @IsOptional()
    @IsString()
    @MaxLength(50)
    surname?: string;

    @ApiPropertyOptional({ example: '5359999999' })
    @IsOptional()
    @IsString()
    @Transform(({ value }) => {
        const cleanPhone = value?.replace(/\D/g, '');
        if (cleanPhone?.length > 10) {
            return cleanPhone.slice(-10);
        }
        return cleanPhone;
    })
    @Matches(/^\d{10,15}$/, {
        message: 'Phone number must be between 10-15 digits',
    })
    phoneNumber?: string;

    @ApiPropertyOptional({ enum: GenderType })
    @IsOptional()
    @IsEnum(GenderType)
    gender?: GenderType;

    @ApiPropertyOptional({ example: '1990-01-15' })
    @IsOptional()
    @IsDateString({}, { message: 'Date of birth must be a valid date' })
    dateOfBirth?: string;

    @ApiPropertyOptional({ description: 'Profile photo URL' })
    @IsOptional()
    @IsString()
    profilePhoto?: string;

    @ApiPropertyOptional({ description: 'Profile banner URL' })
    @IsOptional()
    @IsString()
    profileBanner?: string;

    @ApiPropertyOptional({ description: 'Profile bio/description', maxLength: 500 })
    @IsOptional()
    @IsString()
    @MaxLength(500)
    profileDescription?: string;

    @ApiPropertyOptional({ enum: RiskProfile })
    @IsOptional()
    @IsEnum(RiskProfile)
    riskProfile?: RiskProfile;
}