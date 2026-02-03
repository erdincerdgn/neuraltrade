import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsEmail, IsString, MinLength, IsOptional, IsBoolean } from 'class-validator';

/**
 * Login request DTO
 */
export class LoginDto {
  @ApiProperty({ example: 'user@example.com' })
  @IsEmail({}, { message: 'Please provide a valid email address' })
  email: string;

  @ApiProperty({ example: 'password123', minLength: 6 })
  @IsString()
  @MinLength(6, { message: 'Password must be at least 6 characters' })
  password: string;

  @ApiPropertyOptional({ description: 'Remember me for extended session' })
  @IsOptional()
  @IsBoolean()
  rememberMe?: boolean;
}

/**
 * Refresh token request DTO
 */
export class RefreshTokenDto {
  @ApiProperty()
  @IsString()
  refreshToken: string;
}

/**
 * Forgot password request DTO
 */
export class ForgotPasswordDto {
  @ApiProperty({ example: 'user@example.com' })
  @IsEmail({}, { message: 'Please provide a valid email address' })
  email: string;
}

/**
 * Reset password request DTO
 */
export class ResetPasswordDto {
  @ApiProperty()
  @IsString()
  token: string;

  @ApiProperty({ minLength: 6 })
  @IsString()
  @MinLength(6, { message: 'Password must be at least 6 characters' })
  newPassword: string;

  @ApiProperty({ minLength: 6 })
  @IsString()
  @MinLength(6)
  confirmPassword: string;
}

/**
 * Change password request DTO
 */
export class ChangePasswordDto {
  @ApiProperty()
  @IsString()
  currentPassword: string;

  @ApiProperty({ minLength: 6 })
  @IsString()
  @MinLength(6, { message: 'Password must be at least 6 characters' })
  newPassword: string;

  @ApiProperty({ minLength: 6 })
  @IsString()
  @MinLength(6)
  confirmPassword: string;
}

/**
 * Verify email request DTO
 */
export class VerifyEmailDto {
  @ApiProperty()
  @IsString()
  token: string;
}
