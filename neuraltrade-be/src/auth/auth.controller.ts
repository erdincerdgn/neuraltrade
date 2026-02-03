import {
  Body,
  Controller,
  Get,
  Post,
  Put,
  Patch,
  UseGuards,
  Request,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiBody } from '@nestjs/swagger';
import { AuthService } from './auth.service';
import { RegisterDto, CompleteProfileDto } from './dto/register.dto';
import { LoginDto, ChangePasswordDto, ForgotPasswordDto, ResetPasswordDto } from './dto/login.dto';
import { AuthResponseDto } from './dto/auth.response.dto';
import { CurrentUserDto, UpdateRiskSettingsDto } from './dto/user.dto';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { AdminAuthGuard } from './guards/admin-auth.guard';

@ApiTags('Auth')
@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) { }

  // ==========================================
  // PUBLIC ENDPOINTS
  // ==========================================

  @Post('register')
  @HttpCode(HttpStatus.CREATED)
  @ApiOperation({ summary: 'Register a new user account' })
  @ApiResponse({ status: 201, description: 'User registered successfully', type: AuthResponseDto })
  @ApiResponse({ status: 409, description: 'Email or username already exists' })
  @ApiBody({ type: RegisterDto })
  async register(@Body() dto: RegisterDto): Promise<AuthResponseDto> {
    return this.authService.register(dto);
  }

  @Post('login')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Login for frontend users (USER role)' })
  @ApiResponse({ status: 200, description: 'Login successful', type: AuthResponseDto })
  @ApiResponse({ status: 401, description: 'Invalid credentials' })
  @ApiResponse({ status: 403, description: 'Account suspended or locked' })
  @ApiBody({ type: LoginDto })
  async login(@Body() dto: LoginDto): Promise<AuthResponseDto> {
    return this.authService.loginFrontend(dto);
  }

  @Post('login-admin')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Login for admin panel (NEURALTRADE, SUPER_ADMIN roles)' })
  @ApiResponse({ status: 200, description: 'Admin login successful', type: AuthResponseDto })
  @ApiResponse({ status: 401, description: 'Invalid credentials or unauthorized role' })
  @ApiResponse({ status: 403, description: 'Email not verified or account issues' })
  @ApiBody({ type: LoginDto })
  async loginAdmin(@Body() dto: LoginDto): Promise<AuthResponseDto> {
    return this.authService.loginAdmin(dto);
  }

  @Post('forgot-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Request password reset email' })
  @ApiResponse({ status: 200, description: 'Reset email sent if account exists' })
  @ApiBody({ type: ForgotPasswordDto })
  async forgotPassword(@Body() dto: ForgotPasswordDto) {
    return this.authService.forgotPassword(dto);
  }

  @Post('reset-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Reset password with token from email' })
  @ApiResponse({ status: 200, description: 'Password reset successful' })
  @ApiResponse({ status: 400, description: 'Invalid or expired token' })
  @ApiBody({ type: ResetPasswordDto })
  async resetPassword(@Body() dto: ResetPasswordDto) {
    return this.authService.resetPassword(dto);
  }

  // ==========================================
  // AUTHENTICATED ENDPOINTS
  // ==========================================

  @Get('me')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get current user profile with subscription info' })
  @ApiResponse({ status: 200, description: 'Current user data', type: CurrentUserDto })
  @ApiResponse({ status: 401, description: 'Not authenticated' })
  async getCurrentUser(@Request() req): Promise<CurrentUserDto> {
    return this.authService.getCurrentUser(req.user.id);
  }

  @Put('profile')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update user profile' })
  @ApiResponse({ status: 200, description: 'Profile updated successfully' })
  @ApiResponse({ status: 409, description: 'Username already taken' })
  @ApiBody({ type: CompleteProfileDto })
  async updateProfile(@Request() req, @Body() dto: CompleteProfileDto) {
    return this.authService.updateProfile(req.user.id, dto);
  }

  @Patch('risk-settings')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update risk management settings (risk profile, limits)' })
  @ApiResponse({ status: 200, description: 'Risk settings updated' })
  @ApiBody({ type: UpdateRiskSettingsDto })
  async updateRiskSettings(@Request() req, @Body() dto: UpdateRiskSettingsDto) {
    return this.authService.updateRiskSettings(req.user.id, dto);
  }

  @Post('change-password')
  @UseGuards(JwtAuthGuard)
  @HttpCode(HttpStatus.OK)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Change password for logged in user' })
  @ApiResponse({ status: 200, description: 'Password changed successfully' })
  @ApiResponse({ status: 400, description: 'Passwords do not match' })
  @ApiResponse({ status: 401, description: 'Current password incorrect' })
  @ApiBody({ type: ChangePasswordDto })
  async changePassword(@Request() req, @Body() dto: ChangePasswordDto) {
    return this.authService.changePassword(req.user.id, dto);
  }

  // ==========================================
  // ADMIN ONLY ENDPOINTS
  // ==========================================

  @Get('admin/users/me')
  @UseGuards(AdminAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get admin user profile (admin panel)' })
  @ApiResponse({ status: 200, description: 'Admin user data' })
  async getAdminUser(@Request() req): Promise<CurrentUserDto> {
    return this.authService.getCurrentUser(req.user.id);
  }
}
