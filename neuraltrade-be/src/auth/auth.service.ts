import {
  Injectable,
  UnauthorizedException,
  ConflictException,
  ForbiddenException,
  BadRequestException,
  Logger,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { PrismaService } from '../core/prisma/prisma.service';
import { RegisterDto, CompleteProfileDto } from './dto/register.dto';
import { LoginDto, ChangePasswordDto, ForgotPasswordDto, ResetPasswordDto } from './dto/login.dto';
import { AuthResponseDto, UserResponseDto, SubscriptionInfoDto } from './dto/auth.response.dto';
import { CurrentUserDto, UpdateRiskSettingsDto } from './dto/user.dto';
import * as bcrypt from 'bcrypt';
import { UserRole, UserStatus, RiskProfile, GenderType, User } from '@prisma/client';
import { v4 as uuidv4 } from 'uuid';

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);

  constructor(
    private prisma: PrismaService,
    private jwtService: JwtService,
  ) { }

  // ==========================================
  // REGISTRATION
  // ==========================================

  async register(dto: RegisterDto): Promise<AuthResponseDto> {
    const { email, name, surname, phoneNumber, password, username, gender, riskProfile, dateOfBirth, profileDescription } = dto;

    // Check existing user
    const existingUser = await this.prisma.user.findFirst({
      where: {
        OR: [
          { email: dto.email },
          ...(username ? [{ username }] : []),
        ],
      },
    });

    if (existingUser) {
      if (existingUser.email === email) {
        throw new ConflictException('Email already exists');
      }
      if (username && existingUser.username === username) {
        throw new ConflictException('Username already taken');
      }
    }

    const hashedPassword = await bcrypt.hash(password, 12);

    const user = await this.prisma.user.create({
      data: {
        username: username || uuidv4(),
        name,
        surname,
        phoneNumber,
        email,
        password: hashedPassword,
        role: UserRole.USER,
        status: UserStatus.ACTIVE,
        gender: gender || GenderType.Unspecified,
        riskProfile: riskProfile || RiskProfile.MODERATE,
        dateOfBirth: dateOfBirth ? new Date(dateOfBirth) : null,
        profileDescription,
        tradingEnabled: true,
        emailVerified: false,
      },
    });

    this.logger.log(`New user registered: ${user.email}`);

    const token = this.generateToken(user.id, user.email, user.role, 'frontend');

    return {
      accessToken: token,
      expiresIn: 30 * 24 * 60 * 60, // 30 days in seconds
      user: this.mapUserToResponse(user),
    };
  }

  // ==========================================
  // LOGIN
  // ==========================================

  private async authenticate(dto: LoginDto): Promise<User> {
    const user = await this.prisma.user.findUnique({
      where: { email: dto.email },
    });

    if (!user) {
      this.logger.warn(`Login attempt with non-existent email: ${dto.email}`);
      throw new UnauthorizedException('Invalid credentials');
    }

    // Check account status before password
    if (user.status === UserStatus.SUSPENDED) {
      throw new ForbiddenException('Your account has been suspended. Please contact support.');
    }

    if (user.status === UserStatus.INACTIVE) {
      throw new ForbiddenException('Your account is inactive. Please reactivate your account.');
    }

    // Check failed login attempts (brute force protection)
    if (user.failedLoginAttempts >= 5) {
      throw new ForbiddenException('Account locked due to too many failed login attempts. Please reset your password.');
    }

    const isPasswordValid = await bcrypt.compare(dto.password, user.password);

    if (!isPasswordValid) {
      // Increment failed login attempts
      await this.prisma.user.update({
        where: { id: user.id },
        data: { failedLoginAttempts: { increment: 1 } },
      });

      this.logger.warn(`Failed login attempt for: ${dto.email}`);
      throw new UnauthorizedException('Invalid credentials');
    }

    // Reset failed attempts and update last login
    await this.prisma.user.update({
      where: { id: user.id },
      data: {
        failedLoginAttempts: 0,
        lastLoginAt: new Date(),
      },
    });

    return user;
  }

  // Frontend login: USER only
  async loginFrontend(dto: LoginDto): Promise<AuthResponseDto> {
    const user = await this.authenticate(dto);

    if (user.role !== UserRole.USER) {
      throw new UnauthorizedException('Please use admin panel for admin accounts');
    }

    const token = this.generateToken(user.id, user.email, user.role, 'frontend');
    const subscription = await this.getUserSubscription(user.id);

    this.logger.log(`Frontend login: ${user.email}`);

    return {
      accessToken: token,
      expiresIn: 30 * 24 * 60 * 60,
      user: this.mapUserToResponse(user),
      subscription,
    };
  }

  // Admin panel login: NEURALTRADE and SUPER_ADMIN only
  async loginAdmin(dto: LoginDto): Promise<AuthResponseDto> {
    const user = await this.authenticate(dto);

    // Check admin role - direct comparison avoids TypeScript strict type issues
    if (user.role !== UserRole.NEURALTRADE && user.role !== UserRole.SUPER_ADMIN) {
      this.logger.warn(`Admin login attempt by non-admin user: ${user.email}`);
      throw new UnauthorizedException('Admin access restricted to authorized accounts only');
    }

    // Admin requires email verification
    if (!user.emailVerified) {
      throw new ForbiddenException('Please verify your email before accessing admin panel');
    }

    const token = this.generateToken(user.id, user.email, user.role, 'admin');

    this.logger.log(`Admin login: ${user.email} (${user.role})`);

    return {
      accessToken: token,
      expiresIn: 24 * 60 * 60, // 24 hours for admin
      user: this.mapUserToResponse(user),
    };
  }

  // ==========================================
  // USER VALIDATION (for Guards)
  // ==========================================

  async validateUser(userId: number): Promise<User | null> {
    return this.prisma.user.findUnique({
      where: { id: userId },
    });
  }

  async validateSession(sessionId: number): Promise<boolean> {
    const session = await this.prisma.session.findUnique({
      where: { id: sessionId },
    });

    if (!session) return false;
    if (new Date() > session.expiresAt) return false;

    return true;
  }

  // ==========================================
  // CURRENT USER / PROFILE
  // ==========================================

  async getCurrentUser(userId: number): Promise<CurrentUserDto> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      include: {
        subscription: {
          include: { plan: true },
        },
      },
    });

    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    return {
      id: user.id,
      email: user.email,
      username: user.username,
      name: user.name,
      surname: user.surname,
      role: user.role,
      status: user.status,
      profilePhoto: user.profilePhoto,
      riskProfile: user.riskProfile,
      tradingEnabled: user.tradingEnabled,
      emailVerified: user.emailVerified,
      subscription: user.subscription ? {
        planName: user.subscription.plan.name,
        status: user.subscription.status,
        currentPeriodEnd: user.subscription.currentPeriodEnd,
        features: {
          aiSignals: user.subscription.plan.aiSignalsEnabled,
          quantum: user.subscription.plan.quantumEnabled,
          swarm: user.subscription.plan.swarmEnabled,
          rag: user.subscription.plan.ragEnabled,
          apiAccess: user.subscription.plan.apiAccessEnabled,
          maxPortfolios: user.subscription.plan.maxPortfolios,
          maxWatchlists: user.subscription.plan.maxWatchlists,
          maxAlerts: user.subscription.plan.maxAlerts,
        },
      } : null,
    };
  }

  async updateProfile(userId: number, dto: CompleteProfileDto): Promise<UserResponseDto> {
    const updateData: any = {};

    if (dto.username) {
      // Check if username is taken
      const existing = await this.prisma.user.findFirst({
        where: { username: dto.username, id: { not: userId } },
      });
      if (existing) {
        throw new ConflictException('Username already taken');
      }
      updateData.username = dto.username;
    }

    if (dto.name !== undefined) updateData.name = dto.name;
    if (dto.surname !== undefined) updateData.surname = dto.surname;
    if (dto.phoneNumber !== undefined) updateData.phoneNumber = dto.phoneNumber;
    if (dto.gender !== undefined) updateData.gender = dto.gender;
    if (dto.dateOfBirth !== undefined) updateData.dateOfBirth = new Date(dto.dateOfBirth);
    if (dto.riskProfile !== undefined) updateData.riskProfile = dto.riskProfile;
    if (dto.profileDescription !== undefined) updateData.profileDescription = dto.profileDescription;
    if (dto.profilePhoto !== undefined) updateData.profilePhoto = dto.profilePhoto;

    const user = await this.prisma.user.update({
      where: { id: userId },
      data: updateData,
    });

    return this.mapUserToResponse(user);
  }

  async updateRiskSettings(userId: number, dto: UpdateRiskSettingsDto): Promise<UserResponseDto> {
    const updateData: any = {};

    if (dto.riskProfile !== undefined) updateData.riskProfile = dto.riskProfile;
    if (dto.maxDailyLoss !== undefined) updateData.maxDailyLoss = dto.maxDailyLoss;
    if (dto.maxPositionSize !== undefined) updateData.maxPositionSize = dto.maxPositionSize;
    if (dto.maxLeverage !== undefined) updateData.maxLeverage = dto.maxLeverage;

    const user = await this.prisma.user.update({
      where: { id: userId },
      data: updateData,
    });

    this.logger.log(`Risk settings updated for user ${userId}`);

    return this.mapUserToResponse(user);
  }

  // ==========================================
  // PASSWORD MANAGEMENT
  // ==========================================

  async changePassword(userId: number, dto: ChangePasswordDto): Promise<{ message: string }> {
    if (dto.newPassword !== dto.confirmPassword) {
      throw new BadRequestException('Passwords do not match');
    }

    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    const isCurrentValid = await bcrypt.compare(dto.currentPassword, user.password);
    if (!isCurrentValid) {
      throw new UnauthorizedException('Current password is incorrect');
    }

    const hashedPassword = await bcrypt.hash(dto.newPassword, 12);

    await this.prisma.user.update({
      where: { id: userId },
      data: {
        password: hashedPassword,
        lastPasswordChangeAt: new Date(),
      },
    });

    this.logger.log(`Password changed for user ${userId}`);

    return { message: 'Password changed successfully' };
  }

  // ==========================================
  // FORGOT / RESET PASSWORD
  // ==========================================

  async forgotPassword(dto: ForgotPasswordDto): Promise<{ message: string }> {
    const user = await this.prisma.user.findUnique({
      where: { email: dto.email },
    });

    // Always return success to prevent email enumeration
    if (!user) {
      this.logger.debug(`Forgot password requested for non-existent email: ${dto.email}`);
      return { message: 'If an account exists with this email, you will receive a password reset link.' };
    }

    // Check if user is active
    if (user.status !== UserStatus.ACTIVE) {
      this.logger.warn(`Forgot password for inactive user: ${user.email}`);
      return { message: 'If an account exists with this email, you will receive a password reset link.' };
    }

    // Generate reset token (expires in 1 hour)
    const resetToken = this.jwtService.sign(
      {
        sub: user.id,
        email: user.email,
        type: 'password_reset',
      },
      { expiresIn: '1h' }
    );

    // TODO: Send email with reset link
    // In production, you would use a mail service like:
    // await this.mailService.sendPasswordResetEmail(user.email, resetToken);

    this.logger.log(`Password reset token generated for user: ${user.email}`);

    // For development, log the token (remove in production!)
    this.logger.debug(`Reset token for ${user.email}: ${resetToken}`);

    return {
      message: 'If an account exists with this email, you will receive a password reset link.',
      // Remove this in production - only for testing!
      // resetToken: resetToken, 
    };
  }

  async resetPassword(dto: ResetPasswordDto): Promise<{ message: string }> {
    // Validate passwords match
    if (dto.newPassword !== dto.confirmPassword) {
      throw new BadRequestException('Passwords do not match');
    }

    try {
      // Verify reset token
      const payload = this.jwtService.verify(dto.token);

      // Check token type
      if (payload.type !== 'password_reset') {
        throw new BadRequestException('Invalid reset token');
      }

      // Get user
      const user = await this.prisma.user.findUnique({
        where: { id: payload.sub },
      });

      if (!user) {
        throw new BadRequestException('Invalid reset token');
      }

      // Check email matches (extra security)
      if (user.email !== payload.email) {
        throw new BadRequestException('Invalid reset token');
      }

      // Hash new password
      const hashedPassword = await bcrypt.hash(dto.newPassword, 12);

      // Update password and reset failed login attempts
      await this.prisma.user.update({
        where: { id: user.id },
        data: {
          password: hashedPassword,
          lastPasswordChangeAt: new Date(),
          failedLoginAttempts: 0, // Reset lockout
        },
      });

      this.logger.log(`Password reset successful for user: ${user.email}`);

      return { message: 'Password has been reset successfully. You can now log in with your new password.' };
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        throw new BadRequestException('Reset token has expired. Please request a new password reset.');
      }
      if (error.name === 'JsonWebTokenError') {
        throw new BadRequestException('Invalid reset token');
      }
      if (error instanceof BadRequestException) {
        throw error;
      }
      this.logger.error(`Password reset error: ${error.message}`);
      throw new BadRequestException('Failed to reset password');
    }
  }

  // ==========================================
  // HELPERS
  // ==========================================

  private generateToken(
    userId: number,
    email: string,
    role: UserRole,
    tokenType: 'frontend' | 'admin',
  ): string {
    return this.jwtService.sign({
      sub: userId,
      email,
      role,
      tokenType,
    });
  }

  private mapUserToResponse(user: User): any {
    return {
      id: user.id,
      email: user.email,
      username: user.username,
      name: user.name,
      surname: user.surname,
      role: user.role,
      status: user.status,
      phoneNumber: user.phoneNumber,
      profilePhoto: user.profilePhoto,
      gender: user.gender,
      riskProfile: user.riskProfile,
      tradingEnabled: user.tradingEnabled,
      emailVerified: user.emailVerified,
    };
  }

  private async getUserSubscription(userId: number): Promise<SubscriptionInfoDto | null> {
    const subscription = await this.prisma.subscription.findUnique({
      where: { userId },
      include: { plan: true },
    });

    if (!subscription) return null;

    return {
      planName: subscription.plan.name,
      status: subscription.status,
      billingCycle: subscription.billingCycle,
      currentPeriodEnd: subscription.currentPeriodEnd,
      trialEndsAt: subscription.trialEndsAt,
      aiSignalsEnabled: subscription.plan.aiSignalsEnabled,
      quantumEnabled: subscription.plan.quantumEnabled,
      swarmEnabled: subscription.plan.swarmEnabled,
      ragEnabled: subscription.plan.ragEnabled,
      apiAccessEnabled: subscription.plan.apiAccessEnabled,
    };
  }

  // ==========================================
  // OAUTH
  // ==========================================

  async findOrCreateOAuthUser(data: {
    provider: string;
    providerId: string;
    email?: string;
    firstName?: string;
    lastName?: string;
    picture?: string;
    username?: string;
  }): Promise<User> {
    // Try to find by email
    if (data.email) {
      const existingUser = await this.prisma.user.findUnique({
        where: { email: data.email },
      });

      if (existingUser) {
        // Update last login
        const user = await this.prisma.user.update({
          where: { id: existingUser.id },
          data: {
            lastLoginAt: new Date(),
            emailVerified: true,
            profilePhoto: data.picture || existingUser.profilePhoto,
          },
        });
        this.logger.log(`OAuth login: ${user.email} (${data.provider})`);
        return user;
      }
    }

    // Create new user
    const user = await this.prisma.user.create({
      data: {
        email: data.email || `${data.providerId}@${data.provider}.oauth`,
        username: data.username || `${data.provider}_${data.providerId}`,
        name: data.firstName || 'User',
        surname: data.lastName || '',
        password: uuidv4(), // Random password (won't be used for OAuth)
        role: UserRole.USER,
        status: UserStatus.ACTIVE,
        profilePhoto: data.picture,
        emailVerified: !!data.email,
        tradingEnabled: true,
        riskProfile: RiskProfile.MODERATE,
        gender: GenderType.Unspecified,
      },
    });

    this.logger.log(`New OAuth user created: ${user.email} (${data.provider})`);
    return user;
  }

  async generateTokens(user: User): Promise<{ accessToken: string; refreshToken: string }> {
    const accessToken = this.generateToken(user.id, user.email, user.role, 'frontend');

    const refreshToken = this.jwtService.sign(
      {
        sub: user.id,
        type: 'refresh',
      },
      { expiresIn: '30d' },
    );

    return { accessToken, refreshToken };
  }
}
