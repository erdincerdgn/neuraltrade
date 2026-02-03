import { Controller, Get, UseGuards, Req, Res } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Response } from 'express';
import { ConfigService } from '@nestjs/config';
import { AuthService } from '../auth.service';
import { Public } from '../rbac/decorators';

/**
 * OAuth Controller
 * 
 * Handles OAuth2 authentication flows for Google and GitHub
 */
@Controller('auth')
export class OAuthController {
    constructor(
        private readonly authService: AuthService,
        private readonly configService: ConfigService,
    ) { }

    // ==========================================
    // GOOGLE OAuth
    // ==========================================

    @Public()
    @Get('google')
    @UseGuards(AuthGuard('google'))
    googleLogin(): void {
        // Initiates Google OAuth flow
    }

    @Public()
    @Get('google/callback')
    @UseGuards(AuthGuard('google'))
    async googleCallback(@Req() req: any, @Res() res: Response): Promise<void> {
        const user = req.user;

        // Find or create user in database
        const dbUser = await this.authService.findOrCreateOAuthUser({
            provider: user.provider,
            providerId: user.providerId,
            email: user.email,
            firstName: user.firstName,
            lastName: user.lastName,
            picture: user.picture,
        });

        // Generate JWT tokens
        const tokens = await this.authService.generateTokens(dbUser);

        // Redirect to frontend with tokens
        const frontendUrl = this.configService.get('FRONTEND_URL', 'http://localhost:3001');
        res.redirect(
            `${frontendUrl}/auth/callback?access_token=${tokens.accessToken}&refresh_token=${tokens.refreshToken}`,
        );
    }

    // ==========================================
    // GITHUB OAuth
    // ==========================================

    @Public()
    @Get('github')
    @UseGuards(AuthGuard('github'))
    githubLogin(): void {
        // Initiates GitHub OAuth flow
    }

    @Public()
    @Get('github/callback')
    @UseGuards(AuthGuard('github'))
    async githubCallback(@Req() req: any, @Res() res: Response): Promise<void> {
        const user = req.user;

        // Find or create user in database
        const dbUser = await this.authService.findOrCreateOAuthUser({
            provider: user.provider,
            providerId: user.providerId,
            email: user.email,
            firstName: user.displayName?.split(' ')[0],
            lastName: user.displayName?.split(' ').slice(1).join(' '),
            picture: user.picture,
            username: user.username,
        });

        // Generate JWT tokens
        const tokens = await this.authService.generateTokens(dbUser);

        // Redirect to frontend with tokens
        const frontendUrl = this.configService.get('FRONTEND_URL', 'http://localhost:3001');
        res.redirect(
            `${frontendUrl}/auth/callback?access_token=${tokens.accessToken}&refresh_token=${tokens.refreshToken}`,
        );
    }
}
