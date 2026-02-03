import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, Profile } from 'passport-github2';

/**
 * GitHub OAuth2 Strategy
 * 
 * Handles GitHub authentication for NeuralTrade platform
 */
@Injectable()
export class GithubStrategy extends PassportStrategy(Strategy, 'github') {
    constructor() {
        super({
            clientID: process.env.GITHUB_CLIENT_ID || 'your-client-id',
            clientSecret: process.env.GITHUB_CLIENT_SECRET || 'your-client-secret',
            callbackURL: process.env.GITHUB_CALLBACK_URL || 'http://localhost:4000/api/v1/auth/github/callback',
            scope: ['user:email'],
        });
    }

    async validate(
        accessToken: string,
        _refreshToken: string,
        profile: Profile,
        done: (error: any, user?: any) => void,
    ): Promise<void> {
        const { id, username, displayName, emails, photos } = profile;

        const user = {
            provider: 'github',
            providerId: id,
            username,
            email: emails?.[0]?.value,
            displayName,
            picture: photos?.[0]?.value,
            accessToken,
        };

        done(null, user);
    }
}
