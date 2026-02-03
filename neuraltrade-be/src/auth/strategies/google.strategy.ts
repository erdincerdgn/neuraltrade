import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, VerifyCallback, Profile } from 'passport-google-oauth20';

/**
 * Google OAuth2 Strategy
 * 
 * Handles Google authentication for NeuralTrade platform
 */
@Injectable()
export class GoogleStrategy extends PassportStrategy(Strategy, 'google') {
    constructor() {
        super({
            clientID: process.env.GOOGLE_CLIENT_ID || 'your-client-id',
            clientSecret: process.env.GOOGLE_CLIENT_SECRET || 'your-client-secret',
            callbackURL: process.env.GOOGLE_CALLBACK_URL || 'http://localhost:4000/api/v1/auth/google/callback',
            scope: ['email', 'profile'],
        });
    }

    async validate(
        accessToken: string,
        _refreshToken: string,
        profile: Profile,
        done: VerifyCallback,
    ): Promise<void> {
        const { id, name, emails, photos } = profile;

        const user = {
            provider: 'google',
            providerId: id,
            email: emails?.[0]?.value,
            firstName: name?.givenName,
            lastName: name?.familyName,
            picture: photos?.[0]?.value,
            accessToken,
        };

        done(null, user);
    }
}
