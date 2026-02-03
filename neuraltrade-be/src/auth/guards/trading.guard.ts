import {
    Injectable,
    CanActivate,
    ExecutionContext,
    ForbiddenException,
    Logger,
} from '@nestjs/common';
import { UserStatus } from '@prisma/client';

/**
 * Trading Guard - Circuit Breaker & Risk Management
 * 
 * Security checks:
 * 1. User is authenticated (use after JwtAuthGuard)
 * 2. Trading is enabled for user
 * 3. Circuit breaker is not triggered
 * 4. User status is ACTIVE
 * 
 * Use: @UseGuards(JwtAuthGuard, TradingGuard)
 */
@Injectable()
export class TradingGuard implements CanActivate {
    private readonly logger = new Logger(TradingGuard.name);

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const request = context.switchToHttp().getRequest();
        const user = request.user;

        // 1. Check if user is attached (should be done by JwtAuthGuard first)
        if (!user) {
            throw new ForbiddenException('Authentication required for trading');
        }

        // 2. Check user status
        if (user.status !== UserStatus.ACTIVE) {
            this.logger.warn(`Trading denied for user ${user.id} - Status: ${user.status}`);
            throw new ForbiddenException('Your account must be active to trade');
        }

        // 3. Check if trading is enabled (Circuit Breaker)
        if (user.tradingEnabled === false) {
            this.logger.warn(`Trading disabled for user ${user.id} - Circuit breaker triggered`);

            // Check if there's a circuit breaker timeout
            if (user.circuitBreakerUntil) {
                const now = new Date();
                const unlockTime = new Date(user.circuitBreakerUntil);

                if (now < unlockTime) {
                    const minutesRemaining = Math.ceil((unlockTime.getTime() - now.getTime()) / 60000);
                    throw new ForbiddenException(
                        `Trading is temporarily disabled due to risk limits. ` +
                        `Trading will be re-enabled in ${minutesRemaining} minutes.`
                    );
                }
                // If past the unlock time, trading should be re-enabled by a background job
            }

            throw new ForbiddenException(
                'Trading is disabled for your account. This may be due to reaching your daily loss limit. ' +
                'Please contact support or wait for the next trading day.'
            );
        }

        // 4. Check email verification for trading (optional - uncomment if needed)
        // if (!user.emailVerified) {
        //   throw new ForbiddenException('Please verify your email before trading');
        // }

        this.logger.debug(`Trading access granted for user ${user.id}`);
        return true;
    }
}
