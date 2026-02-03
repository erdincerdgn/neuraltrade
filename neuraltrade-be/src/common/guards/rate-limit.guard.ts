import { Injectable, CanActivate, ExecutionContext, HttpException, HttpStatus } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { RedisService } from '../../core/redis/redis.service';

/**
 * Rate Limit Decorator metadata key
 */
export const RATE_LIMIT_KEY = 'rateLimit';

/**
 * Rate Limit Configuration
 */
export interface RateLimitConfig {
    ttl: number;      // Time window in seconds
    limit: number;    // Max requests in window
}

/**
 * Rate Limit Decorator
 */
export const RateLimit = (config: RateLimitConfig) => {
    return (_target: any, _key?: string, descriptor?: PropertyDescriptor) => {
        Reflect.defineMetadata(RATE_LIMIT_KEY, config, descriptor?.value || _target);
        return descriptor || _target;
    };
};

/**
 * Rate Limit Guard
 * 
 * Redis-backed sliding window rate limiting.
 */
@Injectable()
export class RateLimitGuard implements CanActivate {
    private readonly DEFAULT_TTL = 60;
    private readonly DEFAULT_LIMIT = 100;

    constructor(
        private readonly redis: RedisService,
        private readonly reflector: Reflector,
    ) { }

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const request = context.switchToHttp().getRequest();

        const config = this.reflector.get<RateLimitConfig>(
            RATE_LIMIT_KEY,
            context.getHandler(),
        ) || { ttl: this.DEFAULT_TTL, limit: this.DEFAULT_LIMIT };

        const key = this.buildKey(request);
        const result = await this.redis.checkRateLimit(key, config.limit, config.ttl);

        if (!result.allowed) {
            throw new HttpException(
                {
                    statusCode: HttpStatus.TOO_MANY_REQUESTS,
                    message: 'Rate limit exceeded',
                    retryAfter: result.resetIn,
                },
                HttpStatus.TOO_MANY_REQUESTS,
            );
        }

        const response = context.switchToHttp().getResponse();
        response.header('X-RateLimit-Limit', config.limit.toString());
        response.header('X-RateLimit-Remaining', result.remaining.toString());

        return true;
    }

    private buildKey(request: any): string {
        const ip = request.ip || request.connection?.remoteAddress || 'unknown';
        const userId = request.user?.id || 'anonymous';
        const path = request.route?.path || request.path;
        const method = request.method;

        return `rate_limit:${userId}:${ip}:${method}:${path}`;
    }
}

/**
 * Predefined rate limit configurations
 */
export const RateLimitPresets = {
    STRICT: { ttl: 60, limit: 10 },
    NORMAL: { ttl: 60, limit: 100 },
    RELAXED: { ttl: 60, limit: 500 },
    TRADING: { ttl: 1, limit: 5 },
    AUTH: { ttl: 300, limit: 5 },
};
