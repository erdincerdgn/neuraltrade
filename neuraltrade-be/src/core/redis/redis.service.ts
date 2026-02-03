import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import Redis from 'ioredis';

/**
 * Redis Service for NeuralTrade Trading Platform
 * 
 * Use cases:
 * - Session caching
 * - Rate limiting
 * - Real-time price caching
 * - Market data caching
 * - Circuit breaker state
 * - Pub/Sub for real-time events
 */
@Injectable()
export class RedisService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(RedisService.name);
    private client: Redis;
    private subscriber: Redis;
    private publisher: Redis;

    // Cache TTL defaults (in seconds)
    public static readonly TTL = {
        SHORT: 60,           // 1 minute
        MEDIUM: 300,         // 5 minutes
        LONG: 3600,          // 1 hour
        PRICE: 10,           // 10 seconds for price data
        SESSION: 86400,      // 24 hours
        RATE_LIMIT: 60,      // 1 minute for rate limiting
    };

    async onModuleInit() {
        const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

        try {
            this.client = new Redis(redisUrl, {
                retryStrategy: (times) => {
                    if (times > 3) {
                        this.logger.error('Redis connection failed after 3 attempts');
                        return null;
                    }
                    return Math.min(times * 1000, 3000);
                },
                maxRetriesPerRequest: 3,
            });

            this.subscriber = new Redis(redisUrl);
            this.publisher = new Redis(redisUrl);

            this.client.on('connect', () => {
                this.logger.log('✅ Redis connected successfully');
            });

            this.client.on('error', (err) => {
                this.logger.error('Redis connection error:', err.message);
            });

        } catch (error) {
            this.logger.error('Failed to initialize Redis', error);
        }
    }

    async onModuleDestroy() {
        await Promise.all([
            this.client?.quit(),
            this.subscriber?.quit(),
            this.publisher?.quit(),
        ]);
        this.logger.log('Redis connections closed');
    }

    // ==========================================
    // BASIC OPERATIONS
    // ==========================================

    async get<T>(key: string): Promise<T | null> {
        const value = await this.client.get(key);
        if (!value) return null;
        try {
            return JSON.parse(value) as T;
        } catch {
            return value as unknown as T;
        }
    }

    async set(key: string, value: any, ttlSeconds?: number): Promise<void> {
        const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
        if (ttlSeconds) {
            await this.client.setex(key, ttlSeconds, stringValue);
        } else {
            await this.client.set(key, stringValue);
        }
    }

    async delete(key: string): Promise<void> {
        await this.client.del(key);
    }

    async exists(key: string): Promise<boolean> {
        const result = await this.client.exists(key);
        return result === 1;
    }

    async increment(key: string, amount = 1): Promise<number> {
        return this.client.incrby(key, amount);
    }

    async expire(key: string, ttlSeconds: number): Promise<void> {
        await this.client.expire(key, ttlSeconds);
    }

    // ==========================================
    // RATE LIMITING
    // ==========================================

    async checkRateLimit(
        identifier: string,
        limit: number,
        windowSeconds: number,
    ): Promise<{ allowed: boolean; remaining: number; resetIn: number }> {
        const key = `ratelimit:${identifier}`;
        const current = await this.client.incr(key);

        if (current === 1) {
            await this.client.expire(key, windowSeconds);
        }

        const ttl = await this.client.ttl(key);

        return {
            allowed: current <= limit,
            remaining: Math.max(0, limit - current),
            resetIn: ttl,
        };
    }

    // ==========================================
    // CACHING HELPERS
    // ==========================================

    async cachePrice(symbol: string, price: number): Promise<void> {
        await this.set(`price:${symbol}`, price, RedisService.TTL.PRICE);
    }

    async getCachedPrice(symbol: string): Promise<number | null> {
        return this.get<number>(`price:${symbol}`);
    }

    async cacheMarketData(symbol: string, data: any): Promise<void> {
        await this.set(`market:${symbol}`, data, RedisService.TTL.SHORT);
    }

    async getCachedMarketData<T>(symbol: string): Promise<T | null> {
        return this.get<T>(`market:${symbol}`);
    }

    // ==========================================
    // SESSION MANAGEMENT
    // ==========================================

    async setSession(sessionId: string, userId: number, ttl = RedisService.TTL.SESSION): Promise<void> {
        await this.set(`session:${sessionId}`, { userId, createdAt: Date.now() }, ttl);
    }

    async getSession(sessionId: string): Promise<{ userId: number; createdAt: number } | null> {
        return this.get(`session:${sessionId}`);
    }

    async invalidateSession(sessionId: string): Promise<void> {
        await this.delete(`session:${sessionId}`);
    }

    async invalidateUserSessions(userId: number): Promise<void> {
        const pattern = `session:*`;
        const keys = await this.client.keys(pattern);

        for (const key of keys) {
            const session = await this.get<{ userId: number }>(key);
            if (session?.userId === userId) {
                await this.delete(key);
            }
        }
    }

    // ==========================================
    // PUB/SUB
    // ==========================================

    async publish(channel: string, message: any): Promise<void> {
        const stringMessage = typeof message === 'string' ? message : JSON.stringify(message);
        await this.publisher.publish(channel, stringMessage);
    }

    async subscribe(channel: string, callback: (message: any) => void): Promise<void> {
        await this.subscriber.subscribe(channel);
        this.subscriber.on('message', (ch, message) => {
            if (ch === channel) {
                try {
                    callback(JSON.parse(message));
                } catch {
                    callback(message);
                }
            }
        });
    }

    async unsubscribe(channel: string): Promise<void> {
        await this.subscriber.unsubscribe(channel);
    }

    // ==========================================
    // HEALTH CHECK
    // ==========================================

    async healthCheck(): Promise<{ status: string; latency: number }> {
        const start = Date.now();
        try {
            await this.client.ping();
            return {
                status: 'healthy',
                latency: Date.now() - start,
            };
        } catch {
            return {
                status: 'unhealthy',
                latency: Date.now() - start,
            };
        }
    }

    /**
     * Ping Redis - used for health checks
     */
    async ping(): Promise<string> {
        return this.client.ping();
    }
}
