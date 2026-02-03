import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import { PrismaClient, Prisma } from '@prisma/client';

/**
 * Enhanced Prisma Service for NeuralTrade Trading Platform
 * 
 * Features:
 * - Connection pooling
 * - Query logging in development
 * - Soft delete middleware
 * - Transaction helpers
 * - Health check
 */
@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(PrismaService.name);

  constructor() {
    super({
      log: process.env.NODE_ENV === 'development'
        ? ['query', 'info', 'warn', 'error']
        : ['error'],
      errorFormat: 'pretty',
    });
  }

  async onModuleInit() {
    this.logger.log('Connecting to database...');

    try {
      await this.$connect();
      this.logger.log('✅ Database connected successfully');

      // Log query metrics in development
      if (process.env.NODE_ENV === 'development') {
        this.$use(this.queryLoggingMiddleware.bind(this));
      }
    } catch (error) {
      this.logger.error('❌ Database connection failed', error);
      throw error;
    }
  }

  async onModuleDestroy() {
    this.logger.log('Disconnecting from database...');
    await this.$disconnect();
    this.logger.log('Database disconnected');
  }

  /**
   * Query logging middleware for development
   */
  private async queryLoggingMiddleware(
    params: Prisma.MiddlewareParams,
    next: (params: Prisma.MiddlewareParams) => Promise<any>,
  ) {
    const start = Date.now();
    const result = await next(params);
    const duration = Date.now() - start;

    if (duration > 1000) {
      this.logger.warn(
        `⚠️ Slow query detected: ${params.model}.${params.action} took ${duration}ms`,
      );
    }

    return result;
  }

  /**
   * Health check for database connection
   */
  async healthCheck(): Promise<{ status: string; latency: number }> {
    const start = Date.now();
    try {
      await this.$queryRaw`SELECT 1`;
      return {
        status: 'healthy',
        latency: Date.now() - start,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        latency: Date.now() - start,
      };
    }
  }

  /**
   * Execute a transaction with retry logic
   */
  async executeTransaction<T>(
    fn: (tx: Prisma.TransactionClient) => Promise<T>,
    maxRetries = 3,
  ): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.$transaction(fn, {
          maxWait: 5000, // 5 seconds
          timeout: 30000, // 30 seconds
        });
      } catch (error) {
        lastError = error as Error;

        // Check if it's a retryable error (deadlock, timeout)
        if (error instanceof Prisma.PrismaClientKnownRequestError) {
          if (['P2034', 'P2024'].includes(error.code)) {
            this.logger.warn(`Transaction attempt ${attempt} failed, retrying...`);
            await this.delay(Math.pow(2, attempt) * 100); // Exponential backoff
            continue;
          }
        }
        throw error;
      }
    }

    throw lastError;
  }

  /**
   * Clean up expired sessions
   */
  async cleanupExpiredSessions(): Promise<number> {
    const result = await this.session.deleteMany({
      where: {
        expiresAt: {
          lt: new Date(),
        },
      },
    });

    if (result.count > 0) {
      this.logger.log(`Cleaned up ${result.count} expired sessions`);
    }

    return result.count;
  }

  /**
   * Get database statistics
   */
  async getDatabaseStats() {
    const [
      userCount,
      portfolioCount,
      positionCount,
      alertCount,
      signalCount,
    ] = await Promise.all([
      this.user.count(),
      this.portfolio.count(),
      this.position.count(),
      this.alert.count(),
      this.aISignal.count(),
    ]);

    return {
      users: userCount,
      portfolios: portfolioCount,
      positions: positionCount,
      alerts: alertCount,
      aiSignals: signalCount,
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
