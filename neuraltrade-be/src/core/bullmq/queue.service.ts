import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import { Queue, Worker, Job, QueueEvents, JobsOptions } from 'bullmq';

/**
 * BullMQ Queue Service for NeuralTrade Trading Platform
 * 
 * Queue types:
 * - trading: Order execution, position updates
 * - signals: AI signal processing
 * - notifications: Email, push, SMS notifications
 * - analytics: Portfolio analytics, risk calculations
 * - market-data: Price updates, market sync
 */
@Injectable()
export class QueueService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(QueueService.name);

    private queues: Map<string, Queue> = new Map();
    private workers: Map<string, Worker> = new Map();
    private events: Map<string, QueueEvents> = new Map();

    // Queue names
    public static readonly QUEUES = {
        TRADING: 'trading',
        SIGNALS: 'signals',
        NOTIFICATIONS: 'notifications',
        ANALYTICS: 'analytics',
        MARKET_DATA: 'market-data',
        EMAIL: 'email',
        CLEANUP: 'cleanup',
    };

    // Default job options
    private readonly defaultJobOptions: JobsOptions = {
        attempts: 3,
        backoff: {
            type: 'exponential',
            delay: 1000,
        },
        removeOnComplete: {
            count: 1000,
            age: 24 * 3600, // 24 hours
        },
        removeOnFail: {
            count: 5000,
        },
    };
    // Parse REDIS_URL or use individual host/port
    private getRedisConnection() {
        const redisUrl = process.env.REDIS_URL;
        if (redisUrl) {
            const url = new URL(redisUrl);
            return {
                host: url.hostname,
                port: parseInt(url.port || '6379'),
                password: url.password || undefined,
            };
        }
        return {
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379'),
            password: process.env.REDIS_PASSWORD,
        };
    }

    private connection = this.getRedisConnection();

    async onModuleInit() {
        this.logger.log('Initializing BullMQ queues...');

        // Initialize all queues
        for (const queueName of Object.values(QueueService.QUEUES)) {
            await this.initQueue(queueName);
        }

        this.logger.log('✅ BullMQ queues initialized');
    }

    async onModuleDestroy() {
        this.logger.log('Closing BullMQ connections...');

        // Close all workers
        for (const [name, worker] of this.workers) {
            await worker.close();
            this.logger.debug(`Worker ${name} closed`);
        }

        // Close all queues
        for (const [name, queue] of this.queues) {
            await queue.close();
            this.logger.debug(`Queue ${name} closed`);
        }

        // Close all events
        for (const [_name, event] of this.events) {
            await event.close();
        }

        this.logger.log('BullMQ connections closed');
    }

    private async initQueue(queueName: string) {
        const queue = new Queue(queueName, { connection: this.connection });
        this.queues.set(queueName, queue);

        const events = new QueueEvents(queueName, { connection: this.connection });
        this.events.set(queueName, events);

        // Log job completions and failures
        events.on('completed', ({ jobId }) => {
            this.logger.debug(`Job ${jobId} in ${queueName} completed`);
        });

        events.on('failed', ({ jobId, failedReason }) => {
            this.logger.error(`Job ${jobId} in ${queueName} failed: ${failedReason}`);
        });
    }

    // ==========================================
    // JOB MANAGEMENT
    // ==========================================

    async addJob<T>(
        queueName: string,
        jobName: string,
        data: T,
        options?: JobsOptions,
    ): Promise<Job<T>> {
        const queue = this.queues.get(queueName);
        if (!queue) {
            throw new Error(`Queue ${queueName} not found`);
        }

        const job = await queue.add(jobName, data, {
            ...this.defaultJobOptions,
            ...options,
        });

        this.logger.debug(`Job ${job.id} added to ${queueName}: ${jobName}`);
        return job;
    }

    async addBulk<T>(
        queueName: string,
        jobs: Array<{ name: string; data: T; options?: JobsOptions }>,
    ): Promise<Job<T>[]> {
        const queue = this.queues.get(queueName);
        if (!queue) {
            throw new Error(`Queue ${queueName} not found`);
        }

        return queue.addBulk(
            jobs.map((job) => ({
                name: job.name,
                data: job.data,
                opts: { ...this.defaultJobOptions, ...job.options },
            })),
        );
    }

    async getJob<T>(queueName: string, jobId: string): Promise<Job<T> | null> {
        const queue = this.queues.get(queueName);
        if (!queue) return null;
        return queue.getJob(jobId);
    }

    // ==========================================
    // WORKER REGISTRATION
    // ==========================================

    registerWorker<T>(
        queueName: string,
        processor: (job: Job<T>) => Promise<any>,
        concurrency = 5,
    ): Worker {
        const worker = new Worker(
            queueName,
            async (job: Job<T>) => {
                this.logger.debug(`Processing job ${job.id} in ${queueName}`);
                return processor(job);
            },
            {
                connection: this.connection,
                concurrency,
            },
        );

        worker.on('error', (error) => {
            this.logger.error(`Worker error in ${queueName}: ${error.message}`);
        });

        this.workers.set(queueName, worker);
        this.logger.log(`Worker registered for queue: ${queueName}`);

        return worker;
    }

    // ==========================================
    // TRADING JOBS
    // ==========================================

    async addTradingJob(data: {
        type: 'order' | 'position' | 'portfolio';
        action: string;
        userId: number;
        payload: any;
    }) {
        return this.addJob(QueueService.QUEUES.TRADING, data.type, data, {
            priority: 1, // High priority for trading
        });
    }

    async addSignalJob(data: {
        signalId: number;
        userId: number;
        symbol: string;
        action: 'BUY' | 'SELL' | 'HOLD';
    }) {
        return this.addJob(QueueService.QUEUES.SIGNALS, 'process-signal', data);
    }

    async addNotificationJob(data: {
        userId: number;
        type: 'email' | 'push' | 'sms';
        template: string;
        data: any;
    }) {
        return this.addJob(QueueService.QUEUES.NOTIFICATIONS, data.type, data);
    }

    async addAnalyticsJob(data: {
        type: 'portfolio' | 'risk' | 'performance';
        userId: number;
        portfolioId?: number;
    }) {
        return this.addJob(QueueService.QUEUES.ANALYTICS, data.type, data);
    }

    // ==========================================
    // QUEUE STATS
    // ==========================================

    async getQueueStats(queueName: string) {
        const queue = this.queues.get(queueName);
        if (!queue) return null;

        const [waiting, active, completed, failed, delayed] = await Promise.all([
            queue.getWaitingCount(),
            queue.getActiveCount(),
            queue.getCompletedCount(),
            queue.getFailedCount(),
            queue.getDelayedCount(),
        ]);

        return { waiting, active, completed, failed, delayed };
    }

    async getAllQueueStats() {
        const stats: Record<string, any> = {};

        for (const queueName of Object.values(QueueService.QUEUES)) {
            stats[queueName] = await this.getQueueStats(queueName);
        }

        return stats;
    }

    async healthCheck(): Promise<{ status: string; queues: number }> {
        return {
            status: this.queues.size > 0 ? 'healthy' : 'unhealthy',
            queues: this.queues.size,
        };
    }
}
