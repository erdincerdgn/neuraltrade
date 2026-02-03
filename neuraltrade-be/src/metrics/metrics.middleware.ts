import { Injectable, NestMiddleware } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { MetricsService } from './metrics.service';

/**
 * Metrics Middleware
 * 
 * Tracks HTTP request duration and counts for Prometheus.
 */
@Injectable()
export class MetricsMiddleware implements NestMiddleware {
    constructor(private readonly metricsService: MetricsService) { }

    use(req: Request, res: Response, next: NextFunction): void {
        const start = process.hrtime.bigint();

        res.on('finish', () => {
            const end = process.hrtime.bigint();
            const duration = Number(end - start) / 1e9; // Convert to seconds

            this.metricsService.recordHttpRequest(
                req.method,
                req.route?.path || req.path,
                res.statusCode,
                duration,
            );
        });

        next();
    }
}
