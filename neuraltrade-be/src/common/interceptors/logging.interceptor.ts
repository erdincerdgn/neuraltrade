import { CallHandler, ExecutionContext, Injectable, Logger, NestInterceptor } from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

/**
 * Enhanced Logging Interceptor
 * - Uses NestJS Logger instead of console.log
 * - Logs user info if authenticated
 * - Masks sensitive data (passwords, tokens)
 * - Logs response status and time
 */
@Injectable()
export class LoggingInterceptor implements NestInterceptor {
    private readonly logger = new Logger('HTTP');

    // Fields to mask in logs
    private readonly sensitiveFields = ['password', 'token', 'secret', 'authorization', 'apiKey', 'accessToken', 'refreshToken'];

    intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
        const request = context.switchToHttp().getRequest();
        const { method, url, body = {}, query = {}, user, ip } = request;
        const now = Date.now();
        const requestId = this.generateRequestId();

        // Get user info if authenticated
        const userId = user?.id || 'anonymous';
        const userRole = user?.role || 'guest';

        // Log request
        this.logger.log(
            `[${requestId}] ➡️ ${method} ${url} | User: ${userId} (${userRole}) | IP: ${ip}`,
        );

        // Log query params (non-production)
        if (process.env.NODE_ENV !== 'production' && Object.keys(query).length > 0) {
            this.logger.debug(`[${requestId}] Query: ${JSON.stringify(query)}`);
        }

        // Log body without sensitive data (non-production)
        if (process.env.NODE_ENV !== 'production' && Object.keys(body).length > 0) {
            const maskedBody = this.maskSensitiveData(body);
            this.logger.debug(`[${requestId}] Body: ${JSON.stringify(maskedBody)}`);
        }

        return next.handle().pipe(
            tap((_responseData) => {
                const responseTime = Date.now() - now;
                const statusCode = context.switchToHttp().getResponse().statusCode;

                this.logger.log(
                    `[${requestId}] ⬅️ ${method} ${url} | ${statusCode} | ${responseTime}ms`,
                );

                // Log slow requests
                if (responseTime > 3000) {
                    this.logger.warn(`[${requestId}] ⚠️ Slow request detected: ${responseTime}ms`);
                }
            }),
            catchError((error) => {
                const responseTime = Date.now() - now;
                const statusCode = error.status || 500;

                this.logger.error(
                    `[${requestId}] ❌ ${method} ${url} | ${statusCode} | ${responseTime}ms | ${error.message}`,
                );

                return throwError(() => error);
            }),
        );
    }

    /**
     * Mask sensitive fields in request body
     */
    private maskSensitiveData(data: any): any {
        if (!data || typeof data !== 'object') {
            return data;
        }

        const masked = { ...data };

        for (const key of Object.keys(masked)) {
            if (this.sensitiveFields.some(field => key.toLowerCase().includes(field.toLowerCase()))) {
                masked[key] = '***MASKED***';
            } else if (typeof masked[key] === 'object' && masked[key] !== null) {
                masked[key] = this.maskSensitiveData(masked[key]);
            }
        }

        return masked;
    }

    /**
     * Generate a short request ID for tracing
     */
    private generateRequestId(): string {
        return Math.random().toString(36).substring(2, 10);
    }
}