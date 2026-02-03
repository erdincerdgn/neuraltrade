import { Controller, Get } from '@nestjs/common'; // HttpCode, HttpStatus
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { AppService } from './app.service';

@ApiTags('System')
@Controller()
export class AppController {
    constructor(private readonly appService: AppService) { }

    /**
     * Root endpoint - API info
     */
    @Get()
    @ApiOperation({ summary: 'Get API information' })
    @ApiResponse({ status: 200, description: 'API info returned successfully' })
    getAppInfo() {
        return this.appService.getAppInfo();
    }

    /**
     * Health check endpoint - comprehensive system health
     */
    // @Get('health')
    // @ApiOperation({ summary: 'Get system health status' })
    // @ApiResponse({ status: 200, description: 'System is healthy' })
    // @ApiResponse({ status: 503, description: 'System is degraded or unhealthy' })
    // async healthCheck() {
    //     const health = await this.appService.healthCheck();
    //     return health;
    // }

    /**
     * Kubernetes readiness probe
     * Returns 200 if app is ready to serve requests
     */
    // @Get('health/ready')
    // @HttpCode(HttpStatus.OK)
    // @ApiOperation({ summary: 'Kubernetes readiness probe' })
    // @ApiResponse({ status: 200, description: 'App is ready' })
    // @ApiResponse({ status: 503, description: 'App is not ready' })
    // async readinessCheck() {
    //     const result = await this.appService.readinessCheck();
    //     return result;
    // }

    /**
     * Kubernetes liveness probe
     * Returns 200 if app is alive
     */
    // @Get('health/live')
    // @HttpCode(HttpStatus.OK)
    // @ApiOperation({ summary: 'Kubernetes liveness probe' })
    // @ApiResponse({ status: 200, description: 'App is alive' })
    // livenessCheck() {
    //     return this.appService.livenessCheck();
    // }

    /**
     * System statistics (admin only in production)
     */
    @Get('stats')
    @ApiOperation({ summary: 'Get system statistics' })
    @ApiResponse({ status: 200, description: 'System stats returned' })
    async getSystemStats() {
        return this.appService.getSystemStats();
    }
}