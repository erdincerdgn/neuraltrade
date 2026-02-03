import { Module } from '@nestjs/common';
import { TerminusModule } from '@nestjs/terminus';
import { HealthController } from './health.controller';

/**
 * Health Module
 * 
 * Phase 4: Kubernetes health check endpoints
 */
@Module({
    imports: [TerminusModule],
    controllers: [HealthController],
})
export class HealthModule { }
