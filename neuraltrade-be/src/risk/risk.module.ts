import { Module } from '@nestjs/common';
import { RiskEngineService } from './risk-engine.service';
import { CircuitBreakerService } from './circuit-breaker.service';
import { RiskController } from './risk.controller';
import { AuthModule } from 'src/auth/auth.module';

/**
 * Risk Module
 * 
 * Phase 3: Risk management and circuit breakers
 * 
 * Dependencies:
 * - AuthModule: Authentication guards
 * 
 * Note: The following @Global() modules are injected automatically:
 * - PrismaModule: User risk profiles, trade history
 * - RedisModule: Real-time risk calculations, circuit breaker state
 */
@Module({
    imports: [AuthModule],
    controllers: [RiskController],
    providers: [
        RiskEngineService,
        CircuitBreakerService,
    ],
    exports: [
        RiskEngineService,
        CircuitBreakerService,
    ],
})
export class RiskModule { }
