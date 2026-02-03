import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { EventEmitterModule } from '@nestjs/event-emitter';
import { ScheduleModule } from '@nestjs/schedule';
import { AppController } from './app.controller';
import { AppService } from './app.service';

// Core modules (Global)
import { PrismaModule } from './core/prisma/prisma.module';
import { RedisModule } from './core/redis/redis.module';
import { BullMQModule } from './core/bullmq/bullmq.module';

// Feature modules
import { CommonModule } from './common/common.module';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './user/user.module';
import { AIProxyModule } from './ai-proxy/ai-proxy.module';
import { WebSocketModule } from './websocket/websocket.module';
import { TradingModule } from './trading/trading.module';
import { PortfolioModule } from './portfolio/portfolio.module';
import { ExchangeModule } from './exchange/exchange.module';
import { MockExchangeModule } from './mock-exchange/mock-exchange.module';
import { MarketDataModule } from './market-data/market-data.module';
import { RiskModule } from './risk/risk.module';
import { AuditModule } from './audit/audit.module';
import { MetricsModule } from './metrics/metrics.module';
import { HealthModule } from './health/health.module';

@Module({
    imports: [
        // Configuration
        ConfigModule.forRoot({
            isGlobal: true,
            envFilePath: ['.env.local', '.env'],
        }),

        // Event Emitter (for order events, price updates)
        EventEmitterModule.forRoot(),

        // Schedule Module (cron jobs) - register once here
        ScheduleModule.forRoot(),

        // Core Infrastructure (Global modules - order matters)
        PrismaModule,   // Database ORM
        RedisModule,    // Caching & Pub/Sub
        BullMQModule,   // Background Jobs

        // Common utilities
        CommonModule,

        // Feature modules
        AuthModule,
        UserModule,
        AIProxyModule,     // AI Bot proxy service
        ExchangeModule,    // CCXT multi-exchange connectivity
        MockExchangeModule, // Paper trading simulation
        MarketDataModule,   // Phase 2: TimescaleDB data pipeline
        RiskModule,         // Phase 3: Risk management & circuit breakers
        AuditModule,        // Phase 3: Audit logging & compliance
        MetricsModule,      // Phase 4: Prometheus metrics
        HealthModule,       // Phase 4: Health check endpoints
        WebSocketModule,   // Real-time communication
        TradingModule,     // Order execution & positions
        PortfolioModule,   // Portfolio management
    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule { }

