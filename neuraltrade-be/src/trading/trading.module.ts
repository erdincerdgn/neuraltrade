import { Module } from '@nestjs/common';
import { TradingService } from './trading.service';
import { TradingController } from './trading.controller';
import { AuthModule } from 'src/auth/auth.module';

/**
 * Trading Module
 * 
 * Order execution and position management.
 * 
 * Dependencies:
 * - AuthModule: Authentication guards
 * 
 * Note: The following @Global() modules are injected automatically:
 * - PrismaModule: Database operations
 * - RedisModule: Order caching and pub/sub
 * - ExchangeModule: Exchange adapters
 * - MockExchangeModule: Paper trading
 */
@Module({
    imports: [AuthModule],
    controllers: [TradingController],
    providers: [TradingService],
    exports: [TradingService],
})
export class TradingModule { }
