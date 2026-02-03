import { Module, Global } from '@nestjs/common';
import { MockExchangeService } from './mock-exchange.service';

/**
 * Mock Exchange Module
 * 
 * Provides simulated exchange functionality for development and testing:
 * - Order execution simulation
 * - Virtual balance management
 */
@Global()
@Module({
    providers: [MockExchangeService],
    exports: [MockExchangeService],
})
export class MockExchangeModule { }
