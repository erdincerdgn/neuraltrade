import { Module } from '@nestjs/common';
import { EventsGateway } from './events.gateway';
import { AuthModule } from '../auth/auth.module';

/**
 * WebSocket Module
 * 
 * Real-time communication via Socket.IO with Redis adapter.
 * 
 * Dependencies:
 * - AuthModule: JWT validation for WebSocket connections
 * 
 * Note: Use forwardRef when importing this module in AIProxyModule
 * to avoid circular dependency issues.
 * 
 * Global modules (PrismaModule, RedisModule) are available automatically.
 */
@Module({
    imports: [AuthModule],
    providers: [EventsGateway],
    exports: [EventsGateway],
})
export class WebSocketModule { }
