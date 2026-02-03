import { Module, forwardRef } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ConfigModule } from '@nestjs/config';

import { AIProxyController } from './ai-proxy.controller';
import { AIProxyService } from './ai-proxy.service';
import { GrpcClientService } from './grpc-client.service';
import { SignalProcessorService } from './signal-processor.service';
import { SignalBroadcastHandler } from './signal-broadcast.handler';
import { StrategyRouterService } from './strategy-router.service';
import { ModelSelectorService } from './model-selector.service';

import { WebSocketModule } from 'src/websocket';
import { AuthModule } from 'src/auth/auth.module';

/**
 * NeuralTrade AI Proxy Module v2.0
 * 
 * Unified AI Engine communication module.
 * 
 * PROTOCOL PRIORITY:
 * 1. gRPC (PRIMARY) - Port 50051
 * 2. HTTP (FALLBACK) - Port 8000
 * 
 * Features (Proto v2.0):
 * - Signal Prediction & Streaming
 * - Model Management
 * - Strategy Routing & Backtesting
 * - Volatility Surface (SABR, SVI, Dupire)
 * - Options & Greeks Calculation
 * - Risk Management (VaR, CVaR, Stress Testing)
 * - Portfolio Optimization (Black-Litterman, HRP)
 * - Market Analysis & Regime Detection
 * 
 * @version 2.0.0
 * @author Senior Quant Developer
 */
@Module({
    imports: [
        HttpModule.register({
            timeout: 60000,
            maxRedirects: 3,
            headers: { 'Content-Type': 'application/json' },
        }),
        ConfigModule,
        forwardRef(() => WebSocketModule),
        AuthModule,
    ],
    controllers: [AIProxyController],
    providers: [
        // Core Services
        AIProxyService,
        GrpcClientService,
        // Signal Processing
        SignalProcessorService,
        SignalBroadcastHandler,
        // Strategy & Model Selection
        StrategyRouterService,
        ModelSelectorService,],
    exports: [
        AIProxyService,
        GrpcClientService,
        SignalProcessorService,
        StrategyRouterService,
        ModelSelectorService,
    ],
})
export class AIProxyModule {}