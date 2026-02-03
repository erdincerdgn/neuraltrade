import { Injectable, Logger, BadRequestException, ForbiddenException } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import { QueueService } from '../core/bullmq/queue.service';
import { ExchangeFactory } from '../exchange/exchange.factory';
import { OrderStatus } from '@prisma/client';
import { CreateOrderDto } from './dto';
import { createOrderStateMachine } from './order-state-machine';
import Decimal from 'decimal.js';

/**
 * Enhanced Trading Service
 * 
 * Phase 1 improvements:
 * - Exchange Adapter Pattern integration
 * - Prisma $transaction for ACID operations (balance lock + order create)
 * - Order State Machine for lifecycle management
 * - Decimal.js for financial precision
 */
@Injectable()
export class TradingService {
    private readonly logger = new Logger(TradingService.name);

    constructor(
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
        private readonly queue: QueueService,
        private readonly exchangeFactory: ExchangeFactory,
        private readonly eventEmitter: EventEmitter2,
    ) { }

    // ==========================================
    // ORDER EXECUTION (ACID Transaction)
    // ==========================================

    /**
     * Place order with ATOMIC balance lock
     * Uses Prisma $transaction to ensure balance lock + order creation are atomic
     */
    async placeOrder(userId: number, portfolioId: number, dto: CreateOrderDto) {
        // Validate user can trade
        const user = await this.prisma.user.findUnique({ where: { id: userId } });
        if (!user?.tradingEnabled) {
            throw new ForbiddenException('Trading is disabled for your account');
        }

        // Validate portfolio belongs to user
        const portfolio = await this.prisma.portfolio.findFirst({
            where: { id: portfolioId, userId },
        });
        if (!portfolio) {
            throw new BadRequestException('Portfolio not found');
        }

        // Calculate required balance to lock using Decimal.js
        const quantity = new Decimal(dto.quantity);
        const price = dto.price ? new Decimal(dto.price) : await this.getEstimatedPrice(dto.symbol);
        const requiredBalance = quantity.mul(price);

        // Determine quote currency (e.g., USDT for BTC/USDT)
        const quoteCurrency = this.getQuoteCurrency(dto.symbol);

        // ATOMIC TRANSACTION: Lock balance + Create order
        const order = await this.prisma.$transaction(async (tx) => {
            // 1. Get or create ledger account
            let ledgerAccount = await tx.ledgerAccount.findFirst({
                where: {
                    userId,
                    accountType: 'TRADING',
                    currency: quoteCurrency,
                },
            });

            if (!ledgerAccount) {
                throw new BadRequestException(`No ${quoteCurrency} balance available`);
            }

            // 2. Check sufficient free balance (for BUY orders)
            if (dto.side === 'BUY') {
                const freeBalance = new Decimal(ledgerAccount.balance.toString());
                const lockedBalance = new Decimal(ledgerAccount.lockedBalance.toString());
                const availableBalance = freeBalance.minus(lockedBalance);

                if (availableBalance.lessThan(requiredBalance)) {
                    throw new BadRequestException(
                        `Insufficient ${quoteCurrency} balance. Available: ${availableBalance}, Required: ${requiredBalance}`
                    );
                }

                // 3. Lock the balance (UPDATE within transaction)
                await tx.ledgerAccount.update({
                    where: { id: ledgerAccount.id },
                    data: {
                        lockedBalance: {
                            increment: requiredBalance.toNumber(),
                        },
                    },
                });
            }

            // 4. Create order (INSERT within same transaction)
            const newOrder = await tx.order.create({
                data: {
                    userId,
                    portfolioId,
                    symbol: dto.symbol,
                    side: dto.side,
                    type: dto.type,
                    quantity: dto.quantity,
                    price: dto.price,
                    stopPrice: dto.stopPrice,
                    status: OrderStatus.PENDING,
                    assetType: 'CRYPTO',
                },
            });

            return newOrder;
        });

        // Initialize state machine for order
        const stateMachine = createOrderStateMachine(order.id, 'PENDING', this.eventEmitter);

        // Queue for execution
        await this.queue.addTradingJob({
            type: 'order',
            action: 'execute',
            userId,
            payload: {
                orderId: order.id,
                portfolioId,
                lockedAmount: requiredBalance.toString(),
                quoteCurrency,
            },
        });

        // Transition state to SUBMITTED
        stateMachine.transition('SUBMIT');

        this.logger.log(`📝 Order placed: ${order.id} - ${dto.symbol} ${dto.side} ${dto.quantity} (Locked: ${requiredBalance} ${quoteCurrency})`);

        return {
            ...order,
            lockedAmount: requiredBalance.toString(),
        };
    }

    /**
     * Cancel order and unlock balance
     */
    async cancelOrder(userId: number, orderId: number) {
        // ATOMIC: Cancel order + unlock balance
        return await this.prisma.$transaction(async (tx) => {
            const order = await tx.order.findFirst({
                where: { id: orderId, userId },
            });

            if (!order) {
                throw new BadRequestException('Order not found');
            }

            if (order.status !== OrderStatus.PENDING && order.status !== OrderStatus.OPEN) {
                throw new BadRequestException('Order cannot be cancelled');
            }

            // Calculate locked amount
            const quantity = new Decimal(order.quantity.toString());
            const price = order.price ? new Decimal(order.price.toString()) : new Decimal(0);
            const lockedAmount = quantity.mul(price);
            const quoteCurrency = this.getQuoteCurrency(order.symbol);

            // Unlock balance
            if (order.side === 'BUY' && lockedAmount.greaterThan(0)) {
                await tx.ledgerAccount.updateMany({
                    where: {
                        userId,
                        accountType: 'TRADING',
                        currency: quoteCurrency,
                    },
                    data: {
                        lockedBalance: {
                            decrement: lockedAmount.toNumber(),
                        },
                    },
                });
            }

            // Cancel order
            const cancelledOrder = await tx.order.update({
                where: { id: orderId },
                data: {
                    status: OrderStatus.CANCELLED,
                    cancelledAt: new Date(),
                },
            });

            this.logger.log(`❌ Order cancelled: ${orderId} (Unlocked: ${lockedAmount} ${quoteCurrency})`);

            // Emit event
            this.eventEmitter.emit('order.cancelled', { orderId, userId });

            return cancelledOrder;
        });
    }

    /**
     * Execute order via exchange adapter
     */
    async executeOrder(orderId: number, _portfolioId: number) {
        const order = await this.prisma.order.findUnique({
            where: { id: orderId },
        });

        if (!order) {
            throw new BadRequestException('Order not found');
        }

        try {
            // Get appropriate exchange adapter
            const adapter = await this.exchangeFactory.getDefaultAdapter();

            // Create order on exchange
            const exchangeOrder = await adapter.createOrder({
                symbol: order.symbol,
                side: order.side === 'BUY' ? 'buy' : 'sell',
                type: order.type === 'MARKET' ? 'market' : 'limit',
                amount: order.quantity.toString(),
                price: order.price?.toString(),
            });

            // Update order with exchange ID
            await this.prisma.order.update({
                where: { id: orderId },
                data: {
                    exchangeId: exchangeOrder.id,
                    status: OrderStatus.OPEN,
                },
            });

            this.logger.log(`✅ Order submitted to exchange: ${orderId} → ${exchangeOrder.id}`);
            return exchangeOrder;
        } catch (error) {
            // On error, mark order as rejected
            await this.prisma.order.update({
                where: { id: orderId },
                data: {
                    status: OrderStatus.REJECTED,
                    errorMessage: error.message,
                },
            });

            // Unlock balance
            await this.unlockBalanceForOrder(orderId);

            this.logger.error(`Order execution failed: ${orderId} - ${error.message}`);
            throw error;
        }
    }

    // ==========================================
    // POSITION MANAGEMENT
    // ==========================================

    async getPositions(userId: number, portfolioId: number) {
        const portfolio = await this.prisma.portfolio.findFirst({
            where: { id: portfolioId, userId },
        });
        if (!portfolio) {
            throw new BadRequestException('Portfolio not found');
        }

        return this.prisma.position.findMany({
            where: { portfolioId },
            orderBy: { openedAt: 'desc' },
        });
    }

    async getPositionSummary(portfolioId: number) {
        const positions = await this.prisma.position.findMany({
            where: { portfolioId },
        });

        const totalValue = positions.reduce((sum, p) =>
            sum.plus(new Decimal(p.marketValue.toString())), new Decimal(0));
        const totalUnrealizedPnL = positions.reduce((sum, p) =>
            sum.plus(new Decimal(p.unrealizedPnL.toString())), new Decimal(0));

        return {
            totalPositions: positions.length,
            longPositions: positions.filter(p => p.side === 'LONG').length,
            shortPositions: positions.filter(p => p.side === 'SHORT').length,
            totalValue: totalValue.toString(),
            totalUnrealizedPnL: totalUnrealizedPnL.toString(),
        };
    }

    // ==========================================
    // TRADING STATISTICS
    // ==========================================

    async getTradingStats(userId: number, portfolioId?: number) {
        const cacheKey = `trading:stats:${userId}:${portfolioId || 'all'}`;
        const cached = await this.redis.get(cacheKey);
        if (cached) return cached;

        const where = portfolioId
            ? { portfolioId, userId }
            : { userId };

        const [totalTrades, winningTrades, totalPnL] = await Promise.all([
            this.prisma.trade.count({ where }),
            this.prisma.trade.count({
                where: { ...where, realizedPnL: { gt: 0 } },
            }),
            this.prisma.trade.aggregate({
                where,
                _sum: { realizedPnL: true },
            }),
        ]);

        const stats = {
            totalTrades,
            winningTrades,
            losingTrades: totalTrades - winningTrades,
            winRate: totalTrades > 0 ? (winningTrades / totalTrades * 100).toFixed(2) : '0.00',
            totalPnL: totalPnL._sum.realizedPnL?.toString() || '0',
        };

        await this.redis.set(cacheKey, stats, 300);
        return stats;
    }

    // ==========================================
    // ORDER HISTORY
    // ==========================================

    async getOrderHistory(userId: number, portfolioId: number, limit = 50) {
        return this.prisma.order.findMany({
            where: { portfolioId, userId },
            orderBy: { createdAt: 'desc' },
            take: limit,
        });
    }

    async getTradeHistory(userId: number, portfolioId: number, limit = 50) {
        return this.prisma.trade.findMany({
            where: { portfolioId, userId },
            orderBy: { executedAt: 'desc' },
            take: limit,
        });
    }

    // ==========================================
    // RISK METRICS
    // ==========================================

    async getRiskMetrics(portfolioId: number) {
        const positions = await this.prisma.position.findMany({
            where: { portfolioId },
        });

        const totalExposure = positions.reduce((sum, p) =>
            sum.plus(new Decimal(p.quantity.toString()).mul(new Decimal(p.currentPrice.toString()))), new Decimal(0));

        const totalUnrealizedPnL = positions.reduce((sum, p) =>
            sum.plus(new Decimal(p.unrealizedPnL?.toString() || '0')), new Decimal(0));

        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
        });

        const portfolioValue = new Decimal(portfolio?.totalValue?.toString() || '0');

        return {
            totalExposure: totalExposure.toString(),
            totalUnrealizedPnL: totalUnrealizedPnL.toString(),
            exposurePercent: portfolioValue.greaterThan(0)
                ? totalExposure.div(portfolioValue).mul(100).toFixed(2)
                : '0',
            positionCount: positions.length,
        };
    }

    // ==========================================
    // HELPER METHODS
    // ==========================================

    private async getEstimatedPrice(symbol: string): Promise<Decimal> {
        try {
            const adapter = await this.exchangeFactory.getDefaultAdapter();
            const ticker = await adapter.fetchTicker(symbol);
            return ticker ? new Decimal(ticker.last) : new Decimal(0);
        } catch {
            return new Decimal(0);
        }
    }

    private getQuoteCurrency(symbol: string): string {
        // Extract quote currency from symbol (e.g., BTC/USDT → USDT)
        const parts = symbol.split('/');
        return parts.length > 1 ? parts[1] : 'USDT';
    }

    private async unlockBalanceForOrder(orderId: number): Promise<void> {
        const order = await this.prisma.order.findUnique({
            where: { id: orderId },
        });

        if (!order) return;

        const quantity = new Decimal(order.quantity.toString());
        const price = order.price ? new Decimal(order.price.toString()) : new Decimal(0);
        const lockedAmount = quantity.mul(price);
        const quoteCurrency = this.getQuoteCurrency(order.symbol);

        if (order.side === 'BUY' && lockedAmount.greaterThan(0)) {
            await this.prisma.ledgerAccount.updateMany({
                where: {
                    userId: order.userId,
                    accountType: 'TRADING',
                    currency: quoteCurrency,
                },
                data: {
                    lockedBalance: {
                        decrement: lockedAmount.toNumber(),
                    },
                },
            });
        }
    }
}
