import { Injectable, Logger, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import { ExchangeFactory } from '../exchange/exchange.factory';
import { CreatePortfolioDto, UpdatePortfolioDto } from './dto';
import Decimal from 'decimal.js';

/**
 * Enhanced Portfolio Service
 * 
 * Phase 1 improvements:
 * - Locked balance management for open orders
 * - Real-time PnL calculation with live prices
 * - Position tracking with live updates
 * - Daily loss tracking for circuit breaker
 * - Decimal.js for financial precision
 */
@Injectable()
export class PortfolioService {
    private readonly logger = new Logger(PortfolioService.name);

    constructor(
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
        private readonly exchangeFactory: ExchangeFactory,
    ) { }

    // ==========================================
    // PORTFOLIO CRUD
    // ==========================================

    async createPortfolio(userId: number, dto: CreatePortfolioDto) {
        const portfolio = await this.prisma.portfolio.create({
            data: {
                userId,
                name: dto.name,
                description: dto.description,
                currency: dto.currency || 'USD',
                isDefault: dto.isDefault || false,
            },
        });

        this.logger.log(`Portfolio created: ${portfolio.id} for user ${userId}`);
        return portfolio;
    }

    async getPortfolios(userId: number) {
        return this.prisma.portfolio.findMany({
            where: { userId },
            include: {
                _count: {
                    select: { positions: true, orders: true },
                },
            },
            orderBy: { createdAt: 'desc' },
        });
    }

    async getPortfolioById(userId: number, portfolioId: number) {
        const portfolio = await this.prisma.portfolio.findFirst({
            where: { id: portfolioId, userId },
            include: {
                positions: true,
                _count: { select: { trades: true, orders: true } },
            },
        });

        if (!portfolio) {
            throw new NotFoundException('Portfolio not found');
        }

        return portfolio;
    }

    async updatePortfolio(userId: number, portfolioId: number, dto: UpdatePortfolioDto) {
        const portfolio = await this.prisma.portfolio.findFirst({
            where: { id: portfolioId, userId },
        });

        if (!portfolio) {
            throw new NotFoundException('Portfolio not found');
        }

        return this.prisma.portfolio.update({
            where: { id: portfolioId },
            data: dto,
        });
    }

    async deletePortfolio(userId: number, portfolioId: number) {
        const portfolio = await this.prisma.portfolio.findFirst({
            where: { id: portfolioId, userId },
            include: { positions: true },
        });

        if (!portfolio) {
            throw new NotFoundException('Portfolio not found');
        }

        if (portfolio.positions.length > 0) {
            throw new BadRequestException('Cannot delete portfolio with open positions');
        }

        await this.prisma.portfolio.delete({ where: { id: portfolioId } });
        return { message: 'Portfolio deleted' };
    }

    // ==========================================
    // BALANCE MANAGEMENT (NEW - Phase 1)
    // ==========================================

    /**
     * Get user's ledger balance for a currency
     */
    async getBalance(userId: number, currency: string = 'USDT'): Promise<{
        free: string;
        locked: string;
        total: string;
    }> {
        const account = await this.prisma.ledgerAccount.findFirst({
            where: {
                userId,
                accountType: 'TRADING',
                currency,
                isActive: true,
            },
        });

        if (!account) {
            return { free: '0', locked: '0', total: '0' };
        }

        const free = new Decimal(account.balance.toString());
        const locked = new Decimal(account.lockedBalance.toString());

        return {
            free: free.minus(locked).toString(),
            locked: locked.toString(),
            total: free.toString(),
        };
    }

    /**
     * Lock balance for an order (atomic via Prisma transaction)
     */
    async lockBalance(userId: number, currency: string, amount: string): Promise<boolean> {
        const lockAmount = new Decimal(amount);

        try {
            await this.prisma.$transaction(async (tx) => {
                const account = await tx.ledgerAccount.findFirst({
                    where: {
                        userId,
                        accountType: 'TRADING',
                        currency,
                        isActive: true,
                    },
                });

                if (!account) {
                    throw new BadRequestException(`No ${currency} account found`);
                }

                const available = new Decimal(account.balance.toString())
                    .minus(new Decimal(account.lockedBalance.toString()));

                if (available.lessThan(lockAmount)) {
                    throw new BadRequestException(
                        `Insufficient ${currency} balance. Available: ${available}, Required: ${lockAmount}`
                    );
                }

                await tx.ledgerAccount.update({
                    where: { id: account.id },
                    data: {
                        lockedBalance: {
                            increment: lockAmount.toNumber(),
                        },
                    },
                });
            });

            this.logger.debug(`🔒 Locked ${amount} ${currency} for user ${userId}`);
            return true;
        } catch (error) {
            this.logger.error(`Failed to lock balance: ${error.message}`);
            return false;
        }
    }

    /**
     * Unlock balance (on order cancel/reject)
     */
    async unlockBalance(userId: number, currency: string, amount: string): Promise<void> {
        const unlockAmount = new Decimal(amount);

        await this.prisma.ledgerAccount.updateMany({
            where: {
                userId,
                accountType: 'TRADING',
                currency,
            },
            data: {
                lockedBalance: {
                    decrement: unlockAmount.toNumber(),
                },
            },
        });

        this.logger.debug(`🔓 Unlocked ${amount} ${currency} for user ${userId}`);
    }

    /**
     * Consume locked balance and add received asset (on order fill)
     */
    async processOrderFill(
        userId: number,
        consumeCurrency: string,
        consumeAmount: string,
        receiveCurrency: string,
        receiveAmount: string,
    ): Promise<void> {
        await this.prisma.$transaction(async (tx) => {
            // Consume locked balance
            await tx.ledgerAccount.updateMany({
                where: { userId, accountType: 'TRADING', currency: consumeCurrency },
                data: {
                    balance: { decrement: new Decimal(consumeAmount).toNumber() },
                    lockedBalance: { decrement: new Decimal(consumeAmount).toNumber() },
                },
            });

            // Add received asset (create if doesn't exist)
            const receiveAccount = await tx.ledgerAccount.findFirst({
                where: { userId, accountType: 'TRADING', currency: receiveCurrency },
            });

            if (receiveAccount) {
                await tx.ledgerAccount.update({
                    where: { id: receiveAccount.id },
                    data: {
                        balance: { increment: new Decimal(receiveAmount).toNumber() },
                    },
                });
            } else {
                await tx.ledgerAccount.create({
                    data: {
                        userId,
                        accountType: 'TRADING',
                        currency: receiveCurrency,
                        balance: new Decimal(receiveAmount).toNumber(),
                        lockedBalance: 0,
                    },
                });
            }
        });

        this.logger.log(`💱 Processed fill: ${consumeAmount} ${consumeCurrency} → ${receiveAmount} ${receiveCurrency}`);
    }

    // ==========================================
    // REAL-TIME PnL CALCULATION (NEW - Phase 1)
    // ==========================================

    /**
     * Calculate real-time PnL with live prices from exchange
     */
    async getRealTimePnL(userId: number, portfolioId: number): Promise<{
        positions: PositionPnL[];
        totalUnrealizedPnL: string;
        totalUnrealizedPct: string;
        totalValue: string;
    }> {
        const portfolio = await this.getPortfolioById(userId, portfolioId);
        const positions = portfolio.positions;

        if (positions.length === 0) {
            return {
                positions: [],
                totalUnrealizedPnL: '0',
                totalUnrealizedPct: '0',
                totalValue: '0',
            };
        }

        // Get live prices from exchange
        const adapter = await this.exchangeFactory.getDefaultAdapter();
        const positionsWithPnL: PositionPnL[] = [];
        let totalUnrealizedPnL = new Decimal(0);
        let totalCost = new Decimal(0);
        let totalValue = new Decimal(0);

        for (const position of positions) {
            const ticker = await adapter.fetchTicker(position.symbol);
            const currentPrice = ticker ? new Decimal(ticker.last) : new Decimal(position.currentPrice.toString());

            const quantity = new Decimal(position.quantity.toString());
            const avgCost = new Decimal(position.avgCost.toString());
            const costBasis = quantity.mul(avgCost);
            const marketValue = quantity.mul(currentPrice);
            const unrealizedPnL = marketValue.minus(costBasis);
            const unrealizedPct = costBasis.greaterThan(0)
                ? unrealizedPnL.div(costBasis).mul(100)
                : new Decimal(0);

            positionsWithPnL.push({
                symbol: position.symbol,
                side: position.side,
                quantity: quantity.toString(),
                avgCost: avgCost.toString(),
                currentPrice: currentPrice.toString(),
                costBasis: costBasis.toString(),
                marketValue: marketValue.toString(),
                unrealizedPnL: unrealizedPnL.toString(),
                unrealizedPct: unrealizedPct.toFixed(2),
            });

            totalUnrealizedPnL = totalUnrealizedPnL.plus(unrealizedPnL);
            totalCost = totalCost.plus(costBasis);
            totalValue = totalValue.plus(marketValue);
        }

        const totalPct = totalCost.greaterThan(0)
            ? totalUnrealizedPnL.div(totalCost).mul(100)
            : new Decimal(0);

        return {
            positions: positionsWithPnL,
            totalUnrealizedPnL: totalUnrealizedPnL.toString(),
            totalUnrealizedPct: totalPct.toFixed(2),
            totalValue: totalValue.toString(),
        };
    }

    // ==========================================
    // DAILY LOSS TRACKING (Circuit Breaker)
    // ==========================================

    /**
     * Track daily loss for circuit breaker
     */
    async trackDailyLoss(portfolioId: number, lossAmount: string): Promise<{
        dailyLoss: string;
        maxDailyLoss: string | null;
        circuitBreakerTriggered: boolean;
    }> {
        const portfolio = await this.prisma.portfolio.findUnique({
            where: { id: portfolioId },
            include: { user: true },
        });

        if (!portfolio) {
            throw new NotFoundException('Portfolio not found');
        }

        // Reset daily loss if it's a new day
        const now = new Date();
        const lastReset = portfolio.dailyLossReset;
        const isNewDay = !lastReset || lastReset.toDateString() !== now.toDateString();

        const currentDailyLoss = isNewDay
            ? new Decimal(lossAmount)
            : new Decimal(portfolio.dailyLoss.toString()).plus(new Decimal(lossAmount));

        await this.prisma.portfolio.update({
            where: { id: portfolioId },
            data: {
                dailyLoss: currentDailyLoss.toNumber(),
                dailyLossReset: isNewDay ? now : undefined,
            },
        });

        // Check circuit breaker
        const maxDailyLoss = portfolio.user.maxDailyLoss;
        let circuitBreakerTriggered = false;

        if (maxDailyLoss && currentDailyLoss.greaterThan(new Decimal(maxDailyLoss.toString()))) {
            // Trigger circuit breaker
            await this.prisma.user.update({
                where: { id: portfolio.userId },
                data: {
                    tradingEnabled: false,
                    circuitBreakerUntil: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
                },
            });
            circuitBreakerTriggered = true;
            this.logger.warn(`⚠️ Circuit breaker triggered for user ${portfolio.userId}. Daily loss: ${currentDailyLoss}`);
        }

        return {
            dailyLoss: currentDailyLoss.toString(),
            maxDailyLoss: maxDailyLoss?.toString() || null,
            circuitBreakerTriggered,
        };
    }

    // ==========================================
    // PORTFOLIO OVERVIEW
    // ==========================================

    async getPortfolioOverview(userId: number, portfolioId: number) {
        const cacheKey = `portfolio:overview:${portfolioId}`;
        const cached = await this.redis.get(cacheKey);
        if (cached) return cached;

        const portfolio = await this.getPortfolioById(userId, portfolioId);
        const holdings = await this.getHoldings(portfolioId);
        const pnlSummary = await this.getPnLSummary(portfolioId);

        const overview = {
            portfolio: {
                id: portfolio.id,
                name: portfolio.name,
                currency: portfolio.currency,
                totalValue: portfolio.totalValue?.toString(),
                totalPnL: portfolio.totalPnL?.toString(),
            },
            holdings: {
                count: holdings.length,
                items: holdings,
            },
            performance: pnlSummary,
            stats: {
                openPositions: portfolio.positions.length,
                totalTrades: portfolio._count.trades,
                totalOrders: portfolio._count.orders,
            },
        };

        await this.redis.set(cacheKey, overview, 60);
        return overview;
    }

    // ==========================================
    // HOLDINGS
    // ==========================================

    async getHoldings(portfolioId: number) {
        const positions = await this.prisma.position.findMany({
            where: { portfolioId },
        });

        return positions.map(pos => ({
            symbol: pos.symbol,
            quantity: pos.quantity.toString(),
            avgCost: pos.avgCost.toString(),
            currentPrice: pos.currentPrice.toString(),
            marketValue: pos.marketValue.toString(),
            unrealizedPnL: pos.unrealizedPnL.toString(),
            unrealizedPct: pos.unrealizedPct.toString(),
            allocation: pos.allocation.toString(),
            side: pos.side,
        }));
    }

    async getAllocation(portfolioId: number) {
        const holdings = await this.getHoldings(portfolioId);
        const totalValue = holdings.reduce((sum, h) =>
            sum.plus(new Decimal(h.marketValue)), new Decimal(0));

        return holdings.map(h => ({
            symbol: h.symbol,
            value: h.marketValue,
            percentage: totalValue.greaterThan(0)
                ? new Decimal(h.marketValue).div(totalValue).mul(100).toFixed(2)
                : '0',
        }));
    }

    // ==========================================
    // P&L SUMMARY
    // ==========================================

    async getPnLSummary(portfolioId: number) {
        const realizedResult = await this.prisma.trade.aggregate({
            where: { portfolioId },
            _sum: { realizedPnL: true },
        });

        const unrealizedResult = await this.prisma.position.aggregate({
            where: { portfolioId },
            _sum: { unrealizedPnL: true },
        });

        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const dailyResult = await this.prisma.trade.aggregate({
            where: { portfolioId, executedAt: { gte: today } },
            _sum: { realizedPnL: true },
        });

        const realized = new Decimal(realizedResult._sum.realizedPnL?.toString() || '0');
        const unrealized = new Decimal(unrealizedResult._sum.unrealizedPnL?.toString() || '0');

        return {
            realizedPnL: realized.toString(),
            unrealizedPnL: unrealized.toString(),
            totalPnL: realized.plus(unrealized).toString(),
            dailyPnL: dailyResult._sum.realizedPnL?.toString() || '0',
        };
    }

    // ==========================================
    // RISK METRICS
    // ==========================================

    async getRiskMetrics(portfolioId: number) {
        const trades = await this.prisma.trade.findMany({
            where: { portfolioId },
            orderBy: { executedAt: 'asc' },
        });

        if (trades.length === 0) {
            return {
                maxDrawdown: '0',
                sharpeRatio: '0',
                winRate: '0',
                profitFactor: '0',
                totalTrades: 0,
            };
        }

        const pnls = trades.map(t => new Decimal(t.realizedPnL?.toString() || '0'));
        const winningTrades = pnls.filter(p => p.greaterThan(0));
        const losingTrades = pnls.filter(p => p.lessThan(0));

        const totalWins = winningTrades.reduce((a, b) => a.plus(b), new Decimal(0));
        const totalLosses = losingTrades.reduce((a, b) => a.plus(b), new Decimal(0)).abs();

        return {
            winRate: new Decimal(winningTrades.length).div(trades.length).mul(100).toFixed(2),
            profitFactor: totalLosses.greaterThan(0)
                ? totalWins.div(totalLosses).toFixed(2)
                : 'Infinity',
            totalTrades: trades.length,
        };
    }

    // ==========================================
    // HELPERS
    // ==========================================

    async updateTotalValue(portfolioId: number) {
        const holdings = await this.getHoldings(portfolioId);
        const totalValue = holdings.reduce((sum, h) =>
            sum.plus(new Decimal(h.marketValue)), new Decimal(0));

        await this.prisma.portfolio.update({
            where: { id: portfolioId },
            data: { totalValue: totalValue.toNumber() },
        });

        await this.redis.delete(`portfolio:overview:${portfolioId}`);
        return totalValue.toString();
    }
}

// ==========================================
// Types
// ==========================================

interface PositionPnL {
    symbol: string;
    side: string;
    quantity: string;
    avgCost: string;
    currentPrice: string;
    costBasis: string;
    marketValue: string;
    unrealizedPnL: string;
    unrealizedPct: string;
}
