import { Injectable, Logger } from '@nestjs/common';
import Decimal from 'decimal.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Mock Exchange Service
 * 
 * Paper trading simulation engine for development and testing.
 * Phase 1: Simple implementation (500ms → FILLED)
 * 
 * Features:
 * - Simulates order execution with configurable delay
 * - Virtual balance management per user
 * - Thread-safe order tracking
 */
@Injectable()
export class MockExchangeService {
    private readonly logger = new Logger(MockExchangeService.name);

    // Virtual balances per user
    private userBalances: Map<number, Map<string, { free: Decimal; locked: Decimal }>> = new Map();

    // Orders storage (orderId → order)
    private orders: Map<string, MockOrder> = new Map();

    // Configuration
    private readonly FILL_DELAY_MS = 500;

    constructor() {
        this.logger.log('🎮 Mock Exchange Service initialized');
    }

    // ==========================================
    // Balance Management
    // ==========================================

    /**
     * Initialize or get user balance
     */
    getBalance(userId: number, currency: string): { free: string; locked: string; total: string } {
        const userBalance = this.getOrCreateUserBalance(userId);
        const balance = userBalance.get(currency) || { free: new Decimal(0), locked: new Decimal(0) };

        return {
            free: balance.free.toString(),
            locked: balance.locked.toString(),
            total: balance.free.plus(balance.locked).toString(),
        };
    }

    /**
     * Get all balances for a user
     */
    getAllBalances(userId: number): Record<string, { free: string; locked: string; total: string }> {
        const userBalance = this.getOrCreateUserBalance(userId);
        const result: Record<string, { free: string; locked: string; total: string }> = {};

        for (const [currency, balance] of userBalance.entries()) {
            result[currency] = {
                free: balance.free.toString(),
                locked: balance.locked.toString(),
                total: balance.free.plus(balance.locked).toString(),
            };
        }

        return result;
    }

    /**
     * Set initial balance for a user (for testing/paper trading setup)
     */
    setBalance(userId: number, currency: string, amount: string): void {
        const userBalance = this.getOrCreateUserBalance(userId);
        userBalance.set(currency, {
            free: new Decimal(amount),
            locked: new Decimal(0)
        });
        this.logger.log(`💰 Set balance for user ${userId}: ${amount} ${currency}`);
    }

    /**
     * Lock balance for an order (atomic operation)
     * Returns true if successful, false if insufficient funds
     */
    lockBalance(userId: number, currency: string, amount: string): boolean {
        const userBalance = this.getOrCreateUserBalance(userId);
        const balance = userBalance.get(currency);

        if (!balance) {
            this.logger.warn(`No ${currency} balance for user ${userId}`);
            return false;
        }

        const lockAmount = new Decimal(amount);
        if (balance.free.lessThan(lockAmount)) {
            this.logger.warn(`Insufficient ${currency} balance for user ${userId}: ${balance.free} < ${lockAmount}`);
            return false;
        }

        balance.free = balance.free.minus(lockAmount);
        balance.locked = balance.locked.plus(lockAmount);
        userBalance.set(currency, balance);

        this.logger.debug(`🔒 Locked ${amount} ${currency} for user ${userId}`);
        return true;
    }

    /**
     * Unlock balance (on order cancel/reject)
     */
    unlockBalance(userId: number, currency: string, amount: string): void {
        const userBalance = this.getOrCreateUserBalance(userId);
        const balance = userBalance.get(currency);

        if (!balance) return;

        const unlockAmount = new Decimal(amount);
        balance.locked = balance.locked.minus(unlockAmount);
        balance.free = balance.free.plus(unlockAmount);
        userBalance.set(currency, balance);

        this.logger.debug(`🔓 Unlocked ${amount} ${currency} for user ${userId}`);
    }

    /**
     * Consume locked balance (on order fill)
     */
    consumeLockedBalance(userId: number, currency: string, amount: string): void {
        const userBalance = this.getOrCreateUserBalance(userId);
        const balance = userBalance.get(currency);

        if (!balance) return;

        const consumeAmount = new Decimal(amount);
        balance.locked = balance.locked.minus(consumeAmount);
        userBalance.set(currency, balance);
    }

    /**
     * Add balance (for receiving assets on fill)
     */
    addBalance(userId: number, currency: string, amount: string): void {
        const userBalance = this.getOrCreateUserBalance(userId);
        let balance = userBalance.get(currency);

        if (!balance) {
            balance = { free: new Decimal(0), locked: new Decimal(0) };
        }

        balance.free = balance.free.plus(new Decimal(amount));
        userBalance.set(currency, balance);
    }

    // ==========================================
    // Order Management
    // ==========================================

    /**
     * Place a mock order
     */
    async placeOrder(params: PlaceOrderParams): Promise<MockOrder> {
        const orderId = uuidv4();
        const now = Date.now();

        const order: MockOrder = {
            id: orderId,
            userId: params.userId,
            symbol: params.symbol,
            side: params.side,
            type: params.type,
            status: 'pending',
            amount: params.amount,
            filled: '0',
            price: params.price || '0',
            avgFillPrice: undefined,
            createdAt: now,
            updatedAt: now,
        };

        this.orders.set(orderId, order);
        this.logger.log(`📝 Mock order placed: ${orderId} ${params.symbol} ${params.side} ${params.amount}`);

        // Simulate async fill
        this.simulateFill(orderId, params.userId, params.symbol, params.side, params.amount, params.price);

        return order;
    }

    /**
     * Cancel a mock order
     */
    cancelOrder(orderId: string): boolean {
        const order = this.orders.get(orderId);
        if (!order) return false;

        if (order.status === 'filled') {
            this.logger.warn(`Cannot cancel filled order: ${orderId}`);
            return false;
        }

        order.status = 'cancelled';
        order.updatedAt = Date.now();
        this.orders.set(orderId, order);

        this.logger.log(`❌ Mock order cancelled: ${orderId}`);
        return true;
    }

    /**
     * Get order by ID
     */
    getOrder(orderId: string): MockOrder | undefined {
        return this.orders.get(orderId);
    }

    /**
     * Get all orders for a user
     */
    getUserOrders(userId: number): MockOrder[] {
        const orders: MockOrder[] = [];
        for (const order of this.orders.values()) {
            if (order.userId === userId) {
                orders.push(order);
            }
        }
        return orders;
    }

    // ==========================================
    // Private Helpers
    // ==========================================

    private getOrCreateUserBalance(userId: number): Map<string, { free: Decimal; locked: Decimal }> {
        let userBalance = this.userBalances.get(userId);
        if (!userBalance) {
            userBalance = new Map();
            // Initialize with default paper trading balance
            userBalance.set('USDT', { free: new Decimal('10000'), locked: new Decimal(0) });
            userBalance.set('BTC', { free: new Decimal('1'), locked: new Decimal(0) });
            userBalance.set('ETH', { free: new Decimal('10'), locked: new Decimal(0) });
            this.userBalances.set(userId, userBalance);
        }
        return userBalance;
    }

    private async simulateFill(
        orderId: string,
        userId: number,
        symbol: string,
        side: 'buy' | 'sell',
        amount: string,
        price?: string,
    ): Promise<void> {
        // Wait for simulated execution time
        await new Promise(resolve => setTimeout(resolve, this.FILL_DELAY_MS));

        const order = this.orders.get(orderId);
        if (!order || order.status === 'cancelled') return;

        // Simulate fill
        const fillPrice = price || this.getSimulatedPrice(symbol);

        order.status = 'filled';
        order.filled = amount;
        order.avgFillPrice = fillPrice;
        order.updatedAt = Date.now();
        this.orders.set(orderId, order);

        // Update balances based on side
        const [base, quote] = symbol.split('/');
        const fillValue = new Decimal(amount).mul(new Decimal(fillPrice));

        if (side === 'buy') {
            // Consume quote currency, receive base currency
            this.consumeLockedBalance(userId, quote, fillValue.toString());
            this.addBalance(userId, base, amount);
        } else {
            // Consume base currency, receive quote currency
            this.consumeLockedBalance(userId, base, amount);
            this.addBalance(userId, quote, fillValue.toString());
        }

        this.logger.log(`✅ Mock order filled: ${orderId} @ ${fillPrice}`);
    }

    private getSimulatedPrice(symbol: string): string {
        // Simple price simulation
        const prices: Record<string, number> = {
            'BTC/USDT': 45000,
            'ETH/USDT': 2500,
            'BNB/USDT': 300,
            'SOL/USDT': 100,
        };
        return (prices[symbol] || 100).toString();
    }
}

// ==========================================
// Types
// ==========================================

export interface PlaceOrderParams {
    userId: number;
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit';
    amount: string;
    price?: string;
}

export interface MockOrder {
    id: string;
    userId: number;
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit';
    status: 'pending' | 'open' | 'filled' | 'cancelled';
    amount: string;
    filled: string;
    price: string;
    avgFillPrice?: string;
    createdAt: number;
    updatedAt: number;
}
