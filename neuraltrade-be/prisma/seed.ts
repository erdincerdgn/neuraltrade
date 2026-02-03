/**
 * NeuralTrade Database Seed Script
 * ================================
 * Creates comprehensive fake data for API testing in Postman
 * 
 * Run: npm run seed
 * Or:  npx ts-node prisma/seed.ts
 */

import { PrismaClient, Prisma } from '@prisma/client';
import * as bcrypt from 'bcrypt';

const prisma = new PrismaClient();

// ============================================
// HELPER FUNCTIONS
// ============================================

function randomDecimal(min: number, max: number, decimals: number = 4): Prisma.Decimal {
    const value = Math.random() * (max - min) + min;
    return new Prisma.Decimal(value.toFixed(decimals));
}

function randomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomDate(start: Date, end: Date): Date {
    return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
}

function randomElement<T>(arr: readonly T[]): T {
    return arr[Math.floor(Math.random() * arr.length)];
}

// ============================================
// SEED DATA
// ============================================

async function main() {
    console.log('🌱 Starting NeuralTrade Database Seed...\n');

    // Clean existing data (optional - comment out if you want to keep existing data)
    console.log('🧹 Cleaning existing data...');
    await cleanDatabase();

    // 1. SUBSCRIPTION PLANS
    console.log('📋 Creating Subscription Plans...');
    const plans = await createSubscriptionPlans();

    // 2. USERS
    console.log('👤 Creating Users...');
    const users = await createUsers();

    // 3. SUBSCRIPTIONS
    console.log('💳 Creating Subscriptions...');
    await createSubscriptions(users, plans);

    // 4. ASSETS
    console.log('📈 Creating Assets...');
    const assets = await createAssets();

    // 5. PORTFOLIOS
    console.log('💼 Creating Portfolios...');
    const portfolios = await createPortfolios(users);

    // 6. POSITIONS
    console.log('📊 Creating Positions...');
    await createPositions(portfolios, assets);

    // 7. ORDERS & TRADES
    console.log('📝 Creating Orders & Trades...');
    await createOrdersAndTrades(users, portfolios, assets);

    // 8. AI SIGNALS
    console.log('🤖 Creating AI Signals...');
    await createAISignals(users, assets);

    // 9. STRATEGIES & BACKTESTS
    console.log('📐 Creating Strategies & Backtests...');
    await createStrategiesAndBacktests(users);

    // 10. WATCHLISTS & ALERTS
    console.log('👁️ Creating Watchlists & Alerts...');
    await createWatchlistsAndAlerts(users, assets);

    // 11. LEDGER ACCOUNTS & TRANSACTIONS
    console.log('💰 Creating Ledger Accounts & Transactions...');
    await createLedgerData(users);

    // 12. MARKET DATA (Technical Analysis, Regime Detection)
    console.log('📉 Creating Market Data...');
    await createMarketData(assets);

    // 13. SYSTEM DATA (Jobs, Metrics, Notifications)
    console.log('⚙️ Creating System Data...');
    await createSystemData(users);

    console.log('\n✅ Seed completed successfully!');
    console.log('📊 Database populated with test data for Postman testing.\n');
}

// ============================================
// CLEANUP
// ============================================

async function cleanDatabase() {
    const tables = [
        'ledger_entry', 'ledger_account', 'transaction',
        'trade', 'order', 'position', 'portfolio_snapshot', 'portfolio_optimization', 'portfolio',
        'ai_signal', 'signal_history', 'swarm_consensus',
        'technical_analysis', 'regime_detection', 'risk_metrics',
        'volatility_surface', 'options_greeks', 'stress_test_result',
        'quantum_optimization', 'rag_query',
        'backtest', 'strategy',
        'alert', 'watchlist', 'notification',
        'invoice', 'subscription',
        'price_history', 'fundamental', 'asset',
        'economic_event',
        'job', 'system_metric', 'audit_log',
        'api_key', 'exchange_connection', 'session',
        'user', 'subscription_plan', 'media'
    ];

    for (const table of tables) {
        try {
            await prisma.$executeRawUnsafe(`TRUNCATE TABLE "${table}" CASCADE`);
        } catch (e) {
            // Table might not exist yet
        }
    }
}

// ============================================
// SUBSCRIPTION PLANS
// ============================================

async function createSubscriptionPlans() {
    const plans = [
        {
            name: 'Free',
            slug: 'free',
            description: 'Basic features for getting started',
            priceMonthly: new Prisma.Decimal(0),
            priceYearly: new Prisma.Decimal(0),
            features: { signals: 5, portfolios: 1, backtests: 2 },
            maxPortfolios: 1,
            maxWatchlists: 2,
            maxAlerts: 5,
            maxBacktests: 2,
            aiSignalsEnabled: false,
            sortOrder: 1
        },
        {
            name: 'Pro',
            slug: 'pro',
            description: 'Advanced features for serious traders',
            priceMonthly: new Prisma.Decimal(29.99),
            priceYearly: new Prisma.Decimal(299.99),
            features: { signals: 100, portfolios: 5, backtests: 50, aiEnabled: true },
            maxPortfolios: 5,
            maxWatchlists: 10,
            maxAlerts: 50,
            maxBacktests: 50,
            aiSignalsEnabled: true,
            ragEnabled: true,
            sortOrder: 2
        },
        {
            name: 'Enterprise',
            slug: 'enterprise',
            description: 'Full institutional-grade features',
            priceMonthly: new Prisma.Decimal(99.99),
            priceYearly: new Prisma.Decimal(999.99),
            features: { signals: -1, portfolios: -1, backtests: -1, aiEnabled: true, quantumEnabled: true, swarmEnabled: true },
            maxPortfolios: 100,
            maxWatchlists: 100,
            maxAlerts: 500,
            maxBacktests: 500,
            aiSignalsEnabled: true,
            quantumEnabled: true,
            swarmEnabled: true,
            ragEnabled: true,
            apiAccessEnabled: true,
            prioritySupport: true,
            sortOrder: 3
        }
    ];

    return await prisma.subscriptionPlan.createMany({ data: plans });
}

// ============================================
// USERS
// ============================================

async function createUsers() {
    const hashedPassword = await bcrypt.hash('Test123!', 10);

    const usersData = [
        {
            email: 'admin@neuraltrade.ai',
            name: 'Admin',
            surname: 'User',
            username: 'admin',
            password: hashedPassword,
            role: 'SUPER_ADMIN' as const,
            status: 'ACTIVE' as const,
            emailVerified: true,
            riskProfile: 'MODERATE' as const,
            tradingEnabled: true
        },
        {
            email: 'trader1@test.com',
            name: 'John',
            surname: 'Trader',
            username: 'trader1',
            password: hashedPassword,
            role: 'USER' as const,
            status: 'ACTIVE' as const,
            emailVerified: true,
            riskProfile: 'AGGRESSIVE' as const,
            tradingEnabled: true,
            maxDailyLoss: new Prisma.Decimal(1000),
            maxPositionSize: new Prisma.Decimal(10000),
            maxLeverage: new Prisma.Decimal(5)
        },
        {
            email: 'trader2@test.com',
            name: 'Jane',
            surname: 'Investor',
            username: 'trader2',
            password: hashedPassword,
            role: 'USER' as const,
            status: 'ACTIVE' as const,
            emailVerified: true,
            riskProfile: 'CONSERVATIVE' as const,
            tradingEnabled: true
        },
        {
            email: 'quant@test.com',
            name: 'Quant',
            surname: 'Developer',
            username: 'quantdev',
            password: hashedPassword,
            role: 'NEURALTRADE' as const,
            status: 'ACTIVE' as const,
            emailVerified: true,
            riskProfile: 'MODERATE' as const,
            tradingEnabled: true
        },
        {
            email: 'demo@neuraltrade.ai',
            name: 'Demo',
            surname: 'Account',
            username: 'demo',
            password: hashedPassword,
            role: 'USER' as const,
            status: 'ACTIVE' as const,
            emailVerified: true,
            riskProfile: 'MODERATE' as const,
            tradingEnabled: true
        }
    ];

    const users = [];
    for (const userData of usersData) {
        const user = await prisma.user.create({ data: userData });
        users.push(user);
    }

    return users;
}

// ============================================
// SUBSCRIPTIONS
// ============================================

async function createSubscriptions(users: any[], _plans: any) {
    const planRecords = await prisma.subscriptionPlan.findMany();
    const now = new Date();
    const monthFromNow = new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);

    for (let i = 0; i < users.length; i++) {
        const plan = planRecords[i % planRecords.length];
        await prisma.subscription.create({
            data: {
                userId: users[i].id,
                planId: plan.id,
                status: 'ACTIVE',
                billingCycle: i % 2 === 0 ? 'MONTHLY' : 'YEARLY',
                currentPeriodStart: now,
                currentPeriodEnd: monthFromNow
            }
        });
    }
}

// ============================================
// ASSETS
// ============================================

async function createAssets() {
    const assetsData = [
        // Crypto
        { symbol: 'BTC-USD', name: 'Bitcoin', assetType: 'CRYPTO' as const, currency: 'USD', decimals: 8 },
        { symbol: 'ETH-USD', name: 'Ethereum', assetType: 'CRYPTO' as const, currency: 'USD', decimals: 8 },
        { symbol: 'SOL-USD', name: 'Solana', assetType: 'CRYPTO' as const, currency: 'USD', decimals: 8 },
        { symbol: 'BNB-USD', name: 'Binance Coin', assetType: 'CRYPTO' as const, currency: 'USD', decimals: 8 },
        { symbol: 'XRP-USD', name: 'Ripple', assetType: 'CRYPTO' as const, currency: 'USD', decimals: 8 },

        // Stocks
        { symbol: 'AAPL', name: 'Apple Inc.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Technology', exchange: 'NASDAQ' },
        { symbol: 'MSFT', name: 'Microsoft Corp.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Technology', exchange: 'NASDAQ' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Technology', exchange: 'NASDAQ' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Consumer Cyclical', exchange: 'NASDAQ' },
        { symbol: 'TSLA', name: 'Tesla Inc.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Automotive', exchange: 'NASDAQ' },
        { symbol: 'NVDA', name: 'NVIDIA Corp.', assetType: 'STOCK' as const, currency: 'USD', sector: 'Technology', exchange: 'NASDAQ' },

        // Forex
        { symbol: 'EUR-USD', name: 'Euro / US Dollar', assetType: 'FOREX' as const, currency: 'USD' },
        { symbol: 'GBP-USD', name: 'British Pound / US Dollar', assetType: 'FOREX' as const, currency: 'USD' },
        { symbol: 'USD-JPY', name: 'US Dollar / Japanese Yen', assetType: 'FOREX' as const, currency: 'JPY' },

        // Commodities
        { symbol: 'GOLD', name: 'Gold', assetType: 'COMMODITY' as const, currency: 'USD' },
        { symbol: 'SILVER', name: 'Silver', assetType: 'COMMODITY' as const, currency: 'USD' },
        { symbol: 'OIL', name: 'Crude Oil', assetType: 'COMMODITY' as const, currency: 'USD' },

        // ETFs
        { symbol: 'SPY', name: 'SPDR S&P 500 ETF', assetType: 'ETF' as const, currency: 'USD', exchange: 'NYSE' },
        { symbol: 'QQQ', name: 'Invesco QQQ Trust', assetType: 'ETF' as const, currency: 'USD', exchange: 'NASDAQ' }
    ];

    const assets = [];
    for (const assetData of assetsData) {
        const asset = await prisma.asset.create({ data: assetData });
        assets.push(asset);
    }

    return assets;
}

// ============================================
// PORTFOLIOS
// ============================================

async function createPortfolios(users: any[]) {
    const portfolios = [];

    for (const user of users) {
        // Default portfolio
        const defaultPortfolio = await prisma.portfolio.create({
            data: {
                userId: user.id,
                name: 'Main Portfolio',
                description: 'Primary trading portfolio',
                isDefault: true,
                currency: 'USD',
                totalValue: randomDecimal(10000, 100000),
                totalCost: randomDecimal(8000, 90000),
                totalPnL: randomDecimal(-5000, 15000),
                totalPnLPercent: randomDecimal(-10, 25),
                riskScore: randomDecimal(1, 10, 2),
                sharpeRatio: randomDecimal(0.5, 3, 4),
                maxDrawdown: randomDecimal(0.05, 0.25, 4)
            }
        });
        portfolios.push(defaultPortfolio);

        // Additional portfolio for some users
        if (user.role !== 'USER' || Math.random() > 0.5) {
            const secondPortfolio = await prisma.portfolio.create({
                data: {
                    userId: user.id,
                    name: 'Crypto Portfolio',
                    description: 'Cryptocurrency investments',
                    isDefault: false,
                    currency: 'USD',
                    totalValue: randomDecimal(5000, 50000),
                    totalCost: randomDecimal(4000, 45000),
                    totalPnL: randomDecimal(-2000, 10000),
                    totalPnLPercent: randomDecimal(-15, 30)
                }
            });
            portfolios.push(secondPortfolio);
        }
    }

    return portfolios;
}

// ============================================
// POSITIONS
// ============================================

async function createPositions(portfolios: any[], assets: any[]) {
    for (const portfolio of portfolios) {
        const numPositions = randomInt(3, 8);
        const selectedAssets = assets.sort(() => Math.random() - 0.5).slice(0, numPositions);

        for (const asset of selectedAssets) {
            const side = Math.random() > 0.3 ? 'LONG' : 'SHORT';
            const quantity = randomDecimal(0.1, 100, 8);
            const avgCost = randomDecimal(10, 50000, 8);
            const currentPrice = avgCost.mul(randomDecimal(0.8, 1.3, 4));
            const marketValue = quantity.mul(currentPrice);
            const unrealizedPnL = marketValue.sub(quantity.mul(avgCost));

            await prisma.position.create({
                data: {
                    portfolioId: portfolio.id,
                    symbol: asset.symbol,
                    assetType: asset.assetType,
                    quantity: quantity,
                    avgCost: avgCost,
                    currentPrice: currentPrice,
                    marketValue: marketValue,
                    unrealizedPnL: unrealizedPnL,
                    unrealizedPct: unrealizedPnL.div(quantity.mul(avgCost)).mul(100),
                    allocation: randomDecimal(5, 30, 4),
                    side: side,
                    leverage: side === 'SHORT' ? randomDecimal(1, 3, 2) : null
                }
            });
        }
    }
}

// ============================================
// ORDERS & TRADES
// ============================================

async function createOrdersAndTrades(users: any[], portfolios: any[], assets: any[]) {
    const orderStatuses = ['PENDING', 'OPEN', 'FILLED', 'PARTIAL', 'CANCELLED'] as const;
    const orderTypes = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'] as const;
    const orderSides = ['BUY', 'SELL'] as const;

    for (const user of users) {
        const userPortfolios = portfolios.filter(p => p.userId === user.id);
        const numOrders = randomInt(5, 20);

        for (let i = 0; i < numOrders; i++) {
            const asset = randomElement(assets);
            const side = randomElement(orderSides);
            const type = randomElement(orderTypes);
            const status = randomElement(orderStatuses);
            const quantity = randomDecimal(0.01, 10, 8);
            const price = randomDecimal(10, 50000, 8);

            const order = await prisma.order.create({
                data: {
                    userId: user.id,
                    portfolioId: userPortfolios.length > 0 ? randomElement(userPortfolios).id : null,
                    symbol: asset.symbol,
                    assetType: asset.assetType,
                    side: side,
                    type: type,
                    status: status,
                    quantity: quantity,
                    filledQty: status === 'FILLED' ? quantity : (status === 'PARTIAL' ? quantity.mul(0.5) : new Prisma.Decimal(0)),
                    price: type !== 'MARKET' ? price : null,
                    avgFillPrice: status === 'FILLED' ? price : null,
                    timeInForce: 'GTC',
                    source: Math.random() > 0.7 ? 'AI_SIGNAL' : 'MANUAL',
                    createdAt: randomDate(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), new Date())
                }
            });

            // Create trade for filled orders
            if (status === 'FILLED') {
                await prisma.trade.create({
                    data: {
                        userId: user.id,
                        portfolioId: order.portfolioId,
                        orderId: order.id,
                        symbol: asset.symbol,
                        assetType: asset.assetType,
                        side: side,
                        quantity: quantity,
                        price: price,
                        value: quantity.mul(price),
                        commission: quantity.mul(price).mul(0.001),
                        realizedPnL: side === 'SELL' ? randomDecimal(-500, 1000) : null,
                        executedAt: order.createdAt
                    }
                });
            }
        }
    }
}

// ============================================
// AI SIGNALS
// ============================================

async function createAISignals(users: any[], assets: any[]) {
    const signalTypes = ['ENTRY', 'EXIT', 'SCALE_IN', 'SCALE_OUT', 'RISK_ALERT'] as const;
    const directions = ['BULLISH', 'BEARISH', 'NEUTRAL'] as const;
    const strengths = ['WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG'] as const;
    const sources = ['RAG', 'SWARM', 'QUANTUM', 'ML_PREDICTION', 'TECHNICAL', 'ENSEMBLE'] as const;
    const outcomes = ['PROFITABLE', 'LOSS', 'BREAKEVEN', 'EXPIRED'] as const;

    for (let i = 0; i < 50; i++) {
        const asset = randomElement(assets);
        const direction = randomElement(directions);
        const entryPrice = randomDecimal(10, 50000, 8);

        await prisma.aISignal.create({
            data: {
                userId: i < 30 ? randomElement(users).id : null,
                symbol: asset.symbol,
                signalType: randomElement(signalTypes),
                direction: direction,
                confidence: randomDecimal(0.5, 0.99, 4),
                strength: randomElement(strengths),
                entryPrice: entryPrice,
                targetPrice: direction === 'BULLISH' ? entryPrice.mul(1.1) : entryPrice.mul(0.9),
                stopLoss: direction === 'BULLISH' ? entryPrice.mul(0.95) : entryPrice.mul(1.05),
                riskReward: randomDecimal(1.5, 4, 2),
                timeframe: randomElement(['15m', '1h', '4h', '1d']),
                reasoning: `AI analysis suggests ${direction.toLowerCase()} momentum based on technical and fundamental factors.`,
                source: randomElement(sources),
                modelVersion: 'v2.0.0',
                isActive: i < 20,
                wasActedUpon: Math.random() > 0.6,
                outcome: i >= 20 ? randomElement(outcomes) : null,
                actualReturn: i >= 20 ? randomDecimal(-0.15, 0.25, 4) : null,
                createdAt: randomDate(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), new Date())
            }
        });
    }

    // Signal History
    for (let i = 0; i < 100; i++) {
        const asset = randomElement(assets);
        await prisma.signalHistory.create({
            data: {
                userId: Math.random() > 0.3 ? randomElement(users).id : null,
                symbol: asset.symbol,
                action: randomElement(['BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL']),
                confidence: randomDecimal(0.4, 0.99, 4),
                targetPrice: randomDecimal(10, 50000, 8),
                stopLoss: randomDecimal(8, 45000, 8),
                takeProfit: randomDecimal(12, 55000, 8),
                reasoning: 'Multi-factor analysis with ensemble model consensus.',
                modelUsed: randomElement(['ensemble', 'transformer', 'lstm', 'gradient_boost']),
                regime: randomElement(['bull_quiet', 'bull_volatile', 'bear_quiet', 'ranging']),
                volatility: randomDecimal(0.01, 0.1, 6),
                riskRewardRatio: randomDecimal(1.2, 4, 2),
                expectedReturn: randomDecimal(-0.1, 0.2, 6),
                contributors: { technical: 0.4, fundamental: 0.3, sentiment: 0.3 },
                timestamp: randomDate(new Date(Date.now() - 14 * 24 * 60 * 60 * 1000), new Date())
            }
        });
    }
}

// ============================================
// STRATEGIES & BACKTESTS
// ============================================

async function createStrategiesAndBacktests(users: any[]) {
    const strategyTypes = ['TREND_FOLLOWING', 'MEAN_REVERSION', 'MOMENTUM', 'BREAKOUT', 'MACHINE_LEARNING'] as const;

    // Create strategies
    const strategies = [];
    for (let i = 0; i < 10; i++) {
        const strategy = await prisma.strategy.create({
            data: {
                userId: i < 7 ? randomElement(users).id : null,
                name: `Strategy ${i + 1} - ${randomElement(strategyTypes).replace('_', ' ')}`,
                description: 'Automated trading strategy with dynamic risk management.',
                type: randomElement(strategyTypes),
                config: { lookbackPeriod: randomInt(10, 60), threshold: Math.random() * 0.1 },
                entryRules: { indicator: 'RSI', condition: 'below', value: 30 },
                exitRules: { indicator: 'RSI', condition: 'above', value: 70 },
                riskParams: { stopLoss: 0.02, takeProfit: 0.04, maxPositionSize: 0.1 },
                isPublic: i > 7,
                isTemplate: i === 0
            }
        });
        strategies.push(strategy);
    }

    // Create backtests
    for (const user of users) {
        const numBacktests = randomInt(1, 5);
        for (let i = 0; i < numBacktests; i++) {
            const strategy = randomElement(strategies);
            const status = randomElement(['PENDING', 'RUNNING', 'COMPLETED', 'FAILED'] as const);
            const initialCapital = randomDecimal(10000, 100000);

            await prisma.backtest.create({
                data: {
                    userId: user.id,
                    strategyId: strategy.id,
                    name: `Backtest ${Date.now()}-${i}`,
                    status: status,
                    symbols: ['BTC-USD', 'ETH-USD', 'AAPL'].slice(0, randomInt(1, 3)),
                    timeframe: randomElement(['15m', '1h', '4h', '1d']),
                    startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
                    endDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                    initialCapital: initialCapital,
                    commission: new Prisma.Decimal(0.001),
                    slippage: new Prisma.Decimal(0.0005),
                    finalCapital: status === 'COMPLETED' ? initialCapital.mul(randomDecimal(0.8, 1.5, 4)) : null,
                    totalReturn: status === 'COMPLETED' ? randomDecimal(-0.2, 0.5, 6) : null,
                    sharpeRatio: status === 'COMPLETED' ? randomDecimal(0.5, 2.5, 4) : null,
                    sortinoRatio: status === 'COMPLETED' ? randomDecimal(0.6, 3, 4) : null,
                    maxDrawdown: status === 'COMPLETED' ? randomDecimal(0.05, 0.3, 6) : null,
                    winRate: status === 'COMPLETED' ? randomDecimal(0.4, 0.7, 4) : null,
                    profitFactor: status === 'COMPLETED' ? randomDecimal(1.1, 2.5, 4) : null,
                    totalTrades: status === 'COMPLETED' ? randomInt(50, 500) : null,
                    executionTime: status === 'COMPLETED' ? randomInt(1000, 30000) : null
                }
            });
        }
    }
}

// ============================================
// WATCHLISTS & ALERTS
// ============================================

async function createWatchlistsAndAlerts(users: any[], assets: any[]) {
    const alertTypes = ['PRICE', 'VOLUME', 'AI_SIGNAL', 'PORTFOLIO', 'POSITION'] as const;

    for (const user of users) {
        // Watchlists
        await prisma.watchlist.create({
            data: {
                userId: user.id,
                name: 'Default Watchlist',
                symbols: assets.slice(0, 5).map(a => a.symbol),
                isDefault: true
            }
        });

        await prisma.watchlist.create({
            data: {
                userId: user.id,
                name: 'Tech Stocks',
                symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                isDefault: false
            }
        });

        // Alerts
        for (let i = 0; i < randomInt(3, 8); i++) {
            const asset = randomElement(assets);
            await prisma.alert.create({
                data: {
                    userId: user.id,
                    symbol: asset.symbol,
                    alertType: randomElement(alertTypes),
                    condition: { operator: randomElement(['above', 'below']), value: Math.random() * 50000 },
                    message: `Alert triggered for ${asset.symbol}`,
                    priority: randomElement(['LOW', 'MEDIUM', 'HIGH', 'URGENT'] as const),
                    isActive: Math.random() > 0.3,
                    notifyEmail: Math.random() > 0.5,
                    notifyPush: true
                }
            });
        }

        // Notifications
        for (let i = 0; i < randomInt(5, 15); i++) {
            await prisma.notification.create({
                data: {
                    userId: user.id,
                    type: randomElement(['INFO', 'SUCCESS', 'WARNING', 'AI_SIGNAL', 'TRADE', 'ALERT'] as const),
                    title: `Notification ${i + 1}`,
                    message: 'This is a sample notification message for testing purposes.',
                    isRead: Math.random() > 0.5,
                    createdAt: randomDate(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), new Date())
                }
            });
        }
    }
}

// ============================================
// LEDGER DATA
// ============================================

async function createLedgerData(users: any[]) {
    for (const user of users) {
        // Ledger Accounts
        const walletAccount = await prisma.ledgerAccount.create({
            data: {
                userId: user.id,
                accountType: 'WALLET',
                currency: 'USD',
                balance: randomDecimal(1000, 50000, 8),
                lockedBalance: randomDecimal(0, 1000, 8)
            }
        });

        const tradingAccount = await prisma.ledgerAccount.create({
            data: {
                userId: user.id,
                accountType: 'TRADING',
                currency: 'USD',
                balance: randomDecimal(5000, 100000, 8),
                lockedBalance: randomDecimal(0, 5000, 8)
            }
        });

        // Transactions
        for (let i = 0; i < randomInt(5, 15); i++) {
            const txType = randomElement(['DEPOSIT', 'WITHDRAW', 'TRANSFER', 'FEE'] as const);
            const amount = randomDecimal(100, 5000, 8);

            const transaction = await prisma.transaction.create({
                data: {
                    userId: user.id,
                    type: txType,
                    status: randomElement(['PENDING', 'COMPLETED', 'FAILED'] as const),
                    amount: amount,
                    fee: amount.mul(0.001),
                    currency: 'USD',
                    createdAt: randomDate(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), new Date())
                }
            });

            // Ledger entries for completed transactions
            if (transaction.status === 'COMPLETED') {
                await prisma.ledgerEntry.create({
                    data: {
                        accountId: txType === 'DEPOSIT' ? walletAccount.id : tradingAccount.id,
                        transactionId: transaction.id,
                        entryType: txType === 'DEPOSIT' ? 'CREDIT' : 'DEBIT',
                        amount: amount,
                        balanceAfter: randomDecimal(1000, 100000, 8),
                        description: `${txType} transaction`
                    }
                });
            }
        }
    }
}

// ============================================
// MARKET DATA
// ============================================

async function createMarketData(assets: any[]) {
    // Technical Analysis
    for (const asset of assets.slice(0, 10)) {
        for (const timeframe of ['15m', '1h', '4h', '1d']) {
            await prisma.technicalAnalysis.create({
                data: {
                    symbol: asset.symbol,
                    timeframe: timeframe,
                    price: randomDecimal(10, 50000, 8),
                    rsi: randomDecimal(20, 80, 4),
                    macd: { macd: Math.random() * 100 - 50, signal: Math.random() * 100 - 50, histogram: Math.random() * 50 - 25 },
                    bollingerBands: { upper: Math.random() * 1000, middle: Math.random() * 900, lower: Math.random() * 800 },
                    atr: randomDecimal(1, 100, 8),
                    volume: randomDecimal(1000000, 100000000, 2),
                    trend: randomElement(['UPTREND', 'DOWNTREND', 'SIDEWAYS'] as const),
                    fvgCount: randomInt(0, 5),
                    supports: [Math.random() * 40000, Math.random() * 35000],
                    resistances: [Math.random() * 45000, Math.random() * 50000]
                }
            });
        }
    }

    // Regime Detection
    for (const asset of assets.slice(0, 5)) {
        await prisma.regimeDetection.create({
            data: {
                symbol: asset.symbol,
                regime: randomElement(['BULL_QUIET', 'BULL_VOLATILE', 'BEAR_QUIET', 'BEAR_VOLATILE', 'RANGING'] as const),
                confidence: randomDecimal(0.6, 0.95, 4),
                volatility: randomDecimal(0.01, 0.1, 6),
                trend: randomElement(['UPTREND', 'DOWNTREND', 'SIDEWAYS'] as const),
                momentum: randomDecimal(-2, 2, 4)
            }
        });
    }

    // Risk Metrics
    for (const asset of assets.slice(0, 10)) {
        await prisma.riskMetrics.create({
            data: {
                symbol: asset.symbol,
                volatility: randomDecimal(0.01, 0.1, 6),
                var95: randomDecimal(0.01, 0.05, 6),
                var99: randomDecimal(0.02, 0.08, 6),
                cvar95: randomDecimal(0.015, 0.07, 6),
                cvar99: randomDecimal(0.025, 0.1, 6),
                sharpeRatio: randomDecimal(0.5, 2.5, 4),
                sortinoRatio: randomDecimal(0.6, 3, 4),
                maxDrawdown: randomDecimal(0.05, 0.3, 6),
                skewness: randomDecimal(-1, 1, 4),
                kurtosis: randomDecimal(2, 6, 4)
            }
        });
    }

    // Swarm Consensus
    for (let i = 0; i < 20; i++) {
        const asset = randomElement(assets);
        const bullScore = randomDecimal(0, 1, 4);
        const bearScore = randomDecimal(0, 1, 4);
        const total = bullScore.add(bearScore);

        await prisma.swarmConsensus.create({
            data: {
                symbol: asset.symbol,
                bullScore: bullScore.div(total.add(0.1)),
                bearScore: bearScore.div(total.add(0.1)),
                neutralScore: new Prisma.Decimal(0.1).div(total.add(0.1)),
                finalDecision: bullScore.gt(bearScore) ? 'BULLISH' : 'BEARISH',
                judgeReasoning: 'Swarm agents reached consensus based on multi-factor analysis.',
                agentVotes: { technical: 'BULLISH', fundamental: 'NEUTRAL', sentiment: 'BULLISH' },
                confidence: randomDecimal(0.6, 0.95, 4)
            }
        });
    }
}

// ============================================
// SYSTEM DATA
// ============================================

async function createSystemData(users: any[]) {
    // Jobs
    for (let i = 0; i < 20; i++) {
        await prisma.job.create({
            data: {
                name: `Job ${i + 1}`,
                type: randomElement(['AI_ANALYSIS', 'MARKET_DATA', 'PORTFOLIO_SYNC', 'BACKTEST', 'NOTIFICATION'] as const),
                status: randomElement(['PENDING', 'RUNNING', 'COMPLETED', 'FAILED'] as const),
                priority: randomInt(0, 10),
                payload: { taskId: `task-${i}`, params: {} },
                progress: randomInt(0, 100),
                createdAt: randomDate(new Date(Date.now() - 24 * 60 * 60 * 1000), new Date())
            }
        });
    }

    // System Metrics
    const metricNames = ['api_requests', 'db_queries', 'ai_predictions', 'websocket_connections', 'cpu_usage', 'memory_usage'];
    for (const metricName of metricNames) {
        for (let i = 0; i < 24; i++) {
            await prisma.systemMetric.create({
                data: {
                    metricName: metricName,
                    value: randomDecimal(0, 1000, 6),
                    labels: { environment: 'production', service: 'api' },
                    timestamp: new Date(Date.now() - i * 60 * 60 * 1000)
                }
            });
        }
    }

    // Audit Logs
    const actions = ['LOGIN', 'LOGOUT', 'CREATE_ORDER', 'CANCEL_ORDER', 'UPDATE_PORTFOLIO', 'CHANGE_PASSWORD'];
    for (let i = 0; i < 50; i++) {
        await prisma.auditLog.create({
            data: {
                userId: Math.random() > 0.2 ? randomElement(users).id : null,
                action: randomElement(actions),
                entity: randomElement(['User', 'Order', 'Portfolio', 'Position', 'Alert']),
                entityId: String(randomInt(1, 100)),
                ipAddress: `192.168.1.${randomInt(1, 255)}`,
                userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                createdAt: randomDate(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), new Date())
            }
        });
    }

    // Economic Events
    const events = [
        { name: 'FOMC Meeting', country: 'US', impact: 'HIGH' as const },
        { name: 'CPI Release', country: 'US', impact: 'HIGH' as const },
        { name: 'NFP Report', country: 'US', impact: 'HIGH' as const },
        { name: 'ECB Rate Decision', country: 'EU', impact: 'HIGH' as const },
        { name: 'GDP Growth', country: 'US', impact: 'MEDIUM' as const },
        { name: 'Retail Sales', country: 'US', impact: 'MEDIUM' as const }
    ];

    for (const event of events) {
        await prisma.economicEvent.create({
            data: {
                name: event.name,
                country: event.country,
                impact: event.impact,
                forecast: `${(Math.random() * 5).toFixed(1)}%`,
                previous: `${(Math.random() * 5).toFixed(1)}%`,
                eventTime: randomDate(new Date(), new Date(Date.now() + 30 * 24 * 60 * 60 * 1000))
            }
        });
    }
}

// ============================================
// RUN MAIN
// ============================================

main()
    .catch((e) => {
        console.error('❌ Seed failed:', e);
        process.exit(1);
    })
    .finally(async () => {
        await prisma.$disconnect();
    });
