// ==========================================
// Exchange Types & Interfaces
// ==========================================

/**
 * Supported exchange types
 */
export type ExchangeType = 'binance' | 'bybit' | 'mock';

/**
 * Exchange configuration
 */
export interface ExchangeConfig {
    apiKey?: string;
    secret?: string;
    sandbox?: boolean;
    timeout?: number;
    rateLimit?: boolean;
}

// ==========================================
// Market Data Types
// ==========================================

export interface TickerData {
    symbol: string;
    last: string;          // Use string for decimal precision
    bid: string;
    ask: string;
    high: string;
    low: string;
    volume: string;
    change: string;
    changePercent: string;
    timestamp: number;
}

export interface OrderBookEntry {
    price: string;
    amount: string;
}

export interface OrderBook {
    symbol: string;
    bids: OrderBookEntry[];
    asks: OrderBookEntry[];
    timestamp: number;
}

export interface OHLCV {
    timestamp: number;
    open: string;
    high: string;
    low: string;
    close: string;
    volume: string;
}

// ==========================================
// Trading Types
// ==========================================

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop_limit';
export type OrderStatus = 'pending' | 'open' | 'filled' | 'partial' | 'cancelled' | 'rejected';

export interface OrderRequest {
    symbol: string;
    side: OrderSide;
    type: OrderType;
    amount: string;        // String for decimal precision
    price?: string;        // Required for limit orders
    stopPrice?: string;    // For stop orders
    clientOrderId?: string;
}

export interface OrderResponse {
    id: string;
    clientOrderId?: string;
    symbol: string;
    side: OrderSide;
    type: OrderType;
    status: OrderStatus;
    amount: string;
    filled: string;
    remaining: string;
    price?: string;
    avgFillPrice?: string;
    fee?: string;
    feeCurrency?: string;
    timestamp: number;
    updatedAt?: number;
}

// ==========================================
// Account Types
// ==========================================

export interface BalanceInfo {
    [currency: string]: {
        free: string;
        locked: string;
        total: string;
    };
}

// ==========================================
// Exchange Adapter Interface
// ==========================================

/**
 * IExchangeAdapter - Core interface for all exchange adapters
 * 
 * All implementations (Binance, Bybit, Mock) must implement this interface.
 * This enables seamless switching between exchanges without changing business logic.
 */
export interface IExchangeAdapter {
    // ==========================================
    // Metadata
    // ==========================================

    /**
     * Get exchange identifier
     */
    getExchangeId(): ExchangeType;

    /**
     * Check if running in sandbox/testnet mode
     */
    isSandboxMode(): boolean;

    /**
     * Initialize the exchange connection
     */
    initialize(): Promise<void>;

    // ==========================================
    // Market Data
    // ==========================================

    /**
     * Fetch current ticker data for a symbol
     */
    fetchTicker(symbol: string): Promise<TickerData | null>;

    /**
     * Fetch order book for a symbol
     */
    fetchOrderBook(symbol: string, limit?: number): Promise<OrderBook | null>;

    /**
     * Fetch OHLCV candlestick data
     */
    fetchOHLCV(symbol: string, timeframe: string, limit?: number): Promise<OHLCV[]>;

    // ==========================================
    // Trading
    // ==========================================

    /**
     * Create a new order
     */
    createOrder(order: OrderRequest): Promise<OrderResponse>;

    /**
     * Cancel an existing order
     */
    cancelOrder(orderId: string, symbol: string): Promise<boolean>;

    /**
     * Fetch a specific order by ID
     */
    fetchOrder(orderId: string, symbol: string): Promise<OrderResponse | null>;

    /**
     * Fetch all open orders
     */
    fetchOpenOrders(symbol?: string): Promise<OrderResponse[]>;

    // ==========================================
    // Account
    // ==========================================

    /**
     * Fetch account balances
     */
    fetchBalance(): Promise<BalanceInfo>;
}

// ==========================================
// Adapter Symbol for DI
// ==========================================

export const EXCHANGE_ADAPTER = Symbol('EXCHANGE_ADAPTER');
