"""
Institutional Backtest Engine
Author: Erdinc Erdogan
Purpose: Event-driven backtesting framework with walk-forward optimization, Monte Carlo simulation, transaction cost modeling, and comprehensive performance analytics.
References:
- Event-Driven Backtesting Architecture
- Walk-Forward Optimization
- Monte Carlo Simulation for Strategy Robustness
Usage:
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run(strategy, price_data)
    metrics = engine.calculate_performance_metrics()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from datetime import datetime, timedelta
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class BacktestMode(Enum):
    """Backtest execution mode"""
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    WALK_FORWARD = "walk_forward"


@dataclass
class Order:
    """Order representation"""
    order_id: int
    timestamp: Any
    asset: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: str = "pending"


@dataclass
class Trade:
    """Executed trade"""
    trade_id: int
    order_id: int
    timestamp: Any
    asset: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0
    pnl_percent: float = 0.0


@dataclass
class Position:
    """Position tracking"""
    asset: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    cost_basis: float
    status: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    expectancy: float
    sqn: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    turnover: float


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis results"""
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    current_drawdown: float
    drawdown_series: np.ndarray
    underwater_periods: int
    recovery_factor: float


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result"""
    in_sample_metrics: List[PerformanceMetrics]
    out_sample_metrics: List[PerformanceMetrics]
    combined_metrics: PerformanceMetrics
    optimal_parameters: List[Dict]
    robustness_ratio: float
    degradation: float


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    mean_return: float
    median_return: float
    std_return: float
    var_95: float
    cvar_95: float
    probability_profit: float
    probability_target: float
    max_drawdown_mean: float
    max_drawdown_95: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    simulation_paths: np.ndarray


@dataclass
class BacktestResult:
    """Complete backtest result"""
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: List[Position]
    trades: List[Trade]
    orders: List[Order]
    metrics: PerformanceMetrics
    drawdown_analysis: DrawdownAnalysis
    daily_returns: np.ndarray
    monthly_returns: Optional[np.ndarray]
    yearly_returns: Optional[np.ndarray]


class BacktestEngine(BaseModule):
    """
    Institutional-Grade Backtest Engine.
    
    Implements comprehensive backtesting framework with event-driven
    architecture, walk-forward optimization, and Monte Carlo simulation.
    
    Mathematical Framework:
    ----------------------
    
    Sharpe Ratio:
        SR = (R_p - R_f) / σ_p
        
        Annualized: SR_annual = SR_daily × √252
    
    Sortino Ratio:
        Sortino = (R_p - R_f) / σ_downside
        
        σ_downside = √(Σ min(R_i - R_f, 0)² / n)
    
    Calmar Ratio:
        Calmar = Annualized_Return / Max_Drawdown
    
    Maximum Drawdown:
        DD_t = (Peak_t - Value_t) / Peak_t
        MDD = max(DD_t)
    
    Profit Factor:
        PF = Σ Winning_Trades / |Σ Losing_Trades|
    
    Expectancy:
        E = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss)
    
    System Quality Number (SQN):
        SQN = √n × (Avg_Trade / Std_Trade)
    
    Value at Risk (VaR):
        VaR_α = -Percentile(Returns, α)
    
    Conditional VaR (CVaR):
        CVaR_α = -E[R | R ≤ -VaR_α]
    
    Walk-Forward Efficiency:
        WFE = OOS_Performance / IS_Performance
    
    Monte Carlo VaR:
        MC_VaR = Percentile(Simulated_Returns, α)
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.initial_capital: float = self.config.get('initial_capital', 1000000.0)
        self.commission_rate: float = self.config.get('commission_rate', 0.001)
        self.slippage_rate: float = self.config.get('slippage_rate', 0.0005)
        self.risk_free_rate: float = self.config.get('risk_free_rate', 0.02)
        self.trading_days: int = self.config.get('trading_days', 252)
        
        self._reset_state()
    
    def _reset_state(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.order_id_counter = 0
        self.trade_id_counter = 0
    
    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================
    
    def create_order(
        self,
        timestamp: Any,
        asset: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Create a new order."""
        self.order_id_counter += 1
        
        order = Order(
            order_id=self.order_id_counter,
            timestamp=timestamp,
            asset=asset,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        self.orders.append(order)
        return order
    
    def execute_order(
        self,
        order: Order,
        execution_price: float,
        timestamp: Any
    ) -> Trade:
        """Execute an order and create a trade."""
        commission = abs(order.quantity * execution_price * self.commission_rate)
        
        if order.side == OrderSide.BUY.value:
            slippage = execution_price * self.slippage_rate
            fill_price = execution_price + slippage
        else:
            slippage = execution_price * self.slippage_rate
            fill_price = execution_price - slippage
        
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = commission
        order.slippage = abs(slippage * order.quantity)
        order.status = "filled"
        
        self.trade_id_counter += 1
        
        trade = Trade(
            trade_id=self.trade_id_counter,
            order_id=order.order_id,
            timestamp=timestamp,
            asset=order.asset,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=order.slippage
        )
        
        self._update_position(trade)
        self.trades.append(trade)
        
        return trade
    
    def _update_position(self, trade: Trade):
        """Update position after trade execution."""
        asset = trade.asset
        
        if asset not in self.positions:
            self.positions[asset] = Position(
                asset=asset,
                quantity=0.0,
                avg_entry_price=0.0,
                current_price=trade.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                market_value=0.0,
                cost_basis=0.0,
                status=PositionStatus.CLOSED.value
            )
        
        pos = self.positions[asset]
        
        if trade.side == OrderSide.BUY.value:
            new_quantity = pos.quantity + trade.quantity
            if new_quantity != 0:
                pos.avg_entry_price = (
                    (pos.quantity * pos.avg_entry_price + trade.quantity * trade.price) / new_quantity
                )
            pos.quantity = new_quantity
            pos.cost_basis += trade.quantity * trade.price + trade.commission
            self.capital -= trade.quantity * trade.price + trade.commission
        else:
            if pos.quantity > 0:
                realized = (trade.price - pos.avg_entry_price) * trade.quantity - trade.commission
                pos.realized_pnl += realized
                trade.pnl = realized
                trade.pnl_percent = realized / (pos.avg_entry_price * trade.quantity) * 100
            
            pos.quantity -= trade.quantity
            self.capital += trade.quantity * trade.price - trade.commission
        
        pos.current_price = trade.price
        pos.market_value = pos.quantity * pos.current_price
        pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.quantity if pos.quantity > 0 else 0
        pos.status = PositionStatus.OPEN.value if pos.quantity != 0 else PositionStatus.CLOSED.value
    
    def update_positions_mark_to_market(self, prices: Dict[str, float]):
        """Update all positions with current market prices."""
        total_market_value = self.capital
        
        for asset, price in prices.items():
            if asset in self.positions:
                pos = self.positions[asset]
                pos.current_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity if pos.quantity > 0 else 0
                total_market_value += pos.market_value
        
        self.equity = total_market_value
        self.equity_curve.append(self.equity)
    
    # =========================================================================
    # VECTORIZED BACKTEST
    # =========================================================================
    
    def run_vectorized_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        position_size: float = 1.0
    ) -> BacktestResult:
        """
        Run vectorized backtest.
        
        Fast backtesting using numpy operations.
        """
        self._reset_state()
        
        prices = np.asarray(prices)
        signals = np.asarray(signals)
        n = len(prices)
        
        returns = np.diff(prices) / prices[:-1]
        
        positions = np.zeros(n)
        positions[1:] = signals[:-1] * position_size
        
        strategy_returns = positions[1:] * returns
        
        commission_costs = np.abs(np.diff(positions)) * self.commission_rate
        commission_costs = np.insert(commission_costs, 0, 0)
        
        slippage_costs = np.abs(np.diff(positions)) * self.slippage_rate
        slippage_costs = np.insert(slippage_costs, 0, 0)
        
        net_returns = np.zeros(n)
        net_returns[1:] = strategy_returns - commission_costs[1:] - slippage_costs[1:]
        
        equity_curve = self.initial_capital * np.cumprod(1 + net_returns)
        
        metrics = self._calculate_performance_metrics(net_returns[1:], equity_curve)
        drawdown_analysis = self._analyze_drawdowns(equity_curve)
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            positions=[],
            trades=[],
            orders=[],
            metrics=metrics,
            drawdown_analysis=drawdown_analysis,
            daily_returns=net_returns[1:],
            monthly_returns=None,
            yearly_returns=None
        )
    
    # =========================================================================
    # EVENT-DRIVEN BACKTEST
    # =========================================================================
    
    def run_event_driven_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        **strategy_params
    ) -> BacktestResult:
        """
        Run event-driven backtest.
        
        Process data bar-by-bar with strategy function.
        """
        self._reset_state()
        
        for idx, row in data.iterrows():
            signals = strategy_func(data.loc[:idx], **strategy_params)
            
            if signals:
                for signal in signals:
                    order = self.create_order(
                        timestamp=idx,
                        asset=signal['asset'],
                        side=OrderSide.BUY if signal['side'] == 'buy' else OrderSide.SELL,
                        quantity=signal['quantity'],
                        order_type=OrderType.MARKET
                    )
                    
                    self.execute_order(order, row['close'], idx)
            
            prices = {col.replace('_close', ''): row[col] 
                     for col in data.columns if '_close' in col or col == 'close'}
            if 'close' in row:
                prices['default'] = row['close']
            
            self.update_positions_mark_to_market(prices)
        
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = self._calculate_performance_metrics(returns, equity_curve)
        drawdown_analysis = self._analyze_drawdowns(equity_curve)
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=list(self.positions.values()),
            trades=self.trades,
            orders=self.orders,
            metrics=metrics,
            drawdown_analysis=drawdown_analysis,
            daily_returns=returns,
            monthly_returns=None,
            yearly_returns=None
        )
    
    # =========================================================================
    # WALK-FORWARD OPTIMIZATION
    # =========================================================================
    
    def run_walk_forward_optimization(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        optimization_metric: str = 'sharpe_ratio',
        n_splits: int = 5,
        train_ratio: float = 0.7
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.
        
        Split data into in-sample (training) and out-of-sample (testing) periods.
        """
        n = len(data)
        split_size = n // n_splits
        
        in_sample_metrics = []
        out_sample_metrics = []
        optimal_parameters = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n)
            
            split_data = data.iloc[start_idx:end_idx]
            train_size = int(len(split_data) * train_ratio)
            
            train_data = split_data.iloc[:train_size]
            test_data = split_data.iloc[train_size:]
            
            best_metric = -np.inf
            best_params = {}
            best_is_metrics = None
            
            param_combinations = self._generate_param_combinations(param_grid)
            
            for params in param_combinations:
                self._reset_state()
                
                try:
                    result = self._run_strategy_backtest(train_data, strategy_func, params)
                    metric_value = getattr(result.metrics, optimization_metric)
                    
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_params = params
                        best_is_metrics = result.metrics
                except Exception:
                    continue
            
            if best_is_metrics:
                in_sample_metrics.append(best_is_metrics)
                optimal_parameters.append(best_params)
                
                self._reset_state()
                try:
                    oos_result = self._run_strategy_backtest(test_data, strategy_func, best_params)
                    out_sample_metrics.append(oos_result.metrics)
                except Exception:
                    out_sample_metrics.append(best_is_metrics)
        
        if in_sample_metrics and out_sample_metrics:
            avg_is_sharpe = np.mean([m.sharpe_ratio for m in in_sample_metrics])
            avg_oos_sharpe = np.mean([m.sharpe_ratio for m in out_sample_metrics])
            
            robustness_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0
            degradation = 1 - robustness_ratio
            
            combined_returns = np.concatenate([
                np.random.normal(m.annualized_return, m.volatility, 100) 
                for m in out_sample_metrics
            ])
            combined_metrics = self._calculate_performance_metrics(
                combined_returns, 
                self.initial_capital * np.cumprod(1 + combined_returns)
            )
        else:
            robustness_ratio = 0
            degradation = 1
            combined_metrics = PerformanceMetrics(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, max_drawdown=0, max_drawdown_duration=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, avg_trade=0,
                total_trades=0, winning_trades=0, losing_trades=0, expectancy=0,
                sqn=0, var_95=0, cvar_95=0, skewness=0, kurtosis=0,
                best_trade=0, worst_trade=0, avg_holding_period=0, turnover=0
            )
        
        return WalkForwardResult(
            in_sample_metrics=in_sample_metrics,
            out_sample_metrics=out_sample_metrics,
            combined_metrics=combined_metrics,
            optimal_parameters=optimal_parameters,
            robustness_ratio=float(robustness_ratio),
            degradation=float(degradation)
        )
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        
        def generate(idx, current):
            if idx == len(keys):
                combinations.append(current.copy())
                return
            
            for val in values[idx]:
                current[keys[idx]] = val
                generate(idx + 1, current)
        
        generate(0, {})
        return combinations
    
    def _run_strategy_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict
    ) -> BacktestResult:
        """Run backtest with specific parameters."""
        signals = strategy_func(data, **params)
        
        if isinstance(signals, np.ndarray):
            prices = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
            return self.run_vectorized_backtest(prices, signals)
        else:
            return self.run_event_driven_backtest(data, strategy_func, **params)
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    
    def run_monte_carlo_simulation(
        self,
        returns: np.ndarray,
        n_simulations: int = 10000,
        n_periods: int = 252,
        target_return: float = 0.10
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Simulate future equity paths based on historical return distribution.
        """
        returns = np.asarray(returns)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulated_paths = np.zeros((n_simulations, n_periods + 1))
        simulated_paths[:, 0] = self.initial_capital
        
        for i in range(n_simulations):
            random_returns = np.random.normal(mean_return, std_return, n_periods)
            simulated_paths[i, 1:] = self.initial_capital * np.cumprod(1 + random_returns)
        
        final_values = simulated_paths[:, -1]
        total_returns = (final_values - self.initial_capital) / self.initial_capital
        
        max_drawdowns = np.zeros(n_simulations)
        for i in range(n_simulations):
            path = simulated_paths[i]
            running_max = np.maximum.accumulate(path)
            drawdowns = (running_max - path) / running_max
            max_drawdowns[i] = np.max(drawdowns)
        
        var_95 = np.percentile(total_returns, 5)
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        
        probability_profit = np.mean(total_returns > 0)
        probability_target = np.mean(total_returns >= target_return)
        
        confidence_intervals = {
            '90%': (np.percentile(total_returns, 5), np.percentile(total_returns, 95)),
            '95%': (np.percentile(total_returns, 2.5), np.percentile(total_returns, 97.5)),
            '99%': (np.percentile(total_returns, 0.5), np.percentile(total_returns, 99.5))
        }
        
        return MonteCarloResult(
            mean_return=float(np.mean(total_returns)),
            median_return=float(np.median(total_returns)),
            std_return=float(np.std(total_returns)),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            probability_profit=float(probability_profit),
            probability_target=float(probability_target),
            max_drawdown_mean=float(np.mean(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            confidence_intervals=confidence_intervals,
            simulation_paths=simulated_paths
        )
    
    def run_bootstrap_simulation(
        self,
        returns: np.ndarray,
        n_simulations: int = 10000,
        n_periods: int = 252,
        block_size: int = 20
    ) -> MonteCarloResult:
        """
        Run bootstrap simulation with block resampling.
        
        Preserves autocorrelation structure in returns.
        """
        returns = np.asarray(returns)
        n_returns = len(returns)
        
        simulated_paths = np.zeros((n_simulations, n_periods + 1))
        simulated_paths[:, 0] = self.initial_capital
        
        n_blocks = (n_periods + block_size - 1) // block_size
        
        for i in range(n_simulations):
            sampled_returns = []
            
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, max(1, n_returns - block_size))
                block = returns[start_idx:start_idx + block_size]
                sampled_returns.extend(block)
            
            sampled_returns = np.array(sampled_returns[:n_periods])
            simulated_paths[i, 1:] = self.initial_capital * np.cumprod(1 + sampled_returns)
        
        final_values = simulated_paths[:, -1]
        total_returns = (final_values - self.initial_capital) / self.initial_capital
        
        max_drawdowns = np.zeros(n_simulations)
        for i in range(n_simulations):
            path = simulated_paths[i]
            running_max = np.maximum.accumulate(path)
            drawdowns = (running_max - path) / running_max
            max_drawdowns[i] = np.max(drawdowns)
        
        var_95 = np.percentile(total_returns, 5)
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        
        return MonteCarloResult(
            mean_return=float(np.mean(total_returns)),
            median_return=float(np.median(total_returns)),
            std_return=float(np.std(total_returns)),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            probability_profit=float(np.mean(total_returns > 0)),
            probability_target=float(np.mean(total_returns >= 0.10)),
            max_drawdown_mean=float(np.mean(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            confidence_intervals={
                '90%': (np.percentile(total_returns, 5), np.percentile(total_returns, 95)),
                '95%': (np.percentile(total_returns, 2.5), np.percentile(total_returns, 97.5)),
                '99%': (np.percentile(total_returns, 0.5), np.percentile(total_returns, 99.5))
            },
            simulation_paths=simulated_paths
        )
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    def _calculate_performance_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        returns = np.asarray(returns)
        equity_curve = np.asarray(equity_curve)
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (self.trading_days / n_periods) - 1
        
        volatility = np.std(returns) * np.sqrt(self.trading_days)
        
        daily_rf = self.risk_free_rate / self.trading_days
        excess_returns = returns - daily_rf
        
        if volatility > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days)
        else:
            sharpe_ratio = 0.0
        
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(self.trading_days)
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = 0.0
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown = np.max(drawdowns)
        
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        max_dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        total_trades = len(returns[returns != 0])
        winning_trades = len(winning_returns)
        losing_trades = len(losing_returns)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        avg_trade = np.mean(returns)
        
        gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0
        gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        std_trade = np.std(returns)
        sqn = np.sqrt(total_trades) * (avg_trade / std_trade) if std_trade > 0 and total_trades > 0 else 0
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0
        
        best_trade = np.max(returns) if len(returns) > 0 else 0
        worst_trade = np.min(returns) if len(returns) > 0 else 0
        
        position_changes = np.sum(np.abs(np.diff(np.sign(returns))))
        turnover = position_changes / (2 * n_periods) if n_periods > 0 else 0
        
        return PerformanceMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(calmar_ratio),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_dd_duration),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            avg_trade=float(avg_trade),
            total_trades=int(total_trades),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            expectancy=float(expectancy),
            sqn=float(sqn),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            best_trade=float(best_trade),
            worst_trade=float(worst_trade),
            avg_holding_period=0.0,
            turnover=float(turnover)
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, max_drawdown=0, max_drawdown_duration=0,
            win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, avg_trade=0,
            total_trades=0, winning_trades=0, losing_trades=0, expectancy=0,
            sqn=0, var_95=0, cvar_95=0, skewness=0, kurtosis=0,
            best_trade=0, worst_trade=0, avg_holding_period=0, turnover=0
        )
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdowns > 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================
    
    def _analyze_drawdowns(self, equity_curve: np.ndarray) -> DrawdownAnalysis:
        """Analyze drawdowns in detail."""
        equity_curve = np.asarray(equity_curve)
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        
        max_drawdown = np.max(drawdowns)
        
        max_dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        
        in_drawdown = drawdowns > 0.001
        drawdown_periods = []
        current_dd = []
        
        for i, dd in enumerate(drawdowns):
            if in_drawdown[i]:
                current_dd.append(dd)
            elif current_dd:
                drawdown_periods.append(current_dd)
                current_dd = []
        
        if current_dd:
            drawdown_periods.append(current_dd)
        
        if drawdown_periods:
            avg_drawdown = np.mean([np.max(dd) for dd in drawdown_periods])
            avg_duration = np.mean([len(dd) for dd in drawdown_periods])
        else:
            avg_drawdown = 0
            avg_duration = 0
        
        current_drawdown = drawdowns[-1]
        
        underwater_periods = np.sum(in_drawdown)
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return DrawdownAnalysis(
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_dd_duration),
            avg_drawdown=float(avg_drawdown),
            avg_drawdown_duration=float(avg_duration),
            current_drawdown=float(current_drawdown),
            drawdown_series=drawdowns,
            underwater_periods=int(underwater_periods),
            recovery_factor=float(recovery_factor)
        )
    
    # =========================================================================
    # TRADE ANALYTICS
    # =========================================================================
    
    def analyze_trades(self, trades: List[Trade]) -> Dict:
        """Analyze trade statistics."""
        if not trades:
            return {}
        
        pnls = np.array([t.pnl for t in trades])
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_pnls = np.array([t.pnl for t in winning_trades]) if winning_trades else np.array([0])
        loss_pnls = np.array([t.pnl for t in losing_trades]) if losing_trades else np.array([0])
        
        consecutive_wins = self._max_consecutive(pnls > 0)
        consecutive_losses = self._max_consecutive(pnls < 0)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_win': float(np.mean(win_pnls)),
            'avg_loss': float(np.mean(loss_pnls)),
            'largest_win': float(np.max(win_pnls)),
            'largest_loss': float(np.min(loss_pnls)),
            'avg_trade': float(np.mean(pnls)),
            'total_pnl': float(np.sum(pnls)),
            'profit_factor': float(np.sum(win_pnls) / abs(np.sum(loss_pnls))) if np.sum(loss_pnls) != 0 else 0,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'total_commission': float(sum(t.commission for t in trades)),
            'total_slippage': float(sum(t.slippage for t in trades))
        }
    
    def _max_consecutive(self, condition: np.ndarray) -> int:
        """Calculate maximum consecutive True values."""
        max_count = 0
        current_count = 0
        
        for val in condition:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    # =========================================================================
    # BENCHMARK COMPARISON
    # =========================================================================
    
    def compare_to_benchmark(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict:
        """Compare strategy performance to benchmark."""
        strategy_returns = np.asarray(strategy_returns)
        benchmark_returns = np.asarray(benchmark_returns)
        
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = strategy_returns - benchmark_returns
        
        cov_matrix = np.cov(strategy_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
        
        strategy_annual = np.mean(strategy_returns) * self.trading_days
        benchmark_annual = np.mean(benchmark_returns) * self.trading_days
        alpha = strategy_annual - beta * benchmark_annual
        
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days)
        
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days) if np.std(excess_returns) > 0 else 0
        
        correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
        
        up_capture = np.mean(strategy_returns[benchmark_returns > 0]) / np.mean(benchmark_returns[benchmark_returns > 0]) if np.any(benchmark_returns > 0) else 0
        down_capture = np.mean(strategy_returns[benchmark_returns < 0]) / np.mean(benchmark_returns[benchmark_returns < 0]) if np.any(benchmark_returns < 0) else 0
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'correlation': float(correlation),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'up_capture': float(up_capture),
            'down_capture': float(down_capture),
            'excess_return': float(np.mean(excess_returns) * self.trading_days)
        }
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def generate_report(self, result: BacktestResult) -> Dict:
        """Generate comprehensive backtest report."""
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': float(result.equity_curve[-1]),
                'total_return': result.metrics.total_return,
                'annualized_return': result.metrics.annualized_return,
                'volatility': result.metrics.volatility,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'sortino_ratio': result.metrics.sortino_ratio,
                'calmar_ratio': result.metrics.calmar_ratio,
                'max_drawdown': result.metrics.max_drawdown
            },
            'risk_metrics': {
                'var_95': result.metrics.var_95,
                'cvar_95': result.metrics.cvar_95,
                'max_drawdown': result.drawdown_analysis.max_drawdown,
                'max_drawdown_duration': result.drawdown_analysis.max_drawdown_duration,
                'avg_drawdown': result.drawdown_analysis.avg_drawdown,
                'recovery_factor': result.drawdown_analysis.recovery_factor
            },
            'trade_statistics': {
                'total_trades': result.metrics.total_trades,
                'winning_trades': result.metrics.winning_trades,
                'losing_trades': result.metrics.losing_trades,
                'win_rate': result.metrics.win_rate,
                'profit_factor': result.metrics.profit_factor,
                'avg_win': result.metrics.avg_win,
                'avg_loss': result.metrics.avg_loss,
                'expectancy': result.metrics.expectancy,
                'sqn': result.metrics.sqn
            },
            'distribution': {
                'skewness': result.metrics.skewness,
                'kurtosis': result.metrics.kurtosis,
                'best_trade': result.metrics.best_trade,
                'worst_trade': result.metrics.worst_trade
            }
        }
        
        return report
