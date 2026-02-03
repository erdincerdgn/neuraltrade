"""
Institutional Circuit Breaker with CVaR Thresholds
Author: Erdinc Erdogan
Purpose: Multi-tier circuit breaker with CVaR-based dynamic thresholds, EWMA volatility monitoring, and automatic cooldown with half-open testing.
References:
- CVaR-Based Risk Thresholds
- EWMA Volatility Monitoring
- Circuit Breaker Pattern (Closed/Open/Half-Open)
Usage:
    breaker = CircuitBreaker(initial_capital=100000)
    status = breaker.check(current_capital=95000, daily_pnl=-2000)
    if not status.is_trading_allowed: halt_trading()
"""
import numpy as np
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from colorama import Fore, Style

# Core imports
from ..core.base import (
    StatisticalConstants, RiskTier, MarketRegime,
    calculate_cvar, classify_risk_tier
)


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states (following standard pattern)."""
    CLOSED = auto()      # Normal operation - trading allowed
    OPEN = auto()        # Triggered - trading blocked
    HALF_OPEN = auto()   # Testing recovery - limited trading


class TriggerType(Enum):
    """Types of circuit breaker triggers."""
    DRAWDOWN = auto()           # Maximum drawdown exceeded
    DAILY_LOSS = auto()         # Daily loss limit exceeded
    HOURLY_LOSS = auto()        # Hourly loss limit exceeded
    CVAR_BREACH = auto()        # CVaR threshold exceeded
    VOLATILITY_SPIKE = auto()   # Volatility spike detected
    CONSECUTIVE_LOSSES = auto() # Too many consecutive losses
    MANUAL = auto()             # Manual trigger


@dataclass
class TriggerEvent:
    """Record of a circuit breaker trigger event."""
    trigger_type: TriggerType
    trigger_value: float
    threshold: float
    timestamp: datetime
    message: str
    risk_tier: RiskTier
    
    def to_dict(self) -> Dict:
        return {
            "trigger_type": self.trigger_type.name,
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "risk_tier": self.risk_tier.name
        }


@dataclass
class CircuitBreakerStatus:
    """Current status of the circuit breaker."""
    state: CircuitState
    is_trading_allowed: bool
    current_drawdown_pct: float
    peak_capital: float
    current_capital: float
    daily_pnl: float
    cvar_current: float
    volatility_current: float
    risk_tier: RiskTier
    last_trigger: Optional[TriggerEvent]
    cooldown_remaining_seconds: int
    
    def to_dict(self) -> Dict:
        return {
            "state": self.state.name,
            "is_trading_allowed": self.is_trading_allowed,
            "current_drawdown_pct": self.current_drawdown_pct * 100,
            "peak_capital": self.peak_capital,
            "current_capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "cvar_current_pct": self.cvar_current * 100,
            "volatility_current_pct": self.volatility_current * 100,
            "risk_tier": self.risk_tier.name,
            "last_trigger": self.last_trigger.to_dict() if self.last_trigger else None,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds
        }


# ============================================================================
# CVAR-BASED CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Institutional-Grade Circuit Breaker with CVaR-Based Triggers.
    
    Key Improvements over v1:
    1. Proper peak-to-trough Maximum Drawdown (not from initial capital)
    2. CVaR-based dynamic thresholds
    3. EWMA volatility monitoring
    4. Regime-aware risk limits
    5. Multi-tier circuit states (CLOSED, OPEN, HALF_OPEN)
    
    Mathematical Basis:
    - MDD = (Peak - Current) / Peak  [NOT (Initial - Current) / Initial]
    - CVaR_α = E[X | X ≤ VaR_α]
    - EWMA: σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
    
    Trigger Conditions:
    1. MDD exceeds threshold → OPEN
    2. CVaR(95%) exceeds daily loss limit → OPEN
    3. Volatility spike (EWMA) → OPEN
    4. Consecutive losses → OPEN
    5. Hourly loss limit → OPEN
    """
    
    # Default thresholds (can be overridden)
    DEFAULT_THRESHOLDS = {
        # Drawdown thresholds (percentage as decimal)
        "max_drawdown_pct": 0.05,           # 5% max drawdown
        "warning_drawdown_pct": 0.03,       # 3% warning level
        
        # Daily loss thresholds
        "max_daily_loss_pct": 0.02,         # 2% daily loss limit
        "max_daily_loss_absolute": 5000.0,  # $5000 absolute limit
        
        # Hourly loss
        "max_hourly_loss_pct": 0.01,        # 1% hourly loss
        
        # CVaR thresholds
        "cvar_95_limit": 0.03,              # 3% CVaR(95%) limit
        "cvar_99_limit": 0.05,              # 5% CVaR(99%) limit
        
        # Volatility thresholds
        "volatility_spike_mult": 2.0,       # 2x baseline volatility
        "max_volatility_annual": 0.50,      # 50% annualized volatility
        
        # Consecutive losses
        "max_consecutive_losses": 5,        # 5 consecutive losses
        
        # Cooldown
        "cooldown_minutes": 30,             # 30 minute cooldown
        "half_open_test_trades": 3,         # 3 test trades in half-open
    }
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 thresholds: Dict = None,
                 on_trigger_callback: Callable = None,
                 on_recovery_callback: Callable = None):
        """
        Initialize Circuit Breaker.
        
        Args:
            initial_capital: Starting capital
            thresholds: Custom thresholds (overrides defaults)
            on_trigger_callback: Called when circuit opens
            on_recovery_callback: Called when circuit closes
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital  # High Water Mark
        
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.on_trigger_callback = on_trigger_callback
        self.on_recovery_callback = on_recovery_callback
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.last_trigger: Optional[TriggerEvent] = None
        self.triggered_at: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        
        # Trading history
        self.trade_history: List[Dict] = []
        self.returns_history: List[float] = []
        self.daily_pnl: float = 0.0
        self.hourly_pnl_history: List[Dict] = []
        self.consecutive_losses: int = 0
        
        # Volatility tracking
        self.baseline_volatility: Optional[float] = None
        self.current_volatility: float = 0.0
        self.ewma_variance: float = 0.0
        
        # Half-open state tracking
        self.half_open_trades: int = 0
        self.half_open_wins: int = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Trigger history
        self.trigger_history: List[TriggerEvent] = []
    
    # ========================================================================
    # CAPITAL AND TRADE UPDATES
    # ========================================================================
    
    def update_capital(self, new_capital: float, trade_return: float = None):
        """
        Update capital and run all circuit breaker checks.
        
        Args:
            new_capital: New portfolio value
            trade_return: Return of the last trade (optional)
        """
        with self._lock:
            pnl = new_capital - self.current_capital
            self.current_capital = new_capital
            
            # Update High Water Mark (peak) - CRITICAL FIX
            if new_capital > self.peak_capital:
                self.peak_capital = new_capital
            
            # Record trade
            self.trade_history.append({
                "timestamp": datetime.now(),
                "capital": new_capital,
                "pnl": pnl,
                "return": trade_return or (pnl / (new_capital - pnl) if new_capital != pnl else 0)
            })
            
            # Update returns history
            if trade_return is not None:
                self.returns_history.append(trade_return)
            elif len(self.trade_history) >= 2:
                prev_capital = self.trade_history[-2]["capital"]
                if prev_capital > 0:
                    self.returns_history.append(pnl / prev_capital)
            
            # Update daily PnL
            self.daily_pnl += pnl
            
            # Update hourly PnL
            self.hourly_pnl_history.append({
                "timestamp": datetime.now(),
                "pnl": pnl
            })
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Update EWMA volatility
            self._update_ewma_volatility(trade_return or pnl / self.current_capital if self.current_capital > 0 else 0)
            
            # Run all checks
            self._run_all_checks()
            
            # Handle half-open state
            if self.state == CircuitState.HALF_OPEN:
                self._handle_half_open_trade(pnl > 0)
    
    def _update_ewma_volatility(self, return_value: float):
        """
        Update EWMA volatility estimate.
        
        σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
        
        Uses RiskMetrics standard λ = 0.94
        """
        lambda_decay = StatisticalConstants.EWMA_LAMBDA_DAILY
        
        if self.ewma_variance == 0:
            self.ewma_variance = return_value ** 2
        else:
            self.ewma_variance = (lambda_decay * self.ewma_variance + 
                                  (1 - lambda_decay) * return_value ** 2)
        
        # Annualized volatility
        self.current_volatility = np.sqrt(self.ewma_variance) * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        
        # Set baseline if not set
        if self.baseline_volatility is None and len(self.returns_history) >= 20:
            self.baseline_volatility = self.current_volatility
    
    # ========================================================================
    # CIRCUIT BREAKER CHECKS
    # ========================================================================
    
    def _run_all_checks(self):
        """Run all circuit breaker condition checks."""
        if self.state == CircuitState.OPEN:
            self._check_cooldown_expired()
            return
        
        # Check each condition
        self._check_drawdown()
        self._check_daily_loss()
        self._check_hourly_loss()
        self._check_cvar()
        self._check_volatility_spike()
        self._check_consecutive_losses()
    
    def _check_drawdown(self):
        """
        Check Maximum Drawdown (peak-to-trough).
        
        CRITICAL: MDD = (Peak - Current) / Peak
        NOT: (Initial - Current) / Initial
        
        This is the proper institutional calculation.
        """
        if self.peak_capital <= 0:
            return
        
        # Proper MDD calculation from High Water Mark
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if current_drawdown >= self.thresholds["max_drawdown_pct"]:
            self._trigger(
                TriggerType.DRAWDOWN,
                current_drawdown,
                self.thresholds["max_drawdown_pct"],
                f"🚨 MAX DRAWDOWN: {current_drawdown*100:.2f}% from peak ${self.peak_capital:,.0f} (Limit: {self.thresholds['max_drawdown_pct']*100:.1f}%)"
            )
    
    def _check_daily_loss(self):
        """Check daily loss limits (percentage and absolute)."""
        # Percentage check (relative to peak, not initial)
        daily_loss_pct = abs(self.daily_pnl) / self.peak_capital if self.peak_capital > 0 else 0
        
        if self.daily_pnl < 0 and daily_loss_pct >= self.thresholds["max_daily_loss_pct"]:
            self._trigger(
                TriggerType.DAILY_LOSS,
                daily_loss_pct,
                self.thresholds["max_daily_loss_pct"],
                f"🚨 DAILY LOSS: {daily_loss_pct*100:.2f}% (Limit: {self.thresholds['max_daily_loss_pct']*100:.1f}%)"
            )
            return
        
        # Absolute check
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.thresholds["max_daily_loss_absolute"]:
            self._trigger(
                TriggerType.DAILY_LOSS,
                abs(self.daily_pnl),
                self.thresholds["max_daily_loss_absolute"],
                f"🚨 DAILY LOSS: ${abs(self.daily_pnl):,.0f} (Limit: ${self.thresholds['max_daily_loss_absolute']:,.0f})"
            )
    
    def _check_hourly_loss(self):
        """Check hourly loss limit."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        hourly_pnl = sum(
            t["pnl"] for t in self.hourly_pnl_history 
            if t["timestamp"] > one_hour_ago
        )
        
        if hourly_pnl < 0:
            hourly_loss_pct = abs(hourly_pnl) / self.peak_capital if self.peak_capital > 0 else 0
            
            if hourly_loss_pct >= self.thresholds["max_hourly_loss_pct"]:
                self._trigger(
                    TriggerType.HOURLY_LOSS,
                    hourly_loss_pct,
                    self.thresholds["max_hourly_loss_pct"],
                    f"🚨 HOURLY LOSS: {hourly_loss_pct*100:.2f}% (Limit: {self.thresholds['max_hourly_loss_pct']*100:.1f}%)"
                )
    
    def _check_cvar(self):
        """
        Check CVaR (Expected Shortfall) threshold.
        
        CVaR_α = E[X | X ≤ VaR_α]
        
        Triggers if CVaR(95%) exceeds daily loss limit.
        This is the Basel III preferred risk measure.
        """
        if len(self.returns_history) < StatisticalConstants.MIN_SAMPLES_SIGNIFICANCE:
            return
        
        returns = np.array(self.returns_history[-252:])  # Last year of data
        
        # Calculate CVaR(95%) using base.py function
        cvar_95 = calculate_cvar(returns, confidence=0.95)
        
        if cvar_95 >= self.thresholds["cvar_95_limit"]:
            self._trigger(
                TriggerType.CVAR_BREACH,
                cvar_95,
                self.thresholds["cvar_95_limit"],
                f"🚨 CVaR(95%) BREACH: {cvar_95*100:.2f}% (Limit: {self.thresholds['cvar_95_limit']*100:.1f}%)"
            )
    
    def _check_volatility_spike(self):
        """
        Check for volatility spike using EWMA.
        
        Triggers if current volatility > baseline × multiplier.
        """
        if self.baseline_volatility is None or self.baseline_volatility == 0:
            return
        
        vol_ratio = self.current_volatility / self.baseline_volatility
        
        if vol_ratio >= self.thresholds["volatility_spike_mult"]:
            self._trigger(
                TriggerType.VOLATILITY_SPIKE,
                vol_ratio,
                self.thresholds["volatility_spike_mult"],
                f"🚨 VOLATILITY SPIKE: {vol_ratio:.1f}x baseline (Limit: {self.thresholds['volatility_spike_mult']:.1f}x)"
            )
            return
        
        # Also check absolute volatility
        if self.current_volatility >= self.thresholds["max_volatility_annual"]:
            self._trigger(
                TriggerType.VOLATILITY_SPIKE,
                self.current_volatility,
                self.thresholds["max_volatility_annual"],
                f"🚨 HIGH VOLATILITY: {self.current_volatility*100:.1f}% annual (Limit: {self.thresholds['max_volatility_annual']*100:.1f}%)"
            )
    
    def _check_consecutive_losses(self):
        """Check consecutive loss limit."""
        if self.consecutive_losses >= self.thresholds["max_consecutive_losses"]:
            self._trigger(
                TriggerType.CONSECUTIVE_LOSSES,
                self.consecutive_losses,
                self.thresholds["max_consecutive_losses"],
                f"🚨 CONSECUTIVE LOSSES: {self.consecutive_losses} trades (Limit: {self.thresholds['max_consecutive_losses']})"
            )
    
    # ========================================================================
    # TRIGGER AND RECOVERY
    # ========================================================================
    
    def _trigger(self, 
                 trigger_type: TriggerType,
                 trigger_value: float,
                 threshold: float,
                 message: str):
        """Trigger the circuit breaker."""
        if self.state == CircuitState.OPEN:
            return  # Already triggered
        
        # Determine risk tier using CVaR
        risk_tier = self._calculate_risk_tier()
        
        # Create trigger event
        trigger_event = TriggerEvent(
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            threshold=threshold,
            timestamp=datetime.now(),
            message=message,
            risk_tier=risk_tier
        )
        
        self.last_trigger = trigger_event
        self.trigger_history.append(trigger_event)
        
        # Update state
        self.state = CircuitState.OPEN
        self.triggered_at = datetime.now()
        self.cooldown_until = self.triggered_at + timedelta(
            minutes=self.thresholds["cooldown_minutes"]
        )
        
        # Display alert
        self._display_trigger_alert(trigger_event)
        
        # Callback
        if self.on_trigger_callback:
            self.on_trigger_callback(trigger_event)
    
    def _calculate_risk_tier(self) -> RiskTier:
        """Calculate current risk tier based on CVaR."""
        if len(self.returns_history) < 10:
            return RiskTier.MODERATE
        
        returns = np.array(self.returns_history[-252:])
        cvar = calculate_cvar(returns, confidence=0.95)
        return classify_risk_tier(cvar)
    
    def _display_trigger_alert(self, event: TriggerEvent):
        """Display circuit breaker trigger alert."""
        print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.RED}⚠️ CIRCUIT BREAKER TRIGGERED!{Style.RESET_ALL}", flush=True)
        print(f"{Fore.RED}{event.message}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.RED}Risk Tier: {event.risk_tier.name}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.RED}All trading halted.{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}Cooldown until: {self.cooldown_until.strftime('%H:%M:%S')}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
    
    def _check_cooldown_expired(self):
        """Check if cooldown period has expired."""
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            # Move to half-open state for testing
            self.state = CircuitState.HALF_OPEN
            self.half_open_trades = 0
            self.half_open_wins = 0
            print(f"{Fore.YELLOW}⚡ Circuit Breaker entering HALF-OPEN state for testing{Style.RESET_ALL}", flush=True)
    
    def _handle_half_open_trade(self, is_win: bool):
        """Handle trade result in half-open state."""
        self.half_open_trades += 1
        if is_win:
            self.half_open_wins += 1
        
        # Check if enough test trades
        if self.half_open_trades >= self.thresholds["half_open_test_trades"]:
            win_rate = self.half_open_wins / self.half_open_trades
            
            if win_rate >= 0.5:  # At least 50% win rate
                self._recover()
            else:
                # Back to open state
                self.state = CircuitState.OPEN
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.thresholds["cooldown_minutes"]
                )
                print(f"{Fore.RED}⚠️ Half-open test failed. Returning to OPEN state.{Style.RESET_ALL}", flush=True)
    
    def _recover(self):
        """Recover from triggered state."""
        self.state = CircuitState.CLOSED
        self.triggered_at = None
        self.cooldown_until = None
        self.consecutive_losses = 0
        
        print(f"{Fore.GREEN}✅ Circuit Breaker CLOSED. Trading resumed.{Style.RESET_ALL}", flush=True)
        
        if self.on_recovery_callback:
            self.on_recovery_callback()
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            Tuple of (is_allowed, reason_message)
        """
        if self.state == CircuitState.CLOSED:
            return True, "✅ Trading allowed"
        
        if self.state == CircuitState.HALF_OPEN:
            remaining_tests = self.thresholds["half_open_test_trades"] - self.half_open_trades
            return True, f"⚡ HALF-OPEN: {remaining_tests} test trades remaining"
        
        # OPEN state
        if self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds()
            if remaining > 0:
                return False, f"🚫 BLOCKED: {int(remaining // 60)}m {int(remaining % 60)}s cooldown remaining"
        
        return False, "🚫 Circuit breaker OPEN"
    
    def manual_trigger(self, reason: str = "Manual trigger"):
        """Manually trigger the circuit breaker."""
        self._trigger(
            TriggerType.MANUAL,
            0.0,
            0.0,
            f"🚨 MANUAL TRIGGER: {reason}"
        )
    
    def manual_reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.triggered_at = None
            self.cooldown_until = None
            self.consecutive_losses = 0
            self.daily_pnl = 0.0
            print(f"{Fore.GREEN}✅ Circuit Breaker manually reset{Style.RESET_ALL}", flush=True)
    
    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        with self._lock:
            self.daily_pnl = 0.0
            self.hourly_pnl_history = []
            self.consecutive_losses = 0
            
            # Update baseline volatility
            if len(self.returns_history) >= 20:
                self.baseline_volatility = self.current_volatility
    
    def get_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        current_drawdown = 0.0
        if self.peak_capital > 0:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        cooldown_remaining = 0
        if self.cooldown_until and self.state == CircuitState.OPEN:
            remaining = (self.cooldown_until - datetime.now()).total_seconds()
            cooldown_remaining = max(0, int(remaining))
        
        # Calculate current CVaR
        cvar_current = 0.0
        if len(self.returns_history) >= 10:
            returns = np.array(self.returns_history[-252:])
            cvar_current = calculate_cvar(returns, confidence=0.95)
        
        return CircuitBreakerStatus(
            state=self.state,
            is_trading_allowed=self.state != CircuitState.OPEN,
            current_drawdown_pct=current_drawdown,
            peak_capital=self.peak_capital,
            current_capital=self.current_capital,
            daily_pnl=self.daily_pnl,
            cvar_current=cvar_current,
            volatility_current=self.current_volatility,
            risk_tier=self._calculate_risk_tier(),
            last_trigger=self.last_trigger,
            cooldown_remaining_seconds=cooldown_remaining
        )
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report."""
        status = self.get_status()
        
        state_emoji = {
            CircuitState.CLOSED: "🟢",
            CircuitState.OPEN: "🔴",
            CircuitState.HALF_OPEN: "🟡"
        }
        
        report = f"""
<circuit_breaker_v2>
{state_emoji[status.state]} CIRCUIT BREAKER STATUS - {status.state.name}
══════════════════════════════════════════════════════════════

💰 CAPITAL:
  • Current: ${status.current_capital:,.2f}
  • Peak (HWM): ${status.peak_capital:,.2f}
  • Initial: ${self.initial_capital:,.2f}

📉 RISK METRICS:
  • Current Drawdown: {status.current_drawdown_pct*100:.2f}%
  • Daily PnL: ${status.daily_pnl:+,.2f}
  • CVaR(95%): {status.cvar_current*100:.2f}%
  • Volatility (Annual): {status.volatility_current*100:.1f}%
  • Risk Tier: {status.risk_tier.name}

⚙️ THRESHOLDS:
  • Max Drawdown: {self.thresholds['max_drawdown_pct']*100:.1f}%
  • Daily Loss Limit: {self.thresholds['max_daily_loss_pct']*100:.1f}%
  • CVaR(95%) Limit: {self.thresholds['cvar_95_limit']*100:.1f}%
  • Volatility Spike: {self.thresholds['volatility_spike_mult']:.1f}x
  • Consecutive Losses: {self.thresholds['max_consecutive_losses']}

📊 TRADING STATS:
  • Consecutive Losses: {self.consecutive_losses}
  • Total Trades: {len(self.trade_history)}
  • Triggers Today: {len([t for t in self.trigger_history if t.timestamp.date() == datetime.now().date()])}
"""
        
        if status.state == CircuitState.OPEN:
            report += f"""
🚨 TRIGGERED:
  • Reason: {status.last_trigger.message if status.last_trigger else 'N/A'}
  • Cooldown: {status.cooldown_remaining_seconds // 60}m {status.cooldown_remaining_seconds % 60}s remaining
"""
        elif status.state == CircuitState.HALF_OPEN:
            report += f"""
⚡ HALF-OPEN TESTING:
  • Test Trades: {self.half_open_trades}/{self.thresholds['half_open_test_trades']}
  • Wins: {self.half_open_wins}
"""
        
        report += "\n</circuit_breaker_v2>\n"
        return report


# ============================================================================
# REGIME-AWARE CIRCUIT BREAKER
# ============================================================================

class RegimeAwareCircuitBreaker(CircuitBreaker):
    """
    Circuit Breaker with regime-aware dynamic thresholds.
    
    Adjusts thresholds based on detected market regime:
    - High volatility regime → tighter limits
    - Low volatility regime → relaxed limits
    - Crisis regime → emergency limits
    """
    
    # Regime multipliers for thresholds
    REGIME_MULTIPLIERS = {
        MarketRegime.LOW_VOLATILITY: 1.5,      # Relax limits
        MarketRegime.NORMAL_VOLATILITY: 1.0,   # Normal limits
        MarketRegime.HIGH_VOLATILITY: 0.7,     # Tighten limits
        MarketRegime.EXTREME_VOLATILITY: 0.5,  # Very tight
        MarketRegime.CRISIS: 0.3,              # Emergency limits
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_regime = MarketRegime.NORMAL_VOLATILITY
        self.base_thresholds = self.thresholds.copy()
    
    def update_regime(self, regime: MarketRegime):
        """Update market regime and adjust thresholds."""
        self.current_regime = regime
        multiplier = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        
        # Adjust key thresholds
        self.thresholds["max_drawdown_pct"] = self.base_thresholds["max_drawdown_pct"] * multiplier
        self.thresholds["max_daily_loss_pct"] = self.base_thresholds["max_daily_loss_pct"] * multiplier
        self.thresholds["cvar_95_limit"] = self.base_thresholds["cvar_95_limit"] * multiplier
        
        print(f"{Fore.CYAN}📊 Regime updated to {regime.name}. Thresholds adjusted (×{multiplier:.1f}){Style.RESET_ALL}", flush=True)
    
    def detect_regime_from_volatility(self) -> MarketRegime:
        """Auto-detect regime from current volatility."""
        vol = self.current_volatility
        
        if vol >= 0.40:
            return MarketRegime.EXTREME_VOLATILITY
        elif vol >= 0.25:
            return MarketRegime.HIGH_VOLATILITY
        elif vol >= 0.15:
            return MarketRegime.NORMAL_VOLATILITY
        else:
            return MarketRegime.LOW_VOLATILITY
    
    def auto_adjust_regime(self):
        """Automatically detect and adjust regime."""
        detected_regime = self.detect_regime_from_volatility()
        if detected_regime != self.current_regime:
            self.update_regime(detected_regime)
