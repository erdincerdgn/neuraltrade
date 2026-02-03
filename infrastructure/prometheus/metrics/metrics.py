from prometheus_client import Gauge

# ============================================
# METRIC DEFINITIONS (ENGLISH 🇺🇸)
# ============================================

# 1. Main Balance
portfolio_gauge = Gauge('neuraltrade_portfolio_value_usd', 'Real-time Portfolio Value (USD)')

# 2. Active Positions
active_trades_gauge = Gauge('neuraltrade_active_positions', 'Number of Active Positions')

# 3. Daily PnL
daily_pnl_gauge = Gauge('neuraltrade_daily_pnl', 'Daily Profit and Loss (USD)')

# 4. Win Rate (%)
win_rate_gauge = Gauge('neuraltrade_win_rate', 'Strategy Win Rate Percentage')

# 5. AI Confidence Score (0-100)
ai_confidence_gauge = Gauge('neuraltrade_ai_confidence', 'AI Model Confidence Score')

# 6. System Latency (ms)
latency_gauge = Gauge('neuraltrade_execution_latency_ms', 'Execution Latency (ms)')