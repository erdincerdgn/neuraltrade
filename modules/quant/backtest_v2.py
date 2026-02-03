"""
Event-Driven Backtest Engine v2
Author: Erdinc Erdogan
Purpose: Realistic market simulation with latency modeling, bid-ask spread, slippage, commission, and priority queue event management.
References:
- Event-Driven Architecture with Priority Queue
- Order Book Simulation
- Latency and Slippage Modeling
Usage:
    engine = EventDrivenBacktest(initial_capital=100000, latency_ms=50)
    engine.run(strategy, price_data)
    report = engine.generate_report()
"""
import os
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from colorama import Fore, Style
import numpy as np


class EventType(Enum):
    """Olay tipleri."""
    MARKET_DATA = 1
    ORDER_SUBMITTED = 2
    ORDER_FILLED = 3
    ORDER_CANCELLED = 4
    SIGNAL = 5
    REBALANCE = 6


@dataclass(order=True)
class Event:
    """
    Priority queue için event wrapper.
    """
    timestamp: datetime
    event_type: EventType = field(compare=False)
    data: Dict = field(compare=False, default_factory=dict)


class OrderBook:
    """Basit order book simülasyonu."""
    
    def __init__(self, spread_pct: float = 0.001):
        """
        Args:
            spread_pct: Bid-Ask spread yüzdesi
        """
        self.spread_pct = spread_pct
        self.last_price = 0.0
    
    def get_bid_ask(self, mid_price: float) -> tuple:
        """Bid-Ask fiyatlarını hesapla."""
        half_spread = mid_price * self.spread_pct / 2
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        self.last_price = mid_price
        return bid, ask
    
    def get_execution_price(self, side: str, mid_price: float, size: float = 1.0) -> float:
        """
        Gerçekçi execution fiyatı hesapla (slippage dahil).
        
        Args:
            side: "BUY" veya "SELL"
            mid_price: Orta fiyat
            size: İşlem büyüklüğü
        """
        bid, ask = self.get_bid_ask(mid_price)
        
        # Size'a göre ek slippage
        size_slippage = 0.0001 * np.sqrt(size)  # Büyük emirlerde daha fazla slippage
        
        if side == "BUY":
            return ask * (1 + size_slippage)
        else:
            return bid * (1 - size_slippage)


class EventDrivenBacktest:
    """
    Event-Driven Backtest Engine v2.
    
    Özellikler:
    - Gerçekçi latency simülasyonu
    - Slippage ve spread hesaplama
    - Komisyon ve fee hesaplama
    - Priority queue ile event yönetimi
    - Detaylı trade loglama
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 latency_ms: int = 50):
        """
        Args:
            initial_capital: Başlangıç sermayesi
            commission_rate: Komisyon oranı (0.001 = %0.1)
            slippage_rate: Slippage oranı
            latency_ms: Emir gecikmesi (milisaniye)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.latency_ms = latency_ms
        
        self.reset()
    
    def reset(self):
        """Engine'i sıfırla."""
        self.capital = self.initial_capital
        self.positions = {}  # {ticker: quantity}
        self.pending_orders = []  # Bekleyen emirler
        
        self.event_queue = []  # Priority queue
        self.trade_log = []
        self.equity_curve = []
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.total_slippage = 0
        
        self.order_book = OrderBook()
        self.current_prices = {}
    
    def add_market_data(self, data: List[Dict]):
        """
        Piyasa verilerini event queue'ya ekle.
        
        Args:
            data: [{timestamp, ticker, open, high, low, close, volume}, ...]
        """
        for bar in data:
            event = Event(
                timestamp=bar["timestamp"],
                event_type=EventType.MARKET_DATA,
                data=bar
            )
            heapq.heappush(self.event_queue, event)
    
    def submit_order(self, 
                     timestamp: datetime,
                     ticker: str,
                     side: str,
                     quantity: float,
                     order_type: str = "MARKET") -> str:
        """
        Emir gönder.
        
        Args:
            timestamp: Emir zamanı
            ticker: Sembol
            side: "BUY" veya "SELL"
            quantity: Miktar
            order_type: "MARKET" veya "LIMIT"
        """
        order_id = f"ORD-{len(self.trade_log) + 1:05d}"
        
        # Latency ekle
        fill_time = timestamp + timedelta(milliseconds=self.latency_ms)
        
        order = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "submitted_at": timestamp,
            "status": "PENDING"
        }
        
        # Order submission event
        submit_event = Event(
            timestamp=timestamp,
            event_type=EventType.ORDER_SUBMITTED,
            data=order
        )
        heapq.heappush(self.event_queue, submit_event)
        
        # Order fill event (latency sonrası)
        fill_event = Event(
            timestamp=fill_time,
            event_type=EventType.ORDER_FILLED,
            data=order
        )
        heapq.heappush(self.event_queue, fill_event)
        
        self.pending_orders.append(order)
        return order_id
    
    def _process_order_fill(self, order: Dict, current_price: float):
        """Emir dolumunu işle."""
        ticker = order["ticker"]
        side = order["side"]
        quantity = order["quantity"]
        
        # Gerçekçi execution fiyatı
        execution_price = self.order_book.get_execution_price(
            side, current_price, quantity
        )
        
        # Slippage hesapla
        slippage = abs(execution_price - current_price) * quantity
        self.total_slippage += slippage
        
        # Komisyon hesapla
        commission = execution_price * quantity * self.commission_rate
        self.total_commission += commission
        
        # Pozisyon güncelle
        if side == "BUY":
            cost = (execution_price * quantity) + commission
            if self.capital >= cost:
                self.capital -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + quantity
                order["status"] = "FILLED"
            else:
                order["status"] = "REJECTED"
                order["reject_reason"] = "Insufficient capital"
        else:  # SELL
            current_position = self.positions.get(ticker, 0)
            if current_position >= quantity:
                revenue = (execution_price * quantity) - commission
                self.capital += revenue
                self.positions[ticker] -= quantity
                order["status"] = "FILLED"
            else:
                order["status"] = "REJECTED"
                order["reject_reason"] = "Insufficient shares"
        
        if order["status"] == "FILLED":
            order["execution_price"] = execution_price
            order["commission"] = commission
            order["slippage"] = slippage
            order["filled_at"] = order.get("filled_at", datetime.now())
            self.trade_log.append(order)
            self.total_trades += 1
    
    def run(self, strategy_callback: Callable = None) -> Dict:
        """
        Backtest'i çalıştır.
        
        Args:
            strategy_callback: fn(timestamp, prices, positions, capital) -> List[orders]
        """
        print(f"{Fore.CYAN}  → Event-Driven Backtest başlıyor...{Style.RESET_ALL}", flush=True)
        
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            if event.event_type == EventType.MARKET_DATA:
                bar = event.data
                ticker = bar.get("ticker", "UNKNOWN")
                self.current_prices[ticker] = bar["close"]
                
                # Equity curve güncelle
                portfolio_value = self._calculate_portfolio_value()
                self.equity_curve.append({
                    "timestamp": event.timestamp,
                    "portfolio_value": portfolio_value,
                    "capital": self.capital
                })
                
                # Strategy callback
                if strategy_callback:
                    orders = strategy_callback(
                        event.timestamp,
                        self.current_prices.copy(),
                        self.positions.copy(),
                        self.capital
                    )
                    for order in (orders or []):
                        self.submit_order(
                            event.timestamp,
                            order["ticker"],
                            order["side"],
                            order["quantity"]
                        )
            
            elif event.event_type == EventType.ORDER_FILLED:
                order = event.data
                ticker = order["ticker"]
                if ticker in self.current_prices:
                    self._process_order_fill(order, self.current_prices[ticker])
        
        return self._generate_results()
    
    def _calculate_portfolio_value(self) -> float:
        """Toplam portföy değerini hesapla."""
        positions_value = sum(
            qty * self.current_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
        )
        return self.capital + positions_value
    
    def _generate_results(self) -> Dict:
        """Backtest sonuçlarını hesapla."""
        if not self.equity_curve:
            return {"error": "No data"}
        
        final_value = self.equity_curve[-1]["portfolio_value"]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Drawdown hesapla
        peak = self.initial_capital
        max_drawdown = 0
        for point in self.equity_curve:
            if point["portfolio_value"] > peak:
                peak = point["portfolio_value"]
            drawdown = (peak - point["portfolio_value"]) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (basit)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev = self.equity_curve[i-1]["portfolio_value"]
                curr = self.equity_curve[i]["portfolio_value"]
                returns.append((curr - prev) / prev)
            
            avg_return = np.mean(returns) * 252  # Yıllık
            std_return = np.std(returns) * np.sqrt(252)
            sharpe = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"{Fore.GREEN}  → Backtest Tamamlandı: {total_return:+.2f}% Getiri{Style.RESET_ALL}", flush=True)
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "latency_ms": self.latency_ms,
            "equity_curve": self.equity_curve[-10:],  # Son 10 veri
            "trade_log": self.trade_log[-10:]  # Son 10 trade
        }
    
    def generate_backtest_report(self, results: Dict) -> str:
        """Backtest raporu oluştur."""
        report = f"""
<backtest_v2>
🔬 EVENT-DRIVEN BACKTEST SONUÇLARI
════════════════════════════════════════

💰 PERFORMANS:
  • Başlangıç: ${results['initial_capital']:,.2f}
  • Bitiş: ${results['final_value']:,.2f}
  • Getiri: {results['total_return_pct']:+.2f}%
  • Max Drawdown: {results['max_drawdown_pct']:.2f}%
  • Sharpe Ratio: {results['sharpe_ratio']:.2f}

📊 TRADİNG İSTATİSTİKLERİ:
  • Toplam Trade: {results['total_trades']}
  • Win Rate: {results['win_rate']:.1f}%
  • Komisyon: ${results['total_commission']:.2f}
  • Slippage: ${results['total_slippage']:.2f}
  • Latency: {results['latency_ms']}ms

📈 SON İŞLEMLER:
"""
        for trade in results.get('trade_log', [])[-5:]:
            emoji = "🟢" if trade["side"] == "BUY" else "🔴"
            report += f"  {emoji} {trade['side']} {trade['quantity']} {trade['ticker']} @ ${trade.get('execution_price', 0):.2f}\n"
        
        report += "</backtest_v2>\n"
        return report
