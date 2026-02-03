"""
Event-Driven Market Monitor
Author: Erdinc Erdogan
Purpose: Real-time market surveillance service that triggers alerts on RSI extremes, sudden price movements, and volatility spikes.
References:
- Event-Driven Architecture
- Technical Indicator Alerts
- Real-Time Market Surveillance
Usage:
    monitor = MarketMonitor(advisor_callback=my_callback)
    alerts = monitor.check_conditions("AAPL", price=150.0, rsi=28.5)
    monitor.start_monitoring(["AAPL", "MSFT"])
"""
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from colorama import Fore, Style

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class MarketMonitor:
    """
    Olay güdümlü piyasa izleme servisi.
    RSI < 30, ani fiyat hareketleri, volatilite artışında otomatik tetiklenir.
    """
    
    ALERT_THRESHOLDS = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "price_change_pct": 3.0,
        "volatility_spike": 2.0,
    }
    
    def __init__(self, advisor_callback: Callable = None):
        self.advisor_callback = advisor_callback
        self.last_prices = {}
        self.last_rsi = {}
        self.alerts = []
        self.is_running = False
        self.thread = None
    
    def check_conditions(self, ticker: str, price: float, rsi: float, volatility: float = None) -> List[Dict]:
        """Piyasa koşullarını kontrol et ve alert oluştur."""
        alerts = []
        
        # RSI Oversold
        if rsi < self.ALERT_THRESHOLDS["rsi_oversold"]:
            alerts.append({
                "type": "RSI_OVERSOLD",
                "ticker": ticker,
                "value": rsi,
                "message": f"🔴 ACİL: {ticker} RSI {rsi:.1f} - Aşırı Satım!",
                "urgency": "HIGH"
            })
        
        # RSI Overbought
        elif rsi > self.ALERT_THRESHOLDS["rsi_overbought"]:
            alerts.append({
                "type": "RSI_OVERBOUGHT",
                "ticker": ticker,
                "value": rsi,
                "message": f"🟠 UYARI: {ticker} RSI {rsi:.1f} - Aşırı Alım!",
                "urgency": "MEDIUM"
            })
        
        # Price Change
        if ticker in self.last_prices:
            price_change = abs((price - self.last_prices[ticker]) / self.last_prices[ticker] * 100)
            if price_change > self.ALERT_THRESHOLDS["price_change_pct"]:
                direction = "📈" if price > self.last_prices[ticker] else "📉"
                alerts.append({
                    "type": "PRICE_SPIKE",
                    "ticker": ticker,
                    "value": price_change,
                    "message": f"{direction} ACİL: {ticker} %{price_change:.1f} hareket!",
                    "urgency": "HIGH"
                })
        
        # Volatility Spike
        if volatility and volatility > self.ALERT_THRESHOLDS["volatility_spike"]:
            alerts.append({
                "type": "VOLATILITY_SPIKE",
                "ticker": ticker,
                "value": volatility,
                "message": f"⚠️ {ticker} volatilite {volatility:.1f}x normal!",
                "urgency": "MEDIUM"
            })
        
        self.last_prices[ticker] = price
        self.last_rsi[ticker] = rsi
        
        return alerts
    
    def trigger_analysis(self, alert: Dict) -> str:
        """Alert tetiklendiğinde analiz başlat."""
        if self.advisor_callback:
            print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.RED}🚨 MARKET ALERT: {alert['message']}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
            
            return self.advisor_callback(
                ticker=alert["ticker"],
                tech_signals=f"ALERT: {alert['type']} - {alert['value']}",
                market_sentiment="Volatil"
            )
        return ""
    
    def generate_report(self, alerts: List[Dict]) -> str:
        """Alert raporu oluştur."""
        if not alerts:
            return ""
        
        report = "\n<market_alerts>\n🚨 PİYASA ALERTLERİ:\n"
        for alert in alerts:
            urgency_emoji = "🔴" if alert["urgency"] == "HIGH" else "🟠"
            report += f"  {urgency_emoji} [{alert['type']}] {alert['message']}\n"
        report += "</market_alerts>\n"
        
        return report
    
    def start_monitoring(self, tickers: List[str], interval_seconds: int = 300):
        """Arka plan izleme başlat."""
        def monitor_loop():
            self.is_running = True
            print(f"{Fore.GREEN}📡 MarketMonitor başlatıldı - {len(tickers)} sembol{Style.RESET_ALL}", flush=True)
            
            while self.is_running:
                for ticker in tickers:
                    try:
                        if YFINANCE_AVAILABLE:
                            info = yf.Ticker(ticker).info
                            price = info.get("regularMarketPrice", 0)
                            rsi = 50  # Placeholder
                            
                            alerts = self.check_conditions(ticker, price, rsi)
                            for alert in alerts:
                                if alert["urgency"] == "HIGH":
                                    self.trigger_analysis(alert)
                    except:
                        pass
                
                time.sleep(interval_seconds)
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
        return self.thread
    
    def stop_monitoring(self):
        """İzlemeyi durdur."""
        self.is_running = False
        print(f"{Fore.YELLOW}📡 MarketMonitor durduruldu{Style.RESET_ALL}", flush=True)
