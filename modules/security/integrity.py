"""
Data Integrity Guard and Anomaly Detection
Author: Erdinc Erdogan
Purpose: Validates price data before processing to detect spikes, stale data, missing values, and API errors preventing garbage-in-garbage-out scenarios.
References:
- Data Quality Validation
- Anomaly Detection Patterns
- Real-Time Data Integrity Checks
Usage:
    guard = DataIntegrityGuard(max_price_change_pct=20.0)
    result = guard.validate_price('AAPL', price=150.0)
    if not result['is_valid']: reject_data(result['rejection_reasons'])
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from colorama import Fore, Style


class DataIntegrityGuard:
    """
    Data Integrity & Anomaly Guard.
    
    Verinin işlenmeden önce denetlenmesi:
    - Anormal fiyat hareketleri
    - Missing data
    - Stale data (eski veri)
    - API hataları
    
    Garbage In = Garbage Out'u engeller.
    """
    
    def __init__(self,
                 max_price_change_pct: float = 20.0,
                 max_stale_seconds: int = 60,
                 min_price: float = 0.01):
        """
        Args:
            max_price_change_pct: Max anlık fiyat değişimi (%)
            max_stale_seconds: Maksimum veri yaşı (saniye)
            min_price: Minimum geçerli fiyat
        """
        self.max_price_change_pct = max_price_change_pct
        self.max_stale_seconds = max_stale_seconds
        self.min_price = min_price
        
        self.price_history = {}  # symbol -> deque of (price, timestamp)
        self.rejected_data = []
        self.alerts = []
    
    def validate_price(self, 
                      symbol: str,
                      price: float,
                      timestamp: datetime = None) -> Dict:
        """
        Fiyat verisini doğrula.
        
        Args:
            symbol: Sembol
            price: Fiyat
            timestamp: Zaman damgası
        """
        timestamp = timestamp or datetime.now()
        
        checks = {
            "null_check": True,
            "positive_check": True,
            "min_price_check": True,
            "spike_check": True,
            "stale_check": True
        }
        
        rejection_reasons = []
        
        # 1. Null kontrolü
        if price is None or np.isnan(price):
            checks["null_check"] = False
            rejection_reasons.append("NULL_PRICE")
        
        # 2. Pozitif kontrolü
        if price <= 0:
            checks["positive_check"] = False
            rejection_reasons.append("NON_POSITIVE_PRICE")
        
        # 3. Minimum fiyat kontrolü
        if price < self.min_price:
            checks["min_price_check"] = False
            rejection_reasons.append(f"BELOW_MIN_PRICE ({self.min_price})")
        
        # 4. Spike kontrolü (önceki fiyata göre)
        if symbol in self.price_history and self.price_history[symbol]:
            last_price, last_time = self.price_history[symbol][-1]
            
            if last_price > 0:
                change_pct = abs((price - last_price) / last_price) * 100
                
                if change_pct > self.max_price_change_pct:
                    checks["spike_check"] = False
                    rejection_reasons.append(f"SPIKE_DETECTED ({change_pct:.1f}%)")
        
        # 5. Stale data kontrolü
        if symbol in self.price_history and self.price_history[symbol]:
            _, last_time = self.price_history[symbol][-1]
            age_seconds = (timestamp - last_time).total_seconds()
            
            # Çok eski güncelleme varsa uyar
            if age_seconds > self.max_stale_seconds * 10:
                checks["stale_check"] = False
                rejection_reasons.append(f"STALE_DATA ({age_seconds:.0f}s)")
        
        # Sonuç
        is_valid = all(checks.values())
        
        if is_valid:
            # Geçmişe ekle
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=100)
            self.price_history[symbol].append((price, timestamp))
        else:
            self.rejected_data.append({
                "symbol": symbol,
                "price": price,
                "timestamp": timestamp.isoformat(),
                "reasons": rejection_reasons
            })
            
            print(f"{Fore.RED}🚫 VERİ REDDEDİLDİ: {symbol}=${price} - {', '.join(rejection_reasons)}{Style.RESET_ALL}", flush=True)
        
        return {
            "symbol": symbol,
            "price": price,
            "is_valid": is_valid,
            "checks": checks,
            "rejection_reasons": rejection_reasons,
            "action": "ACCEPT" if is_valid else "REJECT"
        }
    
    def validate_ohlcv(self, 
                      symbol: str,
                      open_p: float,
                      high: float,
                      low: float,
                      close: float,
                      volume: float) -> Dict:
        """OHLCV verisini doğrula."""
        issues = []
        
        # High >= Low
        if high < low:
            issues.append("HIGH_LESS_THAN_LOW")
        
        # Open ve Close High/Low arasında
        if open_p > high or open_p < low:
            issues.append("OPEN_OUT_OF_RANGE")
        if close > high or close < low:
            issues.append("CLOSE_OUT_OF_RANGE")
        
        # Volume negatif olamaz
        if volume < 0:
            issues.append("NEGATIVE_VOLUME")
        
        # Tümü sıfır veya negatif
        if all(p <= 0 for p in [open_p, high, low, close]):
            issues.append("ALL_ZERO_OR_NEGATIVE")
        
        is_valid = len(issues) == 0
        
        return {
            "symbol": symbol,
            "is_valid": is_valid,
            "issues": issues,
            "action": "ACCEPT" if is_valid else "REJECT"
        }
    
    def detect_api_error(self, response: Dict) -> Dict:
        """API yanıt hatası tespit."""
        issues = []
        
        # Boş yanıt
        if not response or response == {}:
            issues.append("EMPTY_RESPONSE")
        
        # Error field
        if "error" in response or "Error" in response:
            issues.append("API_ERROR_FIELD")
        
        # Rate limit
        if response.get("code") in [429, "RATE_LIMIT"]:
            issues.append("RATE_LIMITED")
        
        # Timeout
        if response.get("timeout") or response.get("code") == 408:
            issues.append("TIMEOUT")
        
        return {
            "has_error": len(issues) > 0,
            "issues": issues,
            "action": "RETRY" if "RATE_LIMITED" in issues else "REJECT" if issues else "PROCEED"
        }
    
    def get_data_quality_report(self) -> str:
        """Veri kalitesi raporu."""
        total_symbols = len(self.price_history)
        total_rejected = len(self.rejected_data)
        
        report = f"""
<data_integrity>
🧹 VERİ BÜTÜNLÜĞÜ RAPORU
════════════════════════════════════════

📊 ÖZET:
  • Takip Edilen Sembol: {total_symbols}
  • Reddedilen Veri: {total_rejected}

⚙️ AYARLAR:
  • Max Fiyat Değişimi: %{self.max_price_change_pct}
  • Max Veri Yaşı: {self.max_stale_seconds}s
  • Min Fiyat: ${self.min_price}

🚫 SON REDDEDİLEN:
"""
        for rej in self.rejected_data[-5:]:
            report += f"  • {rej['symbol']}: ${rej['price']} - {', '.join(rej['reasons'])}\n"
        
        report += "</data_integrity>\n"
        return report


class RealTimeTCAMonitor:
    """
    Real-Time Transaction Cost Analysis.
    
    Canlı slippage izleme ve otomatik strateji durdurma.
    """
    
    def __init__(self,
                 max_slippage_bps: float = 50,
                 avg_slippage_threshold_bps: float = 20,
                 lookback_trades: int = 20):
        """
        Args:
            max_slippage_bps: Max tek seferde slippage (basis points)
            avg_slippage_threshold_bps: Ortalama slippage eşiği
            lookback_trades: Geriye bakış işlem sayısı
        """
        self.max_slippage_bps = max_slippage_bps
        self.avg_slippage_threshold = avg_slippage_threshold_bps
        self.lookback_trades = lookback_trades
        
        self.trades = deque(maxlen=1000)
        self.paused_strategies = set()
        self.alerts = []
    
    def record_trade(self,
                    strategy_id: str,
                    expected_price: float,
                    executed_price: float,
                    quantity: float,
                    side: str) -> Dict:
        """
        İşlem kaydet ve analiz et.
        """
        direction = 1 if side == "BUY" else -1
        slippage = direction * (executed_price - expected_price)
        slippage_bps = (slippage / expected_price) * 10000
        
        trade = {
            "timestamp": datetime.now(),
            "strategy_id": strategy_id,
            "expected_price": expected_price,
            "executed_price": executed_price,
            "slippage_bps": slippage_bps,
            "cost_usd": abs(slippage) * quantity
        }
        
        self.trades.append(trade)
        
        # Anlık kontrol
        alerts = []
        
        if abs(slippage_bps) > self.max_slippage_bps:
            alerts.append({
                "type": "EXTREME_SLIPPAGE",
                "severity": "HIGH",
                "slippage_bps": slippage_bps,
                "action": "LOG_ONLY"
            })
        
        # Rolling average kontrolü
        strategy_trades = [t for t in self.trades if t["strategy_id"] == strategy_id][-self.lookback_trades:]
        
        if len(strategy_trades) >= 10:
            avg_slippage = np.mean([abs(t["slippage_bps"]) for t in strategy_trades])
            
            if avg_slippage > self.avg_slippage_threshold:
                alerts.append({
                    "type": "HIGH_AVG_SLIPPAGE",
                    "severity": "CRITICAL",
                    "avg_slippage_bps": avg_slippage,
                    "action": "PAUSE_STRATEGY"
                })
                
                self.pause_strategy(strategy_id, f"Avg slippage {avg_slippage:.1f} bps > {self.avg_slippage_threshold}")
        
        self.alerts.extend(alerts)
        
        return {
            "trade": trade,
            "alerts": alerts,
            "strategy_paused": strategy_id in self.paused_strategies
        }
    
    def pause_strategy(self, strategy_id: str, reason: str):
        """Stratejiyi duraklat."""
        self.paused_strategies.add(strategy_id)
        print(f"{Fore.RED}⏸️ STRATEJİ DURAKLATILDI: {strategy_id} - {reason}{Style.RESET_ALL}", flush=True)
    
    def resume_strategy(self, strategy_id: str):
        """Stratejiyi devam ettir."""
        self.paused_strategies.discard(strategy_id)
        print(f"{Fore.GREEN}▶️ STRATEJİ DEVAM: {strategy_id}{Style.RESET_ALL}", flush=True)
    
    def is_strategy_paused(self, strategy_id: str) -> bool:
        """Strateji duraklatılmış mı?"""
        return strategy_id in self.paused_strategies
    
    def get_strategy_tca(self, strategy_id: str) -> Dict:
        """Strateji TCA özeti."""
        trades = [t for t in self.trades if t["strategy_id"] == strategy_id]
        
        if not trades:
            return {"error": "No trades"}
        
        slippages = [t["slippage_bps"] for t in trades]
        costs = [t["cost_usd"] for t in trades]
        
        return {
            "strategy_id": strategy_id,
            "total_trades": len(trades),
            "avg_slippage_bps": np.mean(slippages),
            "total_cost_usd": sum(costs),
            "is_paused": strategy_id in self.paused_strategies
        }
    
    def generate_realtime_tca_report(self) -> str:
        """Real-time TCA raporu."""
        if not self.trades:
            return "Henüz işlem yok"
        
        all_slippages = [t["slippage_bps"] for t in self.trades]
        all_costs = [t["cost_usd"] for t in self.trades]
        
        report = f"""
<realtime_tca>
📉 GERÇEK ZAMANLI TCA
════════════════════════════════════════

📊 GENEL ({len(self.trades)} işlem):
  • Ort. Slippage: {np.mean(all_slippages):.2f} bps
  • Max Slippage: {max(all_slippages):.2f} bps
  • Toplam Maliyet: ${sum(all_costs):,.2f}

⏸️ DURAKLATILAN STRATEJİLER: {len(self.paused_strategies)}
"""
        for strat in list(self.paused_strategies)[:5]:
            report += f"  • {strat}\n"
        
        report += f"""
⚠️ SON ALARMLAR:
"""
        for alert in self.alerts[-5:]:
            report += f"  • {alert['type']}: {alert.get('slippage_bps', alert.get('avg_slippage_bps', 0)):.1f} bps\n"
        
        report += "</realtime_tca>\n"
        return report
