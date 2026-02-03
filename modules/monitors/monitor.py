
"""
Alpha Decay Monitor with Auto-Retirement
Author: Erdinc Erdogan
Purpose: Monitors strategy performance degradation over time and automatically quarantines or retires strategies showing alpha decay.
References:
- Alpha Decay Detection
- Strategy Lifecycle Management
- Performance Attribution Analysis
Usage:
    monitor = AlphaDecayMonitor(decay_threshold=0.3)
    monitor.register_strategy("momentum_v1", "Momentum Strategy")
    status = monitor.check_alpha_decay("momentum_v1", current_sharpe=0.8)
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
from colorama import Fore, Style


class AlphaDecayMonitor:
    """
    Alpha Decay Monitor.
    
    Stratejilerin performans bozulmasını (alpha decay) izler.
    
    Neden önemli:
    - Her strateji sonunda kalabalıklaşır (crowded)
    - Alpha azalır çünkü herkes aynı şeyi yapar
    - Erken tespit = kayıp önleme
    """
    
    # Strateji durumları
    STATUS = {
        "ACTIVE": {"color": "green", "action": "Full allocation"},
        "WATCH": {"color": "yellow", "action": "Reduced allocation"},
        "QUARANTINE": {"color": "orange", "action": "No new positions"},
        "RETIRED": {"color": "red", "action": "Fully liquidated"}
    }
    
    def __init__(self, 
                 decay_threshold: float = 0.3,
                 quarantine_days: int = 30,
                 retirement_threshold: float = 0.5):
        """
        Args:
            decay_threshold: Alpha decay uyarı eşiği
            quarantine_days: Karantina süresi
            retirement_threshold: Emeklilik eşiği
        """
        self.decay_threshold = decay_threshold
        self.quarantine_days = quarantine_days
        self.retirement_threshold = retirement_threshold
        
        self.strategies = {}
        self.performance_history = {}
        self.alerts = []
    
    def register_strategy(self, 
                         strategy_id: str,
                         name: str,
                         initial_sharpe: float = None):
        """Strateji kaydet."""
        self.strategies[strategy_id] = {
            "id": strategy_id,
            "name": name,
            "status": "ACTIVE",
            "registered_at": datetime.now(),
            "initial_sharpe": initial_sharpe,
            "peak_sharpe": initial_sharpe or 0,
            "current_sharpe": initial_sharpe or 0,
            "quarantine_start": None,
            "allocation_pct": 100
        }
        self.performance_history[strategy_id] = deque(maxlen=365)
        
        print(f"{Fore.GREEN}📊 Strateji kayıt: {name}{Style.RESET_ALL}", flush=True)
    
    def update_performance(self, 
                          strategy_id: str,
                          daily_return: float,
                          volatility: float = None) -> Dict:
        """
        Günlük performans güncelle.
        
        Args:
            strategy_id: Strateji ID
            daily_return: Günlük getiri
            volatility: Gerçekleşen volatilite
        """
        if strategy_id not in self.strategies:
            return {"error": "Strateji bulunamadı"}
        
        strategy = self.strategies[strategy_id]
        history = self.performance_history[strategy_id]
        
        # Geçmişe ekle
        history.append({
            "date": datetime.now(),
            "return": daily_return,
            "volatility": volatility or abs(daily_return) * 15
        })
        
        # Rolling Sharpe hesapla (son 60 gün)
        if len(history) >= 20:
            recent_returns = [h["return"] for h in list(history)[-60:]]
            rolling_sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252) if np.std(recent_returns) > 0 else 0
            
            strategy["current_sharpe"] = rolling_sharpe
            
            # Peak güncelle
            if rolling_sharpe > strategy["peak_sharpe"]:
                strategy["peak_sharpe"] = rolling_sharpe
            
            # Decay hesapla
            decay = self._calculate_decay(strategy_id)
            
            # Durum güncelle
            self._update_status(strategy_id, decay)
            
            return {
                "strategy_id": strategy_id,
                "current_sharpe": rolling_sharpe,
                "peak_sharpe": strategy["peak_sharpe"],
                "decay_pct": decay * 100,
                "status": strategy["status"]
            }
        
        return {"strategy_id": strategy_id, "status": "COLLECTING_DATA"}
    
    def _calculate_decay(self, strategy_id: str) -> float:
        """
        Alpha decay hesapla.
        
        Decay = (Peak Sharpe - Current Sharpe) / Peak Sharpe
        """
        strategy = self.strategies[strategy_id]
        
        peak = strategy["peak_sharpe"]
        current = strategy["current_sharpe"]
        
        if peak <= 0:
            return 0
        
        decay = (peak - current) / peak
        return max(0, decay)
    
    def _update_status(self, strategy_id: str, decay: float):
        """Strateji durumunu güncelle."""
        strategy = self.strategies[strategy_id]
        old_status = strategy["status"]
        
        if strategy["status"] == "RETIRED":
            return  # Emekli stratejiler geri gelmez
        
        # Decay bazlı durum
        if decay >= self.retirement_threshold:
            new_status = "RETIRED"
            strategy["allocation_pct"] = 0
            
        elif decay >= self.decay_threshold:
            if strategy["status"] == "ACTIVE":
                strategy["quarantine_start"] = datetime.now()
            new_status = "QUARANTINE"
            strategy["allocation_pct"] = 0
            
        elif decay >= self.decay_threshold * 0.5:
            new_status = "WATCH"
            strategy["allocation_pct"] = 50
            
        else:
            new_status = "ACTIVE"
            strategy["allocation_pct"] = 100
            strategy["quarantine_start"] = None
        
        # Karantina süre kontrolü
        if strategy["status"] == "QUARANTINE" and strategy["quarantine_start"]:
            days_in_quarantine = (datetime.now() - strategy["quarantine_start"]).days
            
            if days_in_quarantine > self.quarantine_days:
                if decay < self.decay_threshold * 0.5:
                    new_status = "ACTIVE"  # Toparlandı
                    strategy["quarantine_start"] = None
                else:
                    new_status = "RETIRED"  # Toparlanamadı
        
        strategy["status"] = new_status
        
        # Durum değişti mi?
        if new_status != old_status:
            alert = {
                "strategy_id": strategy_id,
                "name": strategy["name"],
                "from_status": old_status,
                "to_status": new_status,
                "decay_pct": decay * 100,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            status_color = Fore.GREEN if new_status == "ACTIVE" else Fore.YELLOW if new_status == "WATCH" else Fore.RED
            
            print(f"{status_color}⚠️ STRATEJİ DURUM: {strategy['name']} {old_status} → {new_status}{Style.RESET_ALL}", flush=True)
    
    def get_active_strategies(self) -> List[Dict]:
        """Aktif stratejileri getir."""
        return [s for s in self.strategies.values() if s["status"] == "ACTIVE"]
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Portföy tahsisini hesapla."""
        total_allocation = sum(s["allocation_pct"] for s in self.strategies.values())
        
        if total_allocation == 0:
            return {}
        
        return {
            s["id"]: s["allocation_pct"] / total_allocation
            for s in self.strategies.values()
            if s["allocation_pct"] > 0
        }
    
    def suggest_replacement(self, retired_strategy_id: str, candidate_strategies: List[Dict]) -> Optional[Dict]:
        """
        Emekli strateji için yedek öner.
        
        Args:
            retired_strategy_id: Emekli strateji
            candidate_strategies: Yedek adaylar (Genetik Algoritma'dan)
        """
        if not candidate_strategies:
            return None
        
        retired = self.strategies.get(retired_strategy_id)
        if not retired:
            return None
        
        # En yüksek Sharpe'lı adayı seç
        best = max(candidate_strategies, key=lambda x: x.get("sharpe", 0))
        
        if best.get("sharpe", 0) > retired.get("initial_sharpe", 0) * 0.8:
            return {
                "replacement": best,
                "reason": f"Sharpe {best.get('sharpe', 0):.2f} > minimum threshold",
                "action": "DEPLOY_CANDIDATE"
            }
        
        return {
            "replacement": None,
            "reason": "Yeterince iyi aday yok",
            "action": "WAIT_FOR_EVOLUTION"
        }
    
    def generate_decay_report(self) -> str:
        """Alpha decay raporu."""
        report = f"""
<alpha_decay_monitor>
💀 ALPHA DECAY RAPORU
════════════════════════════════════════

📊 STRATEJİ DURUMU:
"""
        status_counts = {"ACTIVE": 0, "WATCH": 0, "QUARANTINE": 0, "RETIRED": 0}
        
        for strategy in self.strategies.values():
            status_counts[strategy["status"]] += 1
            decay = self._calculate_decay(strategy["id"])
            
            status_emoji = {"ACTIVE": "🟢", "WATCH": "🟡", "QUARANTINE": "🟠", "RETIRED": "🔴"}
            
            report += f"""  {status_emoji.get(strategy['status'], '⚪')} {strategy['name']}:
    • Sharpe: {strategy['peak_sharpe']:.2f} → {strategy['current_sharpe']:.2f}
    • Decay: {decay*100:.1f}%
    • Allocation: {strategy['allocation_pct']}%

"""
        
        report += f"""📈 ÖZET:
  • Aktif: {status_counts['ACTIVE']}
  • İzlemede: {status_counts['WATCH']}
  • Karantinada: {status_counts['QUARANTINE']}
  • Emekli: {status_counts['RETIRED']}

⚠️ SON ALARMLAR:
"""
        for alert in self.alerts[-5:]:
            report += f"  • {alert['name']}: {alert['from_status']} → {alert['to_status']}\n"
        
        report += "</alpha_decay_monitor>\n"
        return report
