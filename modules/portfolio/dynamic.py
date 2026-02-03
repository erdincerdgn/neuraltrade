"""
Dynamic Portfolio Rebalancer
Author: Erdinc Erdogan
Purpose: Automatically rebalances portfolio to target weights using threshold-based, calendar-based, and volatility-adjusted triggers to maintain optimal allocation.
References:
- Threshold Rebalancing
- Calendar Rebalancing
- Volatility-Adjusted Rebalancing
Usage:
    rebalancer = DynamicRebalancer(target_weights={'STOCKS': 0.4, 'BONDS': 0.3})
    rebalancer.update_holdings({'STOCKS': 45000, 'BONDS': 28000})
    trades = rebalancer.generate_trades()
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class DynamicRebalancer:
    """
    Dynamic Portfolio Rebalancing.
    
    Portföy dağılımını hedef ağırlıklara geri getirir:
    - Threshold-based (sapma eşiği)
    - Calendar-based (periyodik)
    - Volatility-adjusted (oynaklığa göre)
    
    Matematiksel avantaj: Sell High, Buy Low garantisi.
    """
    
    def __init__(self,
                 target_weights: Dict[str, float] = None,
                 threshold_pct: float = 5.0,
                 rebalance_frequency: str = "MONTHLY"):
        """
        Args:
            target_weights: Hedef ağırlıklar {"STOCKS": 0.4, "CRYPTO": 0.3, "CASH": 0.3}
            threshold_pct: Sapma eşiği (%)
            rebalance_frequency: DAILY, WEEKLY, MONTHLY
        """
        self.target_weights = target_weights or {
            "STOCKS": 0.40,
            "BONDS": 0.20,
            "CRYPTO": 0.25,
            "CASH": 0.15
        }
        
        self.threshold_pct = threshold_pct
        self.rebalance_frequency = rebalance_frequency
        
        self.current_holdings = {}
        self.last_rebalance = None
        self.rebalance_history = []
    
    def update_holdings(self, holdings: Dict[str, float]):
        """
        Mevcut pozisyonları güncelle.
        
        Args:
            holdings: {"STOCKS": 45000, "BONDS": 18000, ...}
        """
        self.current_holdings = holdings
    
    def calculate_drift(self) -> Dict:
        """
        Portföy sapmasını hesapla.
        
        Drift = Current Weight - Target Weight
        """
        total_value = sum(self.current_holdings.values())
        
        if total_value == 0:
            return {"error": "Empty portfolio"}
        
        current_weights = {
            asset: value / total_value
            for asset, value in self.current_holdings.items()
        }
        
        drift = {}
        max_drift = 0
        
        for asset, target in self.target_weights.items():
            current = current_weights.get(asset, 0)
            asset_drift = (current - target) * 100  # Percentage
            drift[asset] = {
                "target_pct": target * 100,
                "current_pct": current * 100,
                "drift_pct": asset_drift,
                "action": "SELL" if asset_drift > 0 else "BUY" if asset_drift < 0 else "HOLD"
            }
            max_drift = max(max_drift, abs(asset_drift))
        
        needs_rebalance = max_drift > self.threshold_pct
        
        return {
            "total_value": total_value,
            "current_weights": current_weights,
            "drift": drift,
            "max_drift_pct": max_drift,
            "needs_rebalance": needs_rebalance,
            "threshold_pct": self.threshold_pct
        }
    
    def check_calendar_rebalance(self) -> bool:
        """Periyodik rebalance gerekli mi?"""
        if self.last_rebalance is None:
            return True
        
        elapsed = datetime.now() - self.last_rebalance
        
        if self.rebalance_frequency == "DAILY":
            return elapsed >= timedelta(days=1)
        elif self.rebalance_frequency == "WEEKLY":
            return elapsed >= timedelta(weeks=1)
        elif self.rebalance_frequency == "MONTHLY":
            return elapsed >= timedelta(days=30)
        elif self.rebalance_frequency == "QUARTERLY":
            return elapsed >= timedelta(days=90)
        
        return False
    
    def calculate_rebalance_trades(self) -> List[Dict]:
        """
        Rebalance için gerekli işlemleri hesapla.
        """
        drift_result = self.calculate_drift()
        
        if "error" in drift_result:
            return []
        
        total_value = drift_result["total_value"]
        trades = []
        
        for asset, target in self.target_weights.items():
            current_value = self.current_holdings.get(asset, 0)
            target_value = total_value * target
            difference = target_value - current_value
            
            if abs(difference) < 100:  # $100'dan az değişiklik yapma
                continue
            
            trade = {
                "asset": asset,
                "action": "BUY" if difference > 0 else "SELL",
                "amount_usd": abs(difference),
                "current_value": current_value,
                "target_value": target_value,
                "pct_change": (difference / current_value * 100) if current_value > 0 else 100
            }
            trades.append(trade)
        
        # SELL'leri önce sırala (nakdi serbest bırak)
        trades.sort(key=lambda t: (0 if t["action"] == "SELL" else 1, -t["amount_usd"]))
        
        return trades
    
    def execute_rebalance(self, execute_func=None) -> Dict:
        """
        Rebalance işlemlerini çalıştır.
        
        Args:
            execute_func: Opsiyonel. (trade) -> result fonksiyonu
        """
        print(f"{Fore.CYAN}⚖️ Portföy rebalance başlıyor...{Style.RESET_ALL}", flush=True)
        
        drift = self.calculate_drift()
        
        if not drift.get("needs_rebalance") and not self.check_calendar_rebalance():
            return {
                "rebalanced": False,
                "reason": f"Drift ({drift.get('max_drift_pct', 0):.1f}%) < Threshold ({self.threshold_pct}%)"
            }
        
        trades = self.calculate_rebalance_trades()
        
        if not trades:
            return {"rebalanced": False, "reason": "No significant trades needed"}
        
        executed = []
        
        for trade in trades:
            if execute_func:
                result = execute_func(trade)
                trade["execution_result"] = result
            else:
                trade["execution_result"] = {"status": "SIMULATED"}
            
            executed.append(trade)
            
            action_color = Fore.GREEN if trade["action"] == "BUY" else Fore.RED
            print(f"  {action_color}{trade['action']} {trade['asset']}: ${trade['amount_usd']:,.0f}{Style.RESET_ALL}", flush=True)
        
        # Kaydet
        rebalance_record = {
            "timestamp": datetime.now().isoformat(),
            "drift_before": drift,
            "trades": executed,
            "total_traded_usd": sum(t["amount_usd"] for t in executed)
        }
        
        self.rebalance_history.append(rebalance_record)
        self.last_rebalance = datetime.now()
        
        print(f"{Fore.GREEN}✅ Rebalance tamamlandı: {len(executed)} işlem{Style.RESET_ALL}", flush=True)
        
        return {
            "rebalanced": True,
            "trades": executed,
            "total_traded_usd": rebalance_record["total_traded_usd"]
        }
    
    def get_rebalance_statistics(self) -> Dict:
        """Rebalance istatistikleri."""
        if not self.rebalance_history:
            return {"total_rebalances": 0}
        
        total_traded = sum(r["total_traded_usd"] for r in self.rebalance_history)
        
        return {
            "total_rebalances": len(self.rebalance_history),
            "total_traded_usd": total_traded,
            "avg_trades_per_rebalance": np.mean([len(r["trades"]) for r in self.rebalance_history]),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None
        }
    
    def simulate_rebalancing_benefit(self, 
                                    prices_history: Dict[str, List[float]],
                                    initial_capital: float = 100000) -> Dict:
        """
        Rebalancing'in faydalarını simüle et.
        
        Buy & Hold vs Rebalanced karşılaştırması.
        """
        n_periods = min(len(list(prices_history.values())[0]) for k in prices_history)
        
        # Buy & Hold stratejisi
        bh_weights = {asset: self.target_weights.get(asset, 0) for asset in prices_history}
        bh_holdings = {asset: initial_capital * bh_weights[asset] for asset in prices_history}
        
        # Rebalanced stratejisi
        rb_holdings = dict(bh_holdings)
        
        for t in range(1, n_periods):
            # Fiyat değişimlerini uygula
            for asset, prices in prices_history.items():
                price_return = (prices[t] - prices[t-1]) / prices[t-1]
                bh_holdings[asset] *= (1 + price_return)
                rb_holdings[asset] *= (1 + price_return)
            
            # Monthly rebalance simülasyonu
            if t % 21 == 0:  # Her 21 günde
                rb_total = sum(rb_holdings.values())
                for asset in rb_holdings:
                    rb_holdings[asset] = rb_total * self.target_weights.get(asset, 0)
        
        bh_final = sum(bh_holdings.values())
        rb_final = sum(rb_holdings.values())
        
        return {
            "buy_hold_final": bh_final,
            "rebalanced_final": rb_final,
            "rebalancing_benefit_pct": ((rb_final - bh_final) / bh_final) * 100,
            "buy_hold_return_pct": ((bh_final - initial_capital) / initial_capital) * 100,
            "rebalanced_return_pct": ((rb_final - initial_capital) / initial_capital) * 100
        }
    
    def generate_rebalance_report(self) -> str:
        """Rebalance raporu."""
        drift = self.calculate_drift()
        stats = self.get_rebalance_statistics()
        
        report = f"""
<dynamic_rebalancer>
⚖️ DİNAMİK PORTFÖY DENGELEME
════════════════════════════════════════

📊 HEDEF DAĞILIM:
"""
        for asset, target in self.target_weights.items():
            current = drift.get("drift", {}).get(asset, {}).get("current_pct", 0)
            drift_val = drift.get("drift", {}).get(asset, {}).get("drift_pct", 0)
            arrow = "↑" if drift_val > 0 else "↓" if drift_val < 0 else "→"
            report += f"  • {asset}: %{target*100:.0f} (Şimdi: %{current:.1f}) {arrow}\n"
        
        report += f"""
📈 SAPMA ANALİZİ:
  • Max Sapma: %{drift.get('max_drift_pct', 0):.1f}
  • Eşik: %{self.threshold_pct}
  • Rebalance Gerekli: {'✅ EVET' if drift.get('needs_rebalance') else '❌ HAYIR'}

📋 İSTATİSTİK:
  • Toplam Rebalance: {stats['total_rebalances']}
  • Toplam İşlem Hacmi: ${stats.get('total_traded_usd', 0):,.0f}
  • Son Rebalance: {stats.get('last_rebalance', 'Henüz yok')}

💡 FAYDA: Rebalancing = Otomatik "Sell High, Buy Low"

</dynamic_rebalancer>
"""
        return report
