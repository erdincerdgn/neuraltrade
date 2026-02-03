"""
Transaction Cost Analysis Engine
Author: Erdinc Erdogan
Purpose: Analyzes execution quality by measuring implementation shortfall, market impact, timing costs, and broker performance across all trades.
References:
- Implementation Shortfall (Perold, 1988)
- Transaction Cost Analysis (TCA) Framework
- Market Impact Decomposition
Usage:
    tca = TransactionCostAnalyzer()
    result = tca.analyze_trade(decision_price=100, arrival_price=100.02, execution_price=100.05, post_trade_price=100.03, quantity=1000, side="BUY")
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from colorama import Fore, Style


class TransactionCostAnalyzer:
    """
    Transaction Cost Analysis (TCA) Engine.
    
    Her işlemin "gerçek maliyetini" analiz eder:
    - Implementation Shortfall (Hedef vs Gerçekleşen fiyat)
    - Market Impact (Bizim emirimizin fiyatı ne kadar hareket ettirdiği)
    - Timing Cost (Bekleme maliyeti)
    - Broker/Exchange karşılaştırması
    """
    
    def __init__(self):
        self.trades = []
        self.broker_stats = defaultdict(lambda: {
            "total_trades": 0,
            "total_slippage": 0,
            "total_impact": 0,
            "avg_latency_ms": 0
        })
        self.cumulative_costs = {
            "slippage": 0,
            "market_impact": 0,
            "timing": 0,
            "total": 0
        }
    
    def analyze_trade(self,
                     decision_price: float,
                     arrival_price: float,
                     execution_price: float,
                     post_trade_price: float,
                     quantity: float,
                     side: str,
                     broker: str = "DEFAULT",
                     latency_ms: float = 0) -> Dict:
        """
        İşlem maliyeti analizi.
        
        Args:
            decision_price: Karar anındaki fiyat (strateji sinyali)
            arrival_price: Emrin borsaya ulaştığı andaki fiyat
            execution_price: Gerçekleşen fiyat
            post_trade_price: İşlemden 1 dakika sonraki fiyat
            quantity: Miktar
            side: BUY veya SELL
            broker: Broker/Exchange adı
            latency_ms: Gecikme (ms)
        """
        direction = 1 if side == "BUY" else -1
        
        # Implementation Shortfall: Karar fiyatı vs Gerçekleşen
        implementation_shortfall = direction * (execution_price - decision_price) / decision_price
        
        # Timing Cost: Karar vs Varış (bekleme maliyeti)
        timing_cost = direction * (arrival_price - decision_price) / decision_price
        
        # Market Impact: Varış vs Gerçekleşme (bizim emrimizin etkisi)
        market_impact = direction * (execution_price - arrival_price) / arrival_price
        
        # Realized vs Temporary Impact
        temporary_impact = direction * (execution_price - post_trade_price) / execution_price
        permanent_impact = market_impact - temporary_impact
        
        # Slippage: Beklenen vs Gerçekleşen
        slippage = abs(execution_price - decision_price)
        slippage_bps = (slippage / decision_price) * 10000  # Basis points
        
        # Dolar maliyeti
        notional = execution_price * quantity
        cost_usd = abs(implementation_shortfall) * notional
        
        # Sonuç
        result = {
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "quantity": quantity,
            "decision_price": decision_price,
            "execution_price": execution_price,
            "implementation_shortfall_pct": implementation_shortfall * 100,
            "timing_cost_pct": timing_cost * 100,
            "market_impact_pct": market_impact * 100,
            "permanent_impact_pct": permanent_impact * 100,
            "temporary_impact_pct": temporary_impact * 100,
            "slippage_bps": slippage_bps,
            "cost_usd": cost_usd,
            "broker": broker,
            "latency_ms": latency_ms
        }
        
        # Kaydet
        self.trades.append(result)
        
        # Broker stats güncelle
        self.broker_stats[broker]["total_trades"] += 1
        self.broker_stats[broker]["total_slippage"] += slippage_bps
        self.broker_stats[broker]["total_impact"] += abs(market_impact) * 100
        
        # Update cumulative
        self.cumulative_costs["slippage"] += slippage_bps
        self.cumulative_costs["market_impact"] += abs(market_impact) * 100
        self.cumulative_costs["timing"] += abs(timing_cost) * 100
        self.cumulative_costs["total"] += cost_usd
        
        return result
    
    def get_broker_ranking(self) -> List[Dict]:
        """Broker performans sıralaması."""
        rankings = []
        
        for broker, stats in self.broker_stats.items():
            if stats["total_trades"] > 0:
                avg_slippage = stats["total_slippage"] / stats["total_trades"]
                avg_impact = stats["total_impact"] / stats["total_trades"]
                
                # Score: Düşük = iyi
                score = avg_slippage + avg_impact
                
                rankings.append({
                    "broker": broker,
                    "total_trades": stats["total_trades"],
                    "avg_slippage_bps": avg_slippage,
                    "avg_market_impact_pct": avg_impact,
                    "score": score,
                    "grade": "A" if score < 5 else "B" if score < 10 else "C" if score < 20 else "D"
                })
        
        return sorted(rankings, key=lambda x: x["score"])
    
    def calculate_yearly_impact(self, annual_volume_usd: float = 10_000_000) -> Dict:
        """Yıllık maliyet projeksiyonu."""
        if not self.trades:
            return {"error": "No trades analyzed"}
        
        # Ortalama maliyetler
        avg_is = np.mean([abs(t["implementation_shortfall_pct"]) for t in self.trades])
        avg_slippage = np.mean([t["slippage_bps"] for t in self.trades])
        
        # Yıllık projeksiyonlar
        yearly_is_cost = annual_volume_usd * (avg_is / 100)
        yearly_slippage_cost = annual_volume_usd * (avg_slippage / 10000)
        
        return {
            "annual_volume_usd": annual_volume_usd,
            "avg_implementation_shortfall_pct": avg_is,
            "avg_slippage_bps": avg_slippage,
            "projected_yearly_is_cost_usd": yearly_is_cost,
            "projected_yearly_slippage_cost_usd": yearly_slippage_cost,
            "total_projected_cost_usd": yearly_is_cost + yearly_slippage_cost
        }
    
    def suggest_improvements(self) -> List[Dict]:
        """İyileştirme önerileri."""
        suggestions = []
        
        if not self.trades:
            return suggestions
        
        # Yüksek slippage kontrolü
        avg_slippage = np.mean([t["slippage_bps"] for t in self.trades])
        if avg_slippage > 10:
            suggestions.append({
                "issue": "HIGH_SLIPPAGE",
                "severity": "HIGH",
                "current_value": f"{avg_slippage:.1f} bps",
                "suggestion": "Limit emirler kullan veya daha küçük parçalara böl (TWAP/VWAP)"
            })
        
        # Yüksek market impact
        avg_impact = np.mean([abs(t["market_impact_pct"]) for t in self.trades])
        if avg_impact > 0.5:
            suggestions.append({
                "issue": "HIGH_MARKET_IMPACT",
                "severity": "MEDIUM",
                "current_value": f"{avg_impact:.2f}%",
                "suggestion": "Emir boyutunu küçült veya dark pool kullan"
            })
        
        # Timing cost
        avg_timing = np.mean([abs(t["timing_cost_pct"]) for t in self.trades])
        if avg_timing > 0.3:
            suggestions.append({
                "issue": "HIGH_TIMING_COST",
                "severity": "MEDIUM",
                "current_value": f"{avg_timing:.2f}%",
                "suggestion": "Daha hızlı execution veya colocation kullan"
            })
        
        # Broker karşılaştırması
        rankings = self.get_broker_ranking()
        if len(rankings) > 1:
            best = rankings[0]
            worst = rankings[-1]
            if worst["score"] > best["score"] * 2:
                suggestions.append({
                    "issue": "BROKER_DISPARITY",
                    "severity": "HIGH",
                    "current_value": f"Best: {best['broker']} ({best['score']:.1f}), Worst: {worst['broker']} ({worst['score']:.1f})",
                    "suggestion": f"Emirleri {best['broker']}'a yönlendir"
                })
        
        return suggestions
    
    def generate_tca_report(self) -> str:
        """TCA raporu."""
        if not self.trades:
            return "Henüz işlem analizi yok"
        
        broker_rankings = self.get_broker_ranking()
        yearly = self.calculate_yearly_impact()
        suggestions = self.suggest_improvements()
        
        report = f"""
<transaction_cost_analysis>
📉 İŞLEM MALİYETİ ANALİZİ (TCA)
════════════════════════════════════════

📊 ÖZET ({len(self.trades)} işlem):
  • Toplam Slippage: {self.cumulative_costs['slippage']:.1f} bps
  • Toplam Market Impact: {self.cumulative_costs['market_impact']:.2f}%
  • Toplam USD Maliyet: ${self.cumulative_costs['total']:,.2f}

💰 YILLIK PROJEKSİYON:
  • Impl. Shortfall: ${yearly.get('projected_yearly_is_cost_usd', 0):,.0f}
  • Slippage: ${yearly.get('projected_yearly_slippage_cost_usd', 0):,.0f}

🏆 BROKER SIRALAMASI:
"""
        for rank in broker_rankings[:5]:
            grade_color = Fore.GREEN if rank['grade'] == 'A' else Fore.YELLOW if rank['grade'] in ['B', 'C'] else Fore.RED
            report += f"  {rank['grade']}: {rank['broker']} - {rank['score']:.1f} puan ({rank['total_trades']} işlem)\n"
        
        report += f"""
💡 İYİLEŞTİRME ÖNERİLERİ:
"""
        for sug in suggestions[:3]:
            report += f"  ⚠️ {sug['issue']}: {sug['suggestion']}\n"
        
        report += "</transaction_cost_analysis>\n"
        return report
