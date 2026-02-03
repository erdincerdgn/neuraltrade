"""
Internal Prediction Markets (Skin in the Game)
Author: Erdinc Erdogan
Purpose: Implements an internal prediction market where agents bet virtual capital on outcomes, creating reputation-weighted consensus through economic incentives.
References:
- Prediction Markets (Hanson, 2003)
- Skin in the Game Principle (Taleb)
- Market Scoring Rules
Usage:
    market = PredictionMarket()
    market.place_bet(agent_id="bull_1", outcome="UP", amount=100)
    consensus = market.get_weighted_consensus()
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from colorama import Fore, Style


class AgentWallet:
    """
    Ajan sanal cüzdanı.
    """
    
    def __init__(self, agent_id: str, initial_balance: float = 1000):
        self.agent_id = agent_id
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.locked_balance = 0
        self.bet_history = []
        self.win_count = 0
        self.loss_count = 0
    
    def lock_bet(self, amount: float) -> bool:
        """Bahis için para kilitle."""
        if amount > self.balance:
            return False
        
        self.balance -= amount
        self.locked_balance += amount
        return True
    
    def settle_win(self, amount: float, payout: float):
        """Kazanç öde."""
        self.locked_balance -= amount
        self.balance += payout
        self.win_count += 1
    
    def settle_loss(self, amount: float):
        """Kayıp."""
        self.locked_balance -= amount
        self.loss_count += 1
    
    def get_reputation(self) -> float:
        """Doğruluk oranına göre itibar."""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.5
        
        win_rate = self.win_count / total
        
        # Ağırlıklı itibar: Varlık × Win Rate
        wealth_factor = self.balance / self.initial_balance
        
        return win_rate * min(wealth_factor, 2.0)


class PredictionMarket:
    """
    Internal Prediction Market.
    
    "Skin in the Game" prensibi:
    - Ajanlar tahminlerine bahis koyar
    - Doğru tahmin = para kazanır
    - Yanlış tahmin = para kaybeder
    - Natural Selection: Yanlış yapan sesini kaybeder
    """
    
    def __init__(self):
        self.wallets = {}  # agent_id -> AgentWallet
        self.active_markets = {}
        self.resolved_markets = []
        self.market_counter = 0
    
    def register_agent(self, agent_id: str, initial_balance: float = 1000):
        """Ajan kaydet."""
        self.wallets[agent_id] = AgentWallet(agent_id, initial_balance)
        print(f"{Fore.CYAN}🎰 Ajan kayıt: {agent_id} (${initial_balance}){Style.RESET_ALL}", flush=True)
    
    def create_market(self, 
                     question: str,
                     options: List[str],
                     resolution_time: datetime = None) -> str:
        """
        Tahmin piyasası oluştur.
        
        Args:
            question: "BTC yarın yükselir mi?"
            options: ["EVET", "HAYIR"] veya ["AL", "SAT", "BEKLE"]
            resolution_time: Sonuçlanma zamanı
        """
        self.market_counter += 1
        market_id = f"MKT_{self.market_counter}"
        
        self.active_markets[market_id] = {
            "id": market_id,
            "question": question,
            "options": options,
            "bets": defaultdict(list),  # option -> [(agent, amount), ...]
            "total_pool": 0,
            "option_pools": {opt: 0 for opt in options},
            "created_at": datetime.now(),
            "resolution_time": resolution_time,
            "status": "OPEN"
        }
        
        print(f"{Fore.CYAN}🗳️ Piyasa açıldı: {question}{Style.RESET_ALL}", flush=True)
        
        return market_id
    
    def place_bet(self, 
                 market_id: str,
                 agent_id: str,
                 option: str,
                 amount: float) -> Dict:
        """
        Bahis koy.
        
        Args:
            market_id: Piyasa ID
            agent_id: Ajan ID
            option: Seçilen opsiyon
            amount: Bahis miktarı
        """
        if market_id not in self.active_markets:
            return {"error": "Piyasa bulunamadı"}
        
        market = self.active_markets[market_id]
        
        if market["status"] != "OPEN":
            return {"error": "Piyasa kapalı"}
        
        if option not in market["options"]:
            return {"error": f"Geçersiz opsiyon: {option}"}
        
        if agent_id not in self.wallets:
            return {"error": "Ajan bulunamadı"}
        
        wallet = self.wallets[agent_id]
        
        if not wallet.lock_bet(amount):
            return {"error": "Yetersiz bakiye"}
        
        # Bahsi kaydet
        market["bets"][option].append({
            "agent_id": agent_id,
            "amount": amount,
            "timestamp": datetime.now()
        })
        market["total_pool"] += amount
        market["option_pools"][option] += amount
        
        # Mevcut oranları hesapla
        odds = self._calculate_odds(market)
        
        return {
            "success": True,
            "market_id": market_id,
            "agent_id": agent_id,
            "option": option,
            "amount": amount,
            "current_odds": odds
        }
    
    def _calculate_odds(self, market: Dict) -> Dict:
        """Oranları hesapla (parimutuel system)."""
        total = market["total_pool"]
        
        if total == 0:
            return {opt: 2.0 for opt in market["options"]}
        
        odds = {}
        for opt in market["options"]:
            opt_pool = market["option_pools"][opt]
            if opt_pool > 0:
                odds[opt] = total / opt_pool
            else:
                odds[opt] = float('inf')
        
        return odds
    
    def resolve_market(self, market_id: str, winning_option: str) -> Dict:
        """
        Piyasayı sonuçlandır.
        
        Args:
            market_id: Piyasa ID
            winning_option: Kazanan opsiyon
        """
        if market_id not in self.active_markets:
            return {"error": "Piyasa bulunamadı"}
        
        market = self.active_markets[market_id]
        
        if winning_option not in market["options"]:
            return {"error": f"Geçersiz sonuç: {winning_option}"}
        
        print(f"{Fore.GREEN}✅ Piyasa sonuçlandı: {winning_option}{Style.RESET_ALL}", flush=True)
        
        # Kazananları öde
        winners = market["bets"][winning_option]
        winner_pool = market["option_pools"][winning_option]
        total_pool = market["total_pool"]
        
        payouts = []
        
        for bet in winners:
            agent_id = bet["agent_id"]
            amount = bet["amount"]
            
            # Payout = (bet_amount / winner_pool) * total_pool
            payout = (amount / winner_pool) * total_pool if winner_pool > 0 else 0
            
            self.wallets[agent_id].settle_win(amount, payout)
            payouts.append({
                "agent_id": agent_id,
                "bet_amount": amount,
                "payout": payout,
                "profit": payout - amount
            })
        
        # Kaybedenleri işle
        for opt, bets in market["bets"].items():
            if opt != winning_option:
                for bet in bets:
                    self.wallets[bet["agent_id"]].settle_loss(bet["amount"])
        
        # Piyasayı kapat
        market["status"] = "RESOLVED"
        market["winning_option"] = winning_option
        market["payouts"] = payouts
        
        self.resolved_markets.append(market)
        del self.active_markets[market_id]
        
        return {
            "market_id": market_id,
            "winning_option": winning_option,
            "total_pool": total_pool,
            "winners": len(payouts),
            "payouts": payouts
        }
    
    def get_market_consensus(self, market_id: str) -> Dict:
        """
        Piyasa konsensüsü.
        
        En çok para yatırılan opsiyon = piyasa tahmini.
        """
        if market_id not in self.active_markets:
            return {"error": "Piyasa bulunamadı"}
        
        market = self.active_markets[market_id]
        
        # Ağırlıklı tahmin (reputation × bet)
        weighted_votes = defaultdict(float)
        
        for opt, bets in market["bets"].items():
            for bet in bets:
                agent_id = bet["agent_id"]
                amount = bet["amount"]
                reputation = self.wallets[agent_id].get_reputation()
                
                weighted_votes[opt] += amount * reputation
        
        total_weight = sum(weighted_votes.values())
        
        if total_weight == 0:
            return {"error": "Henüz bahis yok"}
        
        consensus = max(weighted_votes.items(), key=lambda x: x[1])
        
        return {
            "market_id": market_id,
            "consensus": consensus[0],
            "confidence": consensus[1] / total_weight,
            "distribution": {
                opt: w / total_weight 
                for opt, w in weighted_votes.items()
            }
        }
    
    def get_agent_leaderboard(self) -> List[Dict]:
        """Ajan liderlik tablosu."""
        leaderboard = []
        
        for agent_id, wallet in self.wallets.items():
            leaderboard.append({
                "agent_id": agent_id,
                "balance": wallet.balance,
                "roi_pct": ((wallet.balance - wallet.initial_balance) / wallet.initial_balance) * 100,
                "win_rate": wallet.win_count / max(1, wallet.win_count + wallet.loss_count),
                "reputation": wallet.get_reputation(),
                "total_bets": wallet.win_count + wallet.loss_count
            })
        
        return sorted(leaderboard, key=lambda x: -x["balance"])
    
    def generate_market_report(self) -> str:
        """Prediction market raporu."""
        leaderboard = self.get_agent_leaderboard()
        
        report = f"""
<prediction_markets>
🗳️ İÇ TAHMİN PİYASASI
════════════════════════════════════════

📊 ÖZET:
  • Aktif Piyasa: {len(self.active_markets)}
  • Kapanmış Piyasa: {len(self.resolved_markets)}
  • Kayıtlı Ajan: {len(self.wallets)}

🏆 LİDERLİK TABLOSU:
"""
        for i, agent in enumerate(leaderboard[:5], 1):
            roi_color = Fore.GREEN if agent["roi_pct"] > 0 else Fore.RED
            report += f"  {i}. {agent['agent_id']}: ${agent['balance']:.0f} ({roi_color}{agent['roi_pct']:+.1f}%{Style.RESET_ALL})\n"
        
        report += """
💡 SKIN IN THE GAME:
  • Yanlış tahmin = Para kaybı
  • Doğru tahmin = Güç kazanımı
  • Natural Selection aktif

</prediction_markets>
"""
        return report
