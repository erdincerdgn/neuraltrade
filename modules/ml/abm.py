"""
Agent-Based Market Simulation (Digital Twin)
Author: Erdinc Erdogan
Purpose: Creates a digital twin of the market with heterogeneous agents (trend followers, contrarians, institutions, HFT) to simulate crisis scenarios and test strategies.
References:
- Agent-Based Modeling in Finance (LeBaron, 2006)
- Heterogeneous Agent Models
- Zero Intelligence Trader Models
Usage:
    abm = AgentBasedMarket(n_agents=100)
    abm.simulate_flash_crash(severity=0.8)
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from collections import defaultdict
from colorama import Fore, Style
import random


class AgentType(Enum):
    """Yatırımcı ajan tipleri."""
    TREND_FOLLOWER = "trend_follower"      # Momentum takipçisi
    CONTRARIAN = "contrarian"              # Ters yönlü
    NOISE_TRADER = "noise_trader"          # Rastgele
    FUNDAMENTAL = "fundamental"            # Değer yatırımcısı
    NEWS_REACTIVE = "news_reactive"        # Habere tepkici
    INSTITUTIONAL = "institutional"        # Kurumsal (büyük emirler)
    HFT = "hft"                           # Yüksek frekanslı


class MarketAgent:
    """
    Simüle edilmiş yatırımcı ajan.
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: AgentType,
                 capital: float = 10000,
                 risk_tolerance: float = 0.5):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capital = capital
        self.position = 0  # Pozitif = long, negatif = short
        self.risk_tolerance = risk_tolerance
        self.trade_history = []
        self.pnl = 0
    
    def decide(self, market_state: Dict) -> Dict:
        """
        Ajan kararı.
        
        Returns:
            {"action": BUY/SELL/HOLD, "quantity": float}
        """
        price = market_state.get("price", 100)
        trend = market_state.get("trend", 0)  # -1 = düşüş, 0 = yatay, 1 = yükseliş
        news_sentiment = market_state.get("news_sentiment", 0)
        volatility = market_state.get("volatility", 0.02)
        
        action = "HOLD"
        quantity = 0
        
        if self.agent_type == AgentType.TREND_FOLLOWER:
            if trend > 0.3:
                action = "BUY"
                quantity = self.capital * 0.1 * self.risk_tolerance / price
            elif trend < -0.3:
                action = "SELL"
                quantity = abs(self.position) * 0.5
        
        elif self.agent_type == AgentType.CONTRARIAN:
            if trend > 0.5:  # Aşırı iyimserlik
                action = "SELL"
                quantity = abs(self.position) * 0.3
            elif trend < -0.5:  # Aşırı kötümserlik
                action = "BUY"
                quantity = self.capital * 0.1 / price
        
        elif self.agent_type == AgentType.NOISE_TRADER:
            if random.random() > 0.6:
                action = random.choice(["BUY", "SELL"])
                quantity = self.capital * 0.05 / price
        
        elif self.agent_type == AgentType.FUNDAMENTAL:
            fair_value = market_state.get("fair_value", price)
            if price < fair_value * 0.9:  # %10 ucuz
                action = "BUY"
                quantity = self.capital * 0.2 / price
            elif price > fair_value * 1.1:  # %10 pahalı
                action = "SELL"
                quantity = abs(self.position) * 0.5
        
        elif self.agent_type == AgentType.NEWS_REACTIVE:
            if news_sentiment > 0.5:
                action = "BUY"
                quantity = self.capital * news_sentiment * 0.1 / price
            elif news_sentiment < -0.3:
                action = "SELL"
                quantity = abs(self.position) * abs(news_sentiment)
        
        elif self.agent_type == AgentType.INSTITUTIONAL:
            if random.random() > 0.9:  # Nadir ama büyük
                action = random.choice(["BUY", "SELL"])
                quantity = self.capital * 0.3 / price
        
        elif self.agent_type == AgentType.HFT:
            # Her turda küçük işlem
            action = "BUY" if trend > 0 else "SELL" if trend < 0 else "HOLD"
            quantity = self.capital * 0.01 / price
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "action": action,
            "quantity": quantity
        }
    
    def execute_trade(self, action: str, quantity: float, price: float):
        """İşlem gerçekleştir."""
        if action == "BUY":
            cost = quantity * price
            if cost <= self.capital:
                self.position += quantity
                self.capital -= cost
                self.trade_history.append({
                    "action": "BUY", "qty": quantity, "price": price
                })
        elif action == "SELL":
            if self.position >= quantity:
                self.position -= quantity
                self.capital += quantity * price
                self.trade_history.append({
                    "action": "SELL", "qty": quantity, "price": price
                })


class MarketSimulator:
    """
    Agent-Based Market Simulator.
    
    Dijital İkiz: Gerçek piyasayı simüle eden sanal borsa.
    """
    
    def __init__(self, 
                 initial_price: float = 100,
                 tick_size: float = 0.01):
        self.price = initial_price
        self.tick_size = tick_size
        self.price_history = [initial_price]
        self.agents = []
        self.order_book = {"bids": [], "asks": []}
        self.tick = 0
        self.volume_history = []
    
    def add_agents(self, agent_configs: List[Dict]):
        """Ajanları ekle."""
        for config in agent_configs:
            agent = MarketAgent(
                agent_id=config.get("id", f"agent_{len(self.agents)}"),
                agent_type=AgentType(config.get("type", "noise_trader")),
                capital=config.get("capital", 10000),
                risk_tolerance=config.get("risk_tolerance", 0.5)
            )
            self.agents.append(agent)
        
        print(f"{Fore.CYAN}🤖 {len(self.agents)} ajan eklendi{Style.RESET_ALL}", flush=True)
    
    def create_default_population(self, n_agents: int = 1000):
        """Varsayılan popülasyon oluştur."""
        configs = []
        
        # Tip dağılımı
        distribution = {
            AgentType.NOISE_TRADER: 0.40,
            AgentType.TREND_FOLLOWER: 0.20,
            AgentType.CONTRARIAN: 0.10,
            AgentType.FUNDAMENTAL: 0.10,
            AgentType.NEWS_REACTIVE: 0.10,
            AgentType.INSTITUTIONAL: 0.05,
            AgentType.HFT: 0.05,
        }
        
        for agent_type, ratio in distribution.items():
            count = int(n_agents * ratio)
            for i in range(count):
                configs.append({
                    "id": f"{agent_type.value}_{i}",
                    "type": agent_type.value,
                    "capital": np.random.uniform(5000, 50000),
                    "risk_tolerance": np.random.uniform(0.2, 0.8)
                })
        
        self.add_agents(configs)
    
    def simulate_tick(self, external_shock: Dict = None) -> Dict:
        """
        Bir tick simüle et.
        
        Args:
            external_shock: Dış etki (faiz artışı, haber vb.)
        """
        self.tick += 1
        
        # Piyasa durumu
        trend = self._calculate_trend()
        volatility = self._calculate_volatility()
        
        market_state = {
            "price": self.price,
            "trend": trend,
            "volatility": volatility,
            "news_sentiment": external_shock.get("sentiment", 0) if external_shock else 0,
            "fair_value": 100 + np.sin(self.tick / 50) * 10  # Döngüsel değer
        }
        
        # Dış şok uygula
        if external_shock:
            shock_type = external_shock.get("type", "")
            magnitude = external_shock.get("magnitude", 0)
            
            if shock_type == "RATE_HIKE":
                market_state["trend"] -= magnitude * 0.5
            elif shock_type == "CRISIS":
                market_state["trend"] -= magnitude
                market_state["volatility"] *= 2
            elif shock_type == "GOOD_NEWS":
                market_state["trend"] += magnitude * 0.3
        
        # Ajanların kararlarını topla
        buy_volume = 0
        sell_volume = 0
        
        for agent in self.agents:
            decision = agent.decide(market_state)
            
            if decision["action"] == "BUY":
                buy_volume += decision["quantity"]
                agent.execute_trade("BUY", decision["quantity"], self.price)
            elif decision["action"] == "SELL":
                sell_volume += decision["quantity"]
                agent.execute_trade("SELL", decision["quantity"], self.price)
        
        # Fiyat güncelleme (arz-talep)
        order_imbalance = (buy_volume - sell_volume) / max(1, buy_volume + sell_volume)
        price_impact = order_imbalance * volatility * self.price
        
        self.price += price_impact
        self.price = max(self.tick_size, self.price)  # Negatif olamaz
        
        self.price_history.append(self.price)
        self.volume_history.append(buy_volume + sell_volume)
        
        return {
            "tick": self.tick,
            "price": self.price,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "order_imbalance": order_imbalance,
            "trend": trend,
            "volatility": volatility
        }
    
    def _calculate_trend(self) -> float:
        """Trend hesapla (-1 ile 1 arası)."""
        if len(self.price_history) < 10:
            return 0
        
        recent = self.price_history[-10:]
        returns = np.diff(recent) / recent[:-1]
        return np.clip(np.mean(returns) * 50, -1, 1)
    
    def _calculate_volatility(self) -> float:
        """Volatilite hesapla."""
        if len(self.price_history) < 20:
            return 0.02
        
        recent = self.price_history[-20:]
        returns = np.diff(recent) / recent[:-1]
        return np.std(returns)
    
    def run_simulation(self, 
                      n_ticks: int = 1000,
                      shocks: List[Dict] = None) -> Dict:
        """
        Tam simülasyon çalıştır.
        
        Args:
            n_ticks: Tick sayısı
            shocks: {tick: shock_dict} format
        """
        print(f"{Fore.CYAN}🌍 Simülasyon başlıyor: {n_ticks} tick, {len(self.agents)} ajan{Style.RESET_ALL}", flush=True)
        
        shocks = shocks or {}
        results = []
        
        for t in range(n_ticks):
            shock = shocks.get(t)
            result = self.simulate_tick(shock)
            results.append(result)
            
            if t % 200 == 0:
                print(f"  Tick {t}: ${self.price:.2f}", flush=True)
        
        # Sonuç analizi
        prices = [r["price"] for r in results]
        returns = np.diff(prices) / prices[:-1]
        
        final_result = {
            "n_ticks": n_ticks,
            "n_agents": len(self.agents),
            "initial_price": self.price_history[0],
            "final_price": self.price,
            "total_return_pct": ((self.price - 100) / 100) * 100,
            "volatility": np.std(returns),
            "max_drawdown": self._calculate_max_drawdown(prices),
            "price_history": prices,
            "agent_pnl": self._calculate_agent_pnl()
        }
        
        print(f"{Fore.GREEN}✅ Simülasyon tamamlandı: ${self.price:.2f}{Style.RESET_ALL}", flush=True)
        
        return final_result
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Max drawdown hesapla."""
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_agent_pnl(self) -> Dict:
        """Ajan PnL dağılımı."""
        pnl_by_type = defaultdict(list)
        
        for agent in self.agents:
            total_value = agent.capital + agent.position * self.price
            pnl = total_value - 10000  # Başlangıç sermayesi 10k varsayımı
            pnl_by_type[agent.agent_type.value].append(pnl)
        
        return {
            agent_type: {
                "avg_pnl": np.mean(pnls),
                "best": max(pnls),
                "worst": min(pnls)
            }
            for agent_type, pnls in pnl_by_type.items()
        }
    
    def simulate_crisis(self, crisis_type: str = "2008_FINANCIAL") -> Dict:
        """Kriz senaryosu simüle et."""
        print(f"{Fore.RED}🔥 Kriz simülasyonu: {crisis_type}{Style.RESET_ALL}", flush=True)
        
        # Kriz şokları
        shocks = {}
        
        if crisis_type == "2008_FINANCIAL":
            # Yavaş başlayan, hızlanan çöküş
            for t in range(100, 200):
                shocks[t] = {"type": "CRISIS", "magnitude": 0.1 + (t - 100) * 0.01}
        
        elif crisis_type == "FLASH_CRASH":
            # Ani çöküş
            shocks[500] = {"type": "CRISIS", "magnitude": 2.0, "sentiment": -1.0}
        
        elif crisis_type == "RATE_HIKE":
            # Faiz artışı
            shocks[200] = {"type": "RATE_HIKE", "magnitude": 0.5}
            shocks[400] = {"type": "RATE_HIKE", "magnitude": 0.5}
        
        return self.run_simulation(1000, shocks)
    
    def generate_abm_report(self, result: Dict) -> str:
        """ABM raporu."""
        report = f"""
<agent_based_simulation>
🌍 DİJİTAL İKİZ SİMÜLASYONU
════════════════════════════════════════

📊 GENEL:
  • Tick: {result['n_ticks']}
  • Ajan: {result['n_agents']}
  • Başlangıç: ${result['initial_price']:.2f}
  • Son: ${result['final_price']:.2f}
  • Getiri: %{result['total_return_pct']:.1f}
  • Max DD: %{result['max_drawdown']*100:.1f}

🤖 AJAN PERFORMANSI:
"""
        for agent_type, stats in result.get("agent_pnl", {}).items():
            report += f"  • {agent_type}: ${stats['avg_pnl']:.0f} (avg)\n"
        
        report += "</agent_based_simulation>\n"
        return report
