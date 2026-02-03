"""
Gamma Exposure (GEX) Engine
Author: Erdinc Erdogan
Purpose: Calculates net gamma exposure from options open interest to predict market maker hedging behavior and price stabilization/amplification zones.
References:
- Black-Scholes Gamma Calculation
- Dealer Gamma Exposure (GEX)
- Options Market Maker Hedging
Usage:
    gex = GammaExposureEngine(spot_price=450.0)
    gex.add_option(strike=455, expiry_days=30, option_type="CALL", open_interest=5000)
    result = gex.calculate_total_gex()
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from colorama import Fore, Style


class GammaExposureEngine:
    """
    Gamma Exposure (GEX) Engine.
    
    Opsiyon piyasasındaki gamma riskini hesaplar.
    Market Maker'ların hedge davranışını tahmin eder.
    
    Gamma > 0: Price stabilization
    Gamma < 0: Price amplification (squeeze/crash)
    """
    
    def __init__(self, spot_price: float = 100):
        self.spot_price = spot_price
        self.option_chain = []
        self.total_gex = 0
        self.gex_by_strike = {}
        self.flip_levels = []
    
    def add_option(self,
                  strike: float,
                  expiry_days: int,
                  option_type: str,  # CALL or PUT
                  open_interest: int,
                  implied_vol: float = 0.3):
        """
        Opsiyon ekle.
        
        Args:
            strike: Kullanım fiyatı
            expiry_days: Vadeye kalan gün
            option_type: CALL veya PUT
            open_interest: Açık pozisyon sayısı
            implied_vol: Implied volatility
        """
        option = {
            "strike": strike,
            "expiry_days": expiry_days,
            "type": option_type,
            "oi": open_interest,
            "iv": implied_vol,
            "gamma": self._calculate_gamma(strike, expiry_days, implied_vol)
        }
        
        self.option_chain.append(option)
    
    def _calculate_gamma(self, strike: float, expiry_days: int, iv: float) -> float:
        """
        Black-Scholes Gamma hesapla (basitleştirilmiş).
        """
        from math import exp, sqrt, pi
        
        S = self.spot_price
        K = strike
        T = expiry_days / 365
        sigma = iv
        r = 0.05  # Risk-free rate
        
        if T <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
        
        gamma = exp(-d1**2 / 2) / (S * sigma * sqrt(2 * pi * T))
        
        return gamma
    
    def calculate_total_gex(self) -> Dict:
        """
        Toplam Gamma Exposure hesapla.
        
        GEX = Σ (Gamma × Open Interest × 100 × Spot²)
        """
        print(f"{Fore.CYAN}📊 GEX hesaplanıyor...{Style.RESET_ALL}", flush=True)
        
        self.gex_by_strike = {}
        call_gex = 0
        put_gex = 0
        
        for option in self.option_chain:
            strike = option["strike"]
            gamma = option["gamma"]
            oi = option["oi"]
            
            # GEX = gamma × OI × 100 × S²
            gex = gamma * oi * 100 * (self.spot_price ** 2) / 1e9  # Billions
            
            # Dealer perspective: Short calls = positive gamma, short puts = negative gamma
            if option["type"] == "CALL":
                gex = -gex  # Dealers are typically short calls
                call_gex += gex
            else:
                gex = gex   # Dealers are typically short puts
                put_gex += gex
            
            if strike not in self.gex_by_strike:
                self.gex_by_strike[strike] = 0
            self.gex_by_strike[strike] += gex
        
        self.total_gex = call_gex + put_gex
        
        # Gamma flip seviyesi bul (pozitiften negatife geçiş)
        self._find_flip_levels()
        
        return {
            "total_gex_bn": self.total_gex,
            "call_gex_bn": call_gex,
            "put_gex_bn": put_gex,
            "gex_by_strike": self.gex_by_strike,
            "gamma_flip_levels": self.flip_levels,
            "market_regime": self._determine_regime()
        }
    
    def _find_flip_levels(self):
        """Gamma flip seviyelerini bul."""
        self.flip_levels = []
        
        sorted_strikes = sorted(self.gex_by_strike.keys())
        
        for i in range(1, len(sorted_strikes)):
            prev_strike = sorted_strikes[i-1]
            curr_strike = sorted_strikes[i]
            
            prev_gex = self.gex_by_strike[prev_strike]
            curr_gex = self.gex_by_strike[curr_strike]
            
            # İşaret değişimi
            if prev_gex * curr_gex < 0:
                flip_level = (prev_strike + curr_strike) / 2
                self.flip_levels.append({
                    "level": flip_level,
                    "from": "positive" if prev_gex > 0 else "negative",
                    "to": "positive" if curr_gex > 0 else "negative"
                })
    
    def _determine_regime(self) -> str:
        """Gamma rejimine göre piyasa davranışı."""
        if self.total_gex > 0.5:
            return "PINNING"  # Fiyat sabitlenme eğilimi
        elif self.total_gex < -0.5:
            return "AMPLIFICATION"  # Hareketler abartılacak
        else:
            return "NEUTRAL"
    
    def predict_price_magnets(self) -> List[Dict]:
        """
        Fiyatın çekileceği "mıknatıs" seviyeleri.
        
        En yüksek pozitif GEX = fiyat mıknatısı (max pain)
        """
        magnets = []
        
        for strike, gex in sorted(self.gex_by_strike.items(), key=lambda x: -x[1]):
            if gex > 0:
                strength = gex / max(abs(self.total_gex), 0.01)
                magnets.append({
                    "price": strike,
                    "gex_bn": gex,
                    "strength": min(strength, 1.0),
                    "type": "MAGNET"
                })
        
        return magnets[:5]  # Top 5
    
    def predict_walls(self) -> List[Dict]:
        """
        Fiyatın geçemeyeceği "duvar" seviyeleri.
        
        En düşük (negatif) GEX = momentum amplification zone
        """
        walls = []
        
        for strike, gex in sorted(self.gex_by_strike.items(), key=lambda x: x[1]):
            if gex < 0:
                walls.append({
                    "price": strike,
                    "gex_bn": gex,
                    "type": "WALL",
                    "behavior": "Volatilite artar, breakout potansiyeli"
                })
        
        return walls[:5]
    
    def generate_option_flow_signal(self) -> Dict:
        """Trading sinyali üret."""
        gex_result = self.calculate_total_gex()
        magnets = self.predict_price_magnets()
        walls = self.predict_walls()
        
        # Sinyal
        if self.spot_price < self.flip_levels[0]["level"] if self.flip_levels else 0:
            signal = "BULLISH"  # Gamma flip altında = yukarı baskı
        elif gex_result["market_regime"] == "PINNING":
            signal = "RANGE_BOUND"
        elif gex_result["market_regime"] == "AMPLIFICATION":
            signal = "VOLATILE"
        else:
            signal = "NEUTRAL"
        
        return {
            "signal": signal,
            "gex": gex_result,
            "magnets": magnets,
            "walls": walls,
            "flip_levels": self.flip_levels
        }
    
    def generate_gex_report(self) -> str:
        """GEX raporu."""
        gex = self.calculate_total_gex()
        magnets = self.predict_price_magnets()
        
        report = f"""
<gamma_exposure>
📊 GAMMA EXPOSURE ANALİZİ
════════════════════════════════════════

💹 SPOT: ${self.spot_price:.2f}

🔢 GEX TOPLAM: ${gex['total_gex_bn']:.2f}B
  • Call GEX: ${gex['call_gex_bn']:.2f}B
  • Put GEX: ${gex['put_gex_bn']:.2f}B

📈 REJİM: {gex['market_regime']}

🧲 MIKNATIS SEVİYELER:
"""
        for m in magnets[:3]:
            report += f"  • ${m['price']:.0f} (Güç: {m['strength']:.0%})\n"
        
        report += f"""
🔀 GAMMA FLİP:
"""
        for flip in self.flip_levels[:2]:
            report += f"  • ${flip['level']:.0f}: {flip['from']} → {flip['to']}\n"
        
        report += "</gamma_exposure>\n"
        return report
