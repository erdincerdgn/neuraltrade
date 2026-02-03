"""
Chaos Theory and Fractal Analysis - Hurst and Lyapunov Exponents
Author: Erdinc Erdogan
Purpose: Calculates Hurst exponent and Lyapunov exponent to measure market trending behavior
and detect chaotic dynamics for adaptive strategy selection.
References:
- Hurst (1951) "Long-term Storage Capacity of Reservoirs"
- Rosenstein et al. (1993) "A Practical Method for Calculating Largest Lyapunov Exponents"
- Mandelbrot (1982) "The Fractal Geometry of Nature"
Usage:
    hurst_calc = HurstExponentCalculator()
    result = hurst_calc.calculate(returns)
    if result.hurst > 0.5: apply_momentum_strategy()
"""
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class HurstExponentCalculator:
    """
    Hurst Exponent Calculator.
    
    Piyasanın trend mi yoksa mean-reverting mi olduğunu ölçer.
    
    H > 0.5: Trending (momentum stratejisi uygundur)
    H = 0.5: Random walk (tahmin edilemez)
    H < 0.5: Mean-reverting (ortalamaya dönüş stratejisi)
    """
    
    def __init__(self):
        self.history = []
    
    def calculate_rescaled_range(self, series: np.ndarray) -> Tuple[float, float]:
        """
        R/S (Rescaled Range) analizi ile Hurst hesapla.
        """
        n = len(series)
        if n < 20:
            return 0.5, 0  # Yetersiz veri
        
        # Mean-centered cumulative deviations
        mean_val = np.mean(series)
        deviations = series - mean_val
        cumulative = np.cumsum(deviations)
        
        # Range
        R = np.max(cumulative) - np.min(cumulative)
        
        # Standard deviation
        S = np.std(series, ddof=1)
        
        if S == 0:
            return 0.5, 0
        
        RS = R / S
        
        # Hurst = log(R/S) / log(n)
        # Daha doğru: çoklu pencere boyutlarıyla regresyon
        
        return RS, n
    
    def calculate_hurst(self, price_series: List[float], min_window: int = 10) -> Dict:
        """
        Hurst Exponent hesapla.
        
        Args:
            price_series: Fiyat serisi
            min_window: Minimum pencere boyutu
        """
        print(f"{Fore.CYAN}📊 Hurst Exponent hesaplanıyor...{Style.RESET_ALL}", flush=True)
        
        # Returns hesapla
        prices = np.array(price_series)
        returns = np.diff(np.log(prices))
        
        n = len(returns)
        if n < min_window * 2:
            return {"error": "Yetersiz veri", "hurst": 0.5}
        
        # Farklı pencere boyutları
        window_sizes = []
        rs_values = []
        
        for window in range(min_window, n // 2):
            rs_list = []
            
            for start in range(0, n - window, window):
                segment = returns[start:start + window]
                rs, _ = self.calculate_rescaled_range(segment)
                if rs > 0:
                    rs_list.append(rs)
            
            if rs_list:
                window_sizes.append(window)
                rs_values.append(np.mean(rs_list))
        
        if len(window_sizes) < 3:
            return {"error": "Yetersiz hesaplama", "hurst": 0.5}
        
        # Log-log regresyon
        log_n = np.log(window_sizes)
        log_rs = np.log(rs_values)
        
        # Linear regression
        slope, intercept = np.polyfit(log_n, log_rs, 1)
        
        hurst = slope
        
        # Yorumlama
        if hurst > 0.6:
            regime = "TRENDING"
            strategy = "MOMENTUM"
            confidence = min((hurst - 0.5) * 2, 0.9)
        elif hurst < 0.4:
            regime = "MEAN_REVERTING"
            strategy = "MEAN_REVERSION"
            confidence = min((0.5 - hurst) * 2, 0.9)
        else:
            regime = "RANDOM_WALK"
            strategy = "NONE"
            confidence = 0.3
        
        result = {
            "hurst_exponent": hurst,
            "regime": regime,
            "recommended_strategy": strategy,
            "confidence": confidence,
            "interpretation": self._interpret_hurst(hurst),
            "data_points": n
        }
        
        self.history.append(result)
        
        print(f"{Fore.GREEN}  → Hurst = {hurst:.3f} ({regime}){Style.RESET_ALL}", flush=True)
        
        return result
    
    def _interpret_hurst(self, h: float) -> str:
        """Hurst yorumu."""
        if h > 0.7:
            return "Güçlü trend - momentum stratejileri idealdir"
        elif h > 0.55:
            return "Hafif trend eğilimi - dikkatli momentum"
        elif h > 0.45:
            return "Random walk yakın - tahmin zor"
        elif h > 0.3:
            return "Hafif mean-reversion - range trading uygundur"
        else:
            return "Güçlü mean-reversion - counter-trend idealdir"


class LyapunovExponentCalculator:
    """
    Lyapunov Exponent Calculator.
    
    Sistemin kaosa ne kadar yakın olduğunu ölçer.
    
    λ > 0: Kaotik (Kelebek etkisi, küçük değişiklikler büyük sonuçlar)
    λ ≈ 0: Kritik nokta (kriz öncesi)
    λ < 0: Stabil (tahmin edilebilir)
    """
    
    def __init__(self, embedding_dim: int = 3, delay: int = 1):
        """
        Args:
            embedding_dim: Phase space boyutu
            delay: Zaman gecikmesi
        """
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.history = []
    
    def embed_time_series(self, series: np.ndarray) -> np.ndarray:
        """
        Takens teoremi ile zaman serisini phase space'e embed et.
        """
        n = len(series) - (self.embedding_dim - 1) * self.delay
        
        if n <= 0:
            return np.array([])
        
        embedded = np.zeros((n, self.embedding_dim))
        
        for i in range(n):
            for j in range(self.embedding_dim):
                embedded[i, j] = series[i + j * self.delay]
        
        return embedded
    
    def calculate_lyapunov(self, price_series: List[float], 
                          epsilon: float = 0.01,
                          max_iterations: int = 100) -> Dict:
        """
        Lyapunov Exponent hesapla.
        
        Rosenstein algoritması (basitleştirilmiş).
        """
        print(f"{Fore.CYAN}📈 Lyapunov Exponent hesaplanıyor...{Style.RESET_ALL}", flush=True)
        
        # Returns kullan
        prices = np.array(price_series)
        returns = np.diff(np.log(prices))
        
        # Normalize
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        
        # Phase space embedding
        embedded = self.embed_time_series(returns)
        
        if len(embedded) < 50:
            return {"error": "Yetersiz veri", "lyapunov": 0}
        
        n = len(embedded)
        divergences = []
        
        # Her nokta için en yakın komşuyu bul ve ayrışmayı takip et
        for i in range(n - max_iterations):
            # En yakın komşuyu bul (temporal separation ile)
            min_dist = float('inf')
            nearest_idx = -1
            
            for j in range(n):
                if abs(i - j) > 10:  # Temporal separation
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    if dist < min_dist and dist > epsilon:
                        min_dist = dist
                        nearest_idx = j
            
            if nearest_idx == -1:
                continue
            
            # Trajektorilerin ayrışmasını izle
            for k in range(1, min(max_iterations, n - max(i, nearest_idx))):
                dist_k = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                if dist_k > 0 and min_dist > 0:
                    divergence = np.log(dist_k / min_dist) / k
                    divergences.append(divergence)
        
        if not divergences:
            return {"error": "Hesaplama başarısız", "lyapunov": 0}
        
        lyapunov = np.mean(divergences)
        
        # Yorumlama
        if lyapunov > 0.1:
            regime = "CHAOTIC"
            risk_level = "HIGH"
        elif lyapunov > 0:
            regime = "EDGE_OF_CHAOS"
            risk_level = "ELEVATED"
        elif lyapunov > -0.1:
            regime = "STABLE"
            risk_level = "NORMAL"
        else:
            regime = "HIGHLY_STABLE"
            risk_level = "LOW"
        
        result = {
            "lyapunov_exponent": lyapunov,
            "regime": regime,
            "risk_level": risk_level,
            "predictability": "LOW" if lyapunov > 0 else "MEDIUM" if lyapunov > -0.1 else "HIGH",
            "interpretation": self._interpret_lyapunov(lyapunov),
            "early_warning": lyapunov > 0.05  # Kriz erken uyarısı
        }
        
        if result["early_warning"]:
            print(f"{Fore.RED}  ⚠️ KRİZ ERKEN UYARISI: Kaotik davranış tespit edildi!{Style.RESET_ALL}", flush=True)
        
        self.history.append(result)
        
        print(f"{Fore.GREEN}  → λ = {lyapunov:.4f} ({regime}){Style.RESET_ALL}", flush=True)
        
        return result
    
    def _interpret_lyapunov(self, lyap: float) -> str:
        """Lyapunov yorumu."""
        if lyap > 0.1:
            return "Yüksek kaos - tahmin neredeyse imkansız, hedge pozisyonları artır"
        elif lyap > 0.05:
            return "Kaosa yaklaşılıyor - kriz öncesi sinyal olabilir"
        elif lyap > 0:
            return "Hafif instabilite - dikkatli ol"
        elif lyap > -0.1:
            return "Stabil sistem - normal trading koşulları"
        else:
            return "Çok stabil - güçlü mean-reversion beklenir"


class FractalAnalyzer:
    """
    Birleşik Fractal Analyzer.
    
    Hurst + Lyapunov birlikte piyasa durumunu belirler.
    """
    
    def __init__(self):
        self.hurst_calc = HurstExponentCalculator()
        self.lyapunov_calc = LyapunovExponentCalculator()
    
    def full_analysis(self, price_series: List[float]) -> Dict:
        """Tam fractal analizi."""
        hurst = self.hurst_calc.calculate_hurst(price_series)
        lyapunov = self.lyapunov_calc.calculate_lyapunov(price_series)
        
        # Birleşik karar
        H = hurst.get("hurst_exponent", 0.5)
        L = lyapunov.get("lyapunov_exponent", 0)
        
        # Decision matrix
        if L > 0.05:  # Kaotik
            recommendation = "REDUCE_EXPOSURE"
            reason = "Kaotik piyasa koşulları"
        elif H > 0.6 and L <= 0:  # Trending + stabil
            recommendation = "MOMENTUM_STRATEGY"
            reason = "Güçlü trend + stabil koşullar"
        elif H < 0.4 and L <= 0:  # Mean-reverting + stabil
            recommendation = "MEAN_REVERSION_STRATEGY"
            reason = "Mean-reversion + stabil koşullar"
        else:
            recommendation = "NEUTRAL"
            reason = "Belirsiz koşullar"
        
        return {
            "hurst_analysis": hurst,
            "lyapunov_analysis": lyapunov,
            "recommendation": recommendation,
            "reason": reason,
            "market_state": {
                "trending": H > 0.55,
                "chaotic": L > 0,
                "crisis_warning": lyapunov.get("early_warning", False)
            }
        }
    
    def generate_chaos_report(self, analysis: Dict) -> str:
        """Kaos teorisi raporu."""
        hurst = analysis.get("hurst_analysis", {})
        lyap = analysis.get("lyapunov_analysis", {})
        
        report = f"""
<chaos_theory>
🌀 KAOS TEORİSİ ANALİZİ
════════════════════════════════════════

📈 HURST EXPONENT: {hurst.get('hurst_exponent', 'N/A'):.3f}
  • Rejim: {hurst.get('regime', 'N/A')}
  • Strateji: {hurst.get('recommended_strategy', 'N/A')}

🦋 LYAPUNOV EXPONENT: {lyap.get('lyapunov_exponent', 'N/A'):.4f}
  • Rejim: {lyap.get('regime', 'N/A')}
  • Risk: {lyap.get('risk_level', 'N/A')}

⚠️ KRİZ UYARISI: {'🔴 AKTİF' if lyap.get('early_warning') else '🟢 YOK'}

💡 ÖNERİ: {analysis.get('recommendation', 'N/A')}
  {analysis.get('reason', '')}

</chaos_theory>
"""
        return report
