"""
Time-Series GAN for Synthetic Crisis Scenarios
Author: Erdinc Erdogan
Purpose: Generates synthetic market data using GANs to create never-before-seen crisis scenarios for stress testing and augmented backtesting.
References:
- Generative Adversarial Networks (Goodfellow et al., 2014)
- TimeGAN (Yoon et al., 2019)
- Synthetic Financial Data Generation
Usage:
    gan = TimeSeriesGAN(latent_dim=100, sequence_length=50)
    gan.fit(real_prices, epochs=100)
    synthetic = gan.generate(n_samples=1000)
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class TimeSeriesGAN:
    """
    Time-Series GAN (Generative Adversarial Network).
    
    Gerçek piyasa verisine benzeyen sentetik veri üretir.
    
    Kullanım:
    - Backtest için daha fazla veri
    - Hiç yaşanmamış kriz senaryoları
    - Model stres testi
    """
    
    def __init__(self, latent_dim: int = 100, sequence_length: int = 50):
        """
        Args:
            latent_dim: Gürültü vektörü boyutu
            sequence_length: Üretilecek seri uzunluğu
        """
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Basitleştirilmiş "öğrenilmiş" parametreler
        self.learned_mean = 0.0005  # Ortalama günlük getiri
        self.learned_std = 0.02    # Günlük volatilite
        self.learned_autocorr = 0.1  # Otokorelasyon
        
        self.is_trained = False
    
    def fit(self, real_data: List[float], epochs: int = 100):
        """
        GAN eğitimi (basitleştirilmiş - gerçek implementasyon PyTorch gerektirir).
        
        Args:
            real_data: Gerçek fiyat serisi
            epochs: Eğitim döngüsü
        """
        print(f"{Fore.CYAN}🧪 Time-Series GAN eğitimi...{Style.RESET_ALL}", flush=True)
        
        # Gerçek veriden istatistikleri öğren
        returns = np.diff(np.log(real_data))
        
        if len(returns) > 10:
            self.learned_mean = np.mean(returns)
            self.learned_std = np.std(returns)
            
            # Otokorelasyon
            if len(returns) > 20:
                self.learned_autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        # Fat tails (kurtosis)
        self.kurtosis = 3.0 + np.clip((np.mean((returns - self.learned_mean)**4) / self.learned_std**4) - 3, 0, 10)
        
        # Volatility clustering (GARCH-like)
        squared_returns = returns ** 2
        self.vol_cluster = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] if len(squared_returns) > 2 else 0.3
        
        self.is_trained = True
        
        print(f"{Fore.GREEN}✅ GAN eğitildi: μ={self.learned_mean:.5f}, σ={self.learned_std:.4f}{Style.RESET_ALL}", flush=True)
    
    def generate(self, n_samples: int = 1, length: int = None) -> List[np.ndarray]:
        """
        Sentetik fiyat serisi üret.
        
        Args:
            n_samples: Kaç seri üretilecek
            length: Seri uzunluğu
        """
        length = length or self.sequence_length
        samples = []
        
        for _ in range(n_samples):
            # Latent noise
            z = np.random.randn(length)
            
            # Generate returns with learned characteristics
            returns = np.zeros(length)
            volatility = self.learned_std
            
            for t in range(length):
                # GARCH-like volatility clustering
                if t > 0:
                    volatility = np.sqrt(
                        0.1 * self.learned_std**2 +
                        0.7 * volatility**2 +
                        0.2 * returns[t-1]**2
                    )
                
                # Fat tails via Student-t distribution
                if self.kurtosis > 3.5:
                    # Student-t benzeri
                    df = 6  # degrees of freedom
                    noise = np.random.standard_t(df) / np.sqrt(df / (df - 2))
                else:
                    noise = np.random.randn()
                
                # Autoregressive component
                ar_component = self.learned_autocorr * returns[t-1] if t > 0 else 0
                
                returns[t] = self.learned_mean + ar_component + volatility * noise
            
            # Convert returns to prices (starting at 100)
            prices = 100 * np.exp(np.cumsum(returns))
            samples.append(prices)
        
        return samples
    
    def generate_stress_scenario(self, scenario_type: str = "crash", intensity: float = 1.0) -> np.ndarray:
        """
        Stres senaryosu üret.
        
        Args:
            scenario_type: crash, rally, flash_crash, v_recovery
            intensity: Yoğunluk (0.5-2.0)
        """
        length = 100  # 100 günlük senaryo
        
        scenarios = {
            "crash": self._generate_crash(length, intensity),
            "rally": self._generate_rally(length, intensity),
            "flash_crash": self._generate_flash_crash(length, intensity),
            "v_recovery": self._generate_v_recovery(length, intensity),
            "sideways_chop": self._generate_chop(length, intensity),
            "black_swan": self._generate_black_swan(length, intensity)
        }
        
        return scenarios.get(scenario_type, self._generate_crash(length, intensity))
    
    def _generate_crash(self, length: int, intensity: float) -> np.ndarray:
        """Crash senaryosu: Yavaş düşüş sonra hızlanan satış."""
        returns = np.zeros(length)
        
        # Initial slow decline
        for t in range(length // 3):
            returns[t] = -0.005 * intensity + np.random.randn() * 0.02
        
        # Accelerating decline
        for t in range(length // 3, 2 * length // 3):
            returns[t] = -0.02 * intensity + np.random.randn() * 0.04
        
        # Capitulation
        for t in range(2 * length // 3, length):
            returns[t] = -0.005 * intensity + np.random.randn() * 0.03
        
        prices = 100 * np.exp(np.cumsum(returns))
        return prices
    
    def _generate_rally(self, length: int, intensity: float) -> np.ndarray:
        """Rally senaryosu."""
        returns = np.zeros(length)
        
        for t in range(length):
            momentum = 0.003 * intensity * (1 + t / length)  # Artan momentum
            returns[t] = momentum + np.random.randn() * 0.015
        
        return 100 * np.exp(np.cumsum(returns))
    
    def _generate_flash_crash(self, length: int, intensity: float) -> np.ndarray:
        """Flash crash: Aniden düşüş, hızlı toparlanma."""
        returns = np.zeros(length)
        crash_point = length // 2
        
        # Normal
        for t in range(crash_point - 5):
            returns[t] = 0.0005 + np.random.randn() * 0.01
        
        # Flash crash (5 gün)
        crash_magnitude = 0.15 * intensity
        returns[crash_point-5:crash_point] = -crash_magnitude / 5 + np.random.randn(5) * 0.02
        
        # Recovery
        for t in range(crash_point, length):
            recovery_rate = crash_magnitude / (length - crash_point) * 0.8
            returns[t] = recovery_rate + np.random.randn() * 0.01
        
        return 100 * np.exp(np.cumsum(returns))
    
    def _generate_v_recovery(self, length: int, intensity: float) -> np.ndarray:
        """V-şekilli toparlanma."""
        returns = np.zeros(length)
        bottom = length // 2
        
        # Down
        for t in range(bottom):
            returns[t] = -0.01 * intensity + np.random.randn() * 0.02
        
        # Up
        for t in range(bottom, length):
            returns[t] = 0.012 * intensity + np.random.randn() * 0.02
        
        return 100 * np.exp(np.cumsum(returns))
    
    def _generate_chop(self, length: int, intensity: float) -> np.ndarray:
        """Yatay piyasa (chop)."""
        returns = np.zeros(length)
        
        for t in range(length):
            # Mean reverting
            if t > 10:
                cum_return = np.sum(returns[:t])
                mean_reversion = -0.1 * cum_return
                returns[t] = mean_reversion + np.random.randn() * 0.015 * intensity
            else:
                returns[t] = np.random.randn() * 0.015 * intensity
        
        return 100 * np.exp(np.cumsum(returns))
    
    def _generate_black_swan(self, length: int, intensity: float) -> np.ndarray:
        """Black Swan: Beklenmedik felaket."""
        returns = np.zeros(length)
        
        # Normal market
        for t in range(length):
            returns[t] = 0.0003 + np.random.randn() * 0.01
        
        # Black swan event (single day -30% to -50%)
        swan_day = np.random.randint(length // 3, 2 * length // 3)
        returns[swan_day] = -0.30 * intensity - np.random.rand() * 0.20
        
        return 100 * np.exp(np.cumsum(returns))


class StressTestEngine:
    """
    Stress Test Engine.
    
    Stratejiyi sentetik kriz senaryolarına karşı test eder.
    """
    
    def __init__(self):
        self.gan = TimeSeriesGAN()
        self.test_results = []
    
    def train_on_real_data(self, prices: List[float]):
        """GAN'ı gerçek veriyle eğit."""
        self.gan.fit(prices)
    
    def run_stress_test(self, 
                       strategy_func,
                       scenarios: List[str] = None,
                       iterations: int = 100) -> Dict:
        """
        Stres testi çalıştır.
        
        Args:
            strategy_func: (prices) -> {pnl, trades}
            scenarios: Test edilecek senaryolar
            iterations: Monte Carlo iterasyonu
        """
        print(f"{Fore.CYAN}🧪 Stres testi başlıyor: {iterations} iterasyon{Style.RESET_ALL}", flush=True)
        
        scenarios = scenarios or ["crash", "flash_crash", "black_swan", "sideways_chop"]
        
        results = {}
        
        for scenario in scenarios:
            scenario_results = []
            
            for i in range(iterations):
                # Sentetik senaryo üret
                intensity = np.random.uniform(0.5, 2.0)
                prices = self.gan.generate_stress_scenario(scenario, intensity)
                
                # Strateji test et
                try:
                    result = strategy_func(prices)
                    scenario_results.append({
                        "pnl": result.get("pnl", 0),
                        "max_drawdown": result.get("max_drawdown", 0),
                        "intensity": intensity
                    })
                except:
                    scenario_results.append({"pnl": -1, "max_drawdown": 1, "intensity": intensity})
            
            # Aggregate results
            pnls = [r["pnl"] for r in scenario_results]
            drawdowns = [r["max_drawdown"] for r in scenario_results]
            
            results[scenario] = {
                "avg_pnl": np.mean(pnls),
                "worst_pnl": np.min(pnls),
                "best_pnl": np.max(pnls),
                "avg_drawdown": np.mean(drawdowns),
                "worst_drawdown": np.max(drawdowns),
                "survival_rate": sum(1 for p in pnls if p > -0.5) / len(pnls)
            }
        
        self.test_results.append({
            "timestamp": datetime.now().isoformat(),
            "results": results
        })
        
        return results
    
    def generate_stress_report(self, results: Dict) -> str:
        """Stres testi raporu."""
        report = f"""
<stress_test>
🧪 STRES TESTİ RAPORU
════════════════════════════════════════

"""
        for scenario, stats in results.items():
            survival_color = Fore.GREEN if stats["survival_rate"] > 0.8 else Fore.YELLOW if stats["survival_rate"] > 0.5 else Fore.RED
            
            report += f"""📉 {scenario.upper()}:
  • Ort. PnL: {stats['avg_pnl']*100:.1f}%
  • En Kötü: {stats['worst_pnl']*100:.1f}%
  • Max DD: {stats['worst_drawdown']*100:.1f}%
  • Hayatta Kalma: {survival_color}%{stats['survival_rate']*100:.0f}{Style.RESET_ALL}

"""
        
        report += "</stress_test>\n"
        return report
