"""
Tensor Networks and Quantum-Inspired Optimization
Author: Erdinc Erdogan
Purpose: Models financial data as tensor networks (MPS, Tensor Train) to discover hidden long-range correlations and applies quantum annealing simulation.
References:
- Tensor Networks (Orus, 2014)
- Matrix Product States (MPS)
- Quantum Annealing for Optimization
Usage:
    analyzer = TensorNetworkAnalyzer(bond_dimension=10)
    tensor = analyzer.build_price_tensor(prices)
    decomposition = analyzer.tensor_decomposition(tensor)
"""
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class TensorNetworkAnalyzer:
    """
    Tensor Network Pattern Recognition.
    
    Finansal veriyi çok boyutlu tensör ağları olarak modeller.
    Normalde görünmeyen uzun vadeli korelasyonları bulur.
    
    Yöntemler:
    - Matrix Product States (MPS)
    - Tensor Train Decomposition
    - Higher-Order SVD
    """
    
    def __init__(self, bond_dimension: int = 10):
        """
        Args:
            bond_dimension: Tensör ağı bağ boyutu (complexity tradeoff)
        """
        self.bond_dimension = bond_dimension
        self.correlation_cache = {}
    
    def build_price_tensor(self, price_series: List[float], window_size: int = 20) -> np.ndarray:
        """
        Fiyat serisinden tensör oluştur.
        
        Her pencere bir "durum vektörü" olur.
        """
        if len(price_series) < window_size:
            return np.array([])
        
        # Returns hesapla
        returns = np.diff(price_series) / price_series[:-1]
        
        # Pencere tensörleri
        n_windows = len(returns) - window_size + 1
        tensor = np.zeros((n_windows, window_size))
        
        for i in range(n_windows):
            tensor[i] = returns[i:i + window_size]
        
        return tensor
    
    def tensor_decomposition(self, tensor: np.ndarray) -> Dict:
        """
        Tensör ayrıştırma (SVD tabanlı).
        
        Ana bileşenleri ve gizli yapıları çıkarır.
        """
        if tensor.size == 0:
            return {"error": "Empty tensor"}
        
        try:
            # SVD
            U, S, Vt = np.linalg.svd(tensor, full_matrices=False)
            
            # Truncate to bond dimension
            rank = min(self.bond_dimension, len(S))
            
            U_truncated = U[:, :rank]
            S_truncated = S[:rank]
            Vt_truncated = Vt[:rank, :]
            
            # Explained variance
            total_variance = np.sum(S ** 2)
            explained_variance = np.sum(S_truncated ** 2) / total_variance
            
            return {
                "rank": rank,
                "singular_values": S_truncated.tolist(),
                "explained_variance": explained_variance,
                "principal_modes": Vt_truncated,
                "temporal_weights": U_truncated
            }
        except Exception as e:
            return {"error": str(e)}
    
    def find_hidden_correlations(self, 
                                multi_asset_data: Dict[str, List[float]],
                                lookback: int = 100) -> Dict:
        """
        Çoklu varlık arasındaki gizli korelasyonları bul.
        
        Args:
            multi_asset_data: {"AAPL": [prices], "MSFT": [prices], ...}
            lookback: Geriye bakış periyodu
        """
        print(f"{Fore.CYAN}🌌 Tensör korelasyon analizi...{Style.RESET_ALL}", flush=True)
        
        assets = list(multi_asset_data.keys())
        n_assets = len(assets)
        
        if n_assets < 2:
            return {"error": "En az 2 varlık gerekli"}
        
        # Her varlık için returns
        returns_matrix = []
        for asset in assets:
            prices = multi_asset_data[asset][-lookback:]
            returns = np.diff(prices) / np.array(prices[:-1])
            returns_matrix.append(returns)
        
        returns_tensor = np.array(returns_matrix).T  # (time, assets)
        
        # Standart korelasyon
        standard_corr = np.corrcoef(np.array(returns_matrix))
        
        # Tensör tabanlı "derin" korelasyon
        # Gecikmeli korelasyonlar (lead-lag relationships)
        lag_correlations = {}
        
        for lag in range(1, 6):  # 1-5 gün gecikme
            lagged_corr = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(n_assets):
                    if i != j:
                        r1 = returns_matrix[i][:-lag]
                        r2 = returns_matrix[j][lag:]
                        if len(r1) > 5:
                            lagged_corr[i, j] = np.corrcoef(r1, r2)[0, 1]
            lag_correlations[f"lag_{lag}"] = lagged_corr
        
        # En güçlü lead-lag ilişkileri
        strong_leads = []
        for lag_name, corr_matrix in lag_correlations.items():
            for i in range(n_assets):
                for j in range(n_assets):
                    if abs(corr_matrix[i, j]) > 0.3 and i != j:
                        strong_leads.append({
                            "leader": assets[i],
                            "follower": assets[j],
                            "lag": int(lag_name.split("_")[1]),
                            "correlation": corr_matrix[i, j]
                        })
        
        # En güçlülere göre sırala
        strong_leads.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "assets": assets,
            "standard_correlation": standard_corr.tolist(),
            "lag_correlations": {k: v.tolist() for k, v in lag_correlations.items()},
            "strong_lead_lag": strong_leads[:10],
            "trading_signals": self._generate_lead_lag_signals(strong_leads[:5])
        }
    
    def _generate_lead_lag_signals(self, leads: List[Dict]) -> List[Dict]:
        """Lead-lag ilişkilerinden sinyal üret."""
        signals = []
        for lead in leads:
            if lead["correlation"] > 0.3:
                signals.append({
                    "type": "FOLLOW_LEADER",
                    "watch": lead["leader"],
                    "trade": lead["follower"],
                    "direction": "SAME",
                    "lag_days": lead["lag"],
                    "confidence": min(abs(lead["correlation"]) * 1.5, 0.9)
                })
            elif lead["correlation"] < -0.3:
                signals.append({
                    "type": "CONTRARIAN",
                    "watch": lead["leader"],
                    "trade": lead["follower"],
                    "direction": "OPPOSITE",
                    "lag_days": lead["lag"],
                    "confidence": min(abs(lead["correlation"]) * 1.5, 0.9)
                })
        return signals


class QuantumAnnealingOptimizer:
    """
    Quantum Annealing Simulation.
    
    Portföy optimizasyonu için kuantum tavlama simülasyonu.
    Kombinatoryal optimizasyonu "enerji minimizasyonu" olarak çözer.
    
    Avantaj: Global optimum'a yakınsama (local minima'dan kaçış)
    """
    
    def __init__(self, 
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.995,
                 min_temperature: float = 0.01):
        """
        Args:
            initial_temperature: Başlangıç sıcaklığı
            cooling_rate: Soğuma oranı
            min_temperature: Minimum sıcaklık
        """
        self.T0 = initial_temperature
        self.cooling_rate = cooling_rate
        self.T_min = min_temperature
    
    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          risk_aversion: float = 1.0,
                          constraints: Dict = None) -> Dict:
        """
        Kuantum tavlama ile portföy optimizasyonu.
        
        Hedef: max(returns) - risk_aversion * variance
        """
        print(f"{Fore.CYAN}⚛️ Kuantum tavlama başlatılıyor...{Style.RESET_ALL}", flush=True)
        
        n_assets = len(expected_returns)
        
        # Başlangıç çözümü (eşit ağırlık)
        current_weights = np.ones(n_assets) / n_assets
        current_energy = self._calculate_energy(
            current_weights, expected_returns, covariance_matrix, risk_aversion
        )
        
        best_weights = current_weights.copy()
        best_energy = current_energy
        
        T = self.T0
        iteration = 0
        energy_history = []
        
        while T > self.T_min:
            iteration += 1
            
            # Komşu çözüm üret (kuantum tünelleme simülasyonu)
            neighbor_weights = self._quantum_tunneling_move(current_weights)
            
            # Enerji hesapla
            neighbor_energy = self._calculate_energy(
                neighbor_weights, expected_returns, covariance_matrix, risk_aversion
            )
            
            # Kabul kriterleri (Metropolis-Hastings)
            delta_E = neighbor_energy - current_energy
            
            if delta_E < 0:  # Daha iyi çözüm
                current_weights = neighbor_weights
                current_energy = neighbor_energy
            else:  # Daha kötü ama şansla kabul
                acceptance_prob = np.exp(-delta_E / T)
                if np.random.random() < acceptance_prob:
                    current_weights = neighbor_weights
                    current_energy = neighbor_energy
            
            # En iyiyi güncelle
            if current_energy < best_energy:
                best_weights = current_weights.copy()
                best_energy = current_energy
            
            # Sıcaklığı düşür
            T *= self.cooling_rate
            energy_history.append(current_energy)
        
        print(f"{Fore.GREEN}  → {iteration} iterasyon tamamlandı{Style.RESET_ALL}", flush=True)
        
        # Sonuç metrikleri
        portfolio_return = np.dot(best_weights, expected_returns)
        portfolio_variance = np.dot(best_weights.T, np.dot(covariance_matrix, best_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            "optimal_weights": best_weights.tolist(),
            "expected_return": portfolio_return,
            "volatility": portfolio_std,
            "sharpe_ratio": sharpe,
            "iterations": iteration,
            "final_energy": best_energy,
            "energy_improvement": energy_history[0] - best_energy if energy_history else 0
        }
    
    def _calculate_energy(self, weights: np.ndarray, 
                         returns: np.ndarray,
                         cov: np.ndarray,
                         risk_aversion: float) -> float:
        """
        Enerji fonksiyonu (minimize edilecek).
        
        E = -returns + risk_aversion * variance + penalty
        """
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov, weights))
        
        # Constraint penalty (ağırlıklar 0-1 arası ve toplam 1)
        sum_penalty = (np.sum(weights) - 1) ** 2 * 100
        bound_penalty = np.sum(np.maximum(0, -weights) ** 2 + np.maximum(0, weights - 1) ** 2) * 100
        
        energy = -portfolio_return + risk_aversion * portfolio_variance + sum_penalty + bound_penalty
        
        return energy
    
    def _quantum_tunneling_move(self, weights: np.ndarray) -> np.ndarray:
        """
        Kuantum tünelleme hareketi simülasyonu.
        
        Klasik tavlamadan farklı olarak, enerji bariyerlerini
        "tünelleyerek" aşabilir.
        """
        new_weights = weights.copy()
        n = len(weights)
        
        # Rastgele iki varlık seç ve ağırlık transfer et
        i, j = np.random.choice(n, 2, replace=False)
        
        # Transfer miktarı (kuantum süperpozisyon simülasyonu)
        transfer = np.random.normal(0, 0.05)
        
        new_weights[i] += transfer
        new_weights[j] -= transfer
        
        # Normalize
        new_weights = np.maximum(0, new_weights)
        if np.sum(new_weights) > 0:
            new_weights /= np.sum(new_weights)
        else:
            new_weights = np.ones(n) / n
        
        return new_weights
    
    def solve_asset_selection(self, 
                             candidates: List[str],
                             scores: np.ndarray,
                             max_assets: int = 10) -> Dict:
        """
        Varlık seçimi problemi (QUBO formülasyonu).
        
        N varlık içinden en iyi K tanesini seç.
        """
        n = len(candidates)
        
        # Binary değişkenler (0 veya 1)
        x = np.zeros(n)
        
        # Greedy başlangıç
        top_indices = np.argsort(scores)[-max_assets:]
        x[top_indices] = 1
        
        # Simulated annealing ile iyileştir
        T = self.T0
        current_score = np.sum(scores * x)
        
        while T > self.T_min:
            # Flip bir bit
            flip_idx = np.random.randint(n)
            x_new = x.copy()
            x_new[flip_idx] = 1 - x_new[flip_idx]
            
            # Constraint: max_assets
            if np.sum(x_new) > max_assets:
                # Random deselect one
                selected = np.where(x_new == 1)[0]
                deselect = np.random.choice(selected)
                x_new[deselect] = 0
            
            new_score = np.sum(scores * x_new)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                x = x_new
                current_score = new_score
            
            T *= self.cooling_rate
        
        selected_assets = [candidates[i] for i in range(n) if x[i] == 1]
        
        return {
            "selected_assets": selected_assets,
            "total_score": current_score,
            "selection_binary": x.tolist()
        }
    
    def generate_quantum_report(self, result: Dict) -> str:
        """Kuantum optimizasyon raporu."""
        report = f"""
<quantum_annealing>
⚛️ KUANTUM TAVLAMA OPTİMİZASYONU
════════════════════════════════════════

📊 SONUÇ:
  • Beklenen Getiri: %{result.get('expected_return', 0)*100:.2f}
  • Volatilite: %{result.get('volatility', 0)*100:.2f}
  • Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}

⚙️ SÜREÇ:
  • İterasyon: {result.get('iterations', 0)}
  • Enerji İyileşmesi: {result.get('energy_improvement', 0):.4f}

📈 OPTİMAL AĞIRLIKLAR:
"""
        weights = result.get('optimal_weights', [])
        for i, w in enumerate(weights[:10]):
            bar = "█" * int(w * 20)
            report += f"  Varlık {i+1}: {bar} ({w*100:.1f}%)\n"
        
        report += "</quantum_annealing>\n"
        return report
