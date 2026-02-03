"""
Topological Data Analysis for Crisis Detection
Author: Erdinc Erdogan
Purpose: Applies persistent homology to detect market crashes as topological shape distortions using Betti numbers and Takens embedding.
References:
- Persistent Homology (Edelsbrunner et al., 2000)
- Takens Embedding Theorem
- TDA in Finance (Gidea & Katz, 2018)
Usage:
    tda = PersistentHomology(max_dimension=2)
    cloud = tda.build_point_cloud(prices, embedding_dim=3)
    betti = tda.compute_betti_numbers(cloud)
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style
from scipy.spatial.distance import pdist, squareform


class PersistentHomology:
    """
    Persistent Homology Analyzer.
    
    Finansal veriyi geometrik şekil olarak görür.
    Piyasa çöküşlerini "şekil bozulması" olarak tespit eder.
    
    Betti Numbers:
    - β0: Bağlı bileşen sayısı (parçalanma)
    - β1: Delik sayısı (döngüler, anormallik)
    - β2: Boşluk sayısı (3D kaviteler)
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Args:
            max_dimension: Maximum homology boyutu
        """
        self.max_dimension = max_dimension
        self.diagrams = []
        self.betti_history = []
    
    def build_point_cloud(self, 
                         prices: np.ndarray,
                         embedding_dim: int = 3,
                         delay: int = 1) -> np.ndarray:
        """
        Fiyat serisinden point cloud oluştur (Takens embedding).
        """
        n = len(prices) - (embedding_dim - 1) * delay
        
        if n <= 0:
            return np.array([])
        
        cloud = np.zeros((n, embedding_dim))
        
        for i in range(n):
            for j in range(embedding_dim):
                cloud[i, j] = prices[i + j * delay]
        
        # Normalize
        cloud = (cloud - np.mean(cloud, axis=0)) / (np.std(cloud, axis=0) + 1e-10)
        
        return cloud
    
    def compute_vietoris_rips(self, 
                             point_cloud: np.ndarray,
                             max_epsilon: float = 2.0,
                             n_steps: int = 50) -> Dict:
        """
        Vietoris-Rips complexi hesapla.
        
        Basitleştirilmiş implementasyon (gerçekte: Ripser, GUDHI kullanılır).
        """
        n_points = len(point_cloud)
        
        if n_points < 4:
            return {"error": "Yetersiz nokta"}
        
        # Mesafe matrisi
        distances = squareform(pdist(point_cloud))
        
        # Epsilon değerleri
        epsilons = np.linspace(0, max_epsilon, n_steps)
        
        persistence_diagram = {
            "dim_0": [],  # Connected components
            "dim_1": [],  # Loops/holes
        }
        
        betti_numbers = {"beta_0": [], "beta_1": []}
        
        for eps in epsilons:
            # Adjacency matrix (ε-neighborhood graph)
            adj = (distances <= eps).astype(int)
            np.fill_diagonal(adj, 0)
            
            # β0: Connected components (basit BFS)
            visited = set()
            components = 0
            
            for i in range(n_points):
                if i not in visited:
                    # BFS
                    queue = [i]
                    while queue:
                        node = queue.pop(0)
                        if node not in visited:
                            visited.add(node)
                            neighbors = np.where(adj[node] == 1)[0]
                            queue.extend([n for n in neighbors if n not in visited])
                    components += 1
            
            betti_numbers["beta_0"].append(components)
            
            # β1: Approximate loop count (edge count - vertex count + components)
            edges = np.sum(adj) // 2
            beta_1 = max(0, edges - n_points + components)
            betti_numbers["beta_1"].append(beta_1)
        
        # Persistence hesapla (doğum-ölüm çiftleri)
        beta_0_series = betti_numbers["beta_0"]
        for i in range(1, len(beta_0_series)):
            if beta_0_series[i] < beta_0_series[i-1]:
                # Component öldü
                birth = epsilons[0]
                death = epsilons[i]
                persistence_diagram["dim_0"].append((birth, death))
        
        beta_1_series = betti_numbers["beta_1"]
        for i in range(1, len(beta_1_series)):
            if beta_1_series[i] > beta_1_series[i-1]:
                # Loop doğdu
                birth = epsilons[i]
                persistence_diagram["dim_1"].append((birth, None))  # Ölüm zamanı belirsiz
        
        return {
            "epsilons": epsilons.tolist(),
            "betti_numbers": betti_numbers,
            "persistence_diagram": persistence_diagram
        }
    
    def detect_topological_anomaly(self, prices: np.ndarray, lookback: int = 100) -> Dict:
        """
        Topolojik anormallik tespit et.
        
        Kriz öncesi sinyal: Betti sayılarında ani değişim.
        """
        print(f"{Fore.CYAN}📐 TDA Analizi başlıyor...{Style.RESET_ALL}", flush=True)
        
        if len(prices) < lookback:
            return {"error": "Yetersiz veri"}
        
        # Son veriyle point cloud oluştur
        recent_prices = prices[-lookback:]
        point_cloud = self.build_point_cloud(recent_prices)
        
        if len(point_cloud) == 0:
            return {"error": "Point cloud oluşturulamadı"}
        
        # Homology hesapla
        homology = self.compute_vietoris_rips(point_cloud)
        
        if "error" in homology:
            return homology
        
        # Mevcut Betti sayıları
        beta_0 = homology["betti_numbers"]["beta_0"][-1] if homology["betti_numbers"]["beta_0"] else 1
        beta_1 = homology["betti_numbers"]["beta_1"][-1] if homology["betti_numbers"]["beta_1"] else 0
        
        # Anormallik skoru
        # Yüksek β1 (çok fazla döngü/delik) = yapı bozulması
        anomaly_score = beta_1 / max(1, beta_0)
        
        # Geçmiş karşılaştırma
        if self.betti_history:
            avg_beta_1 = np.mean([b["beta_1"] for b in self.betti_history[-20:]])
            beta_1_spike = beta_1 > avg_beta_1 * 2
        else:
            beta_1_spike = False
        
        # Kaydet
        self.betti_history.append({
            "timestamp": datetime.now().isoformat(),
            "beta_0": beta_0,
            "beta_1": beta_1,
            "anomaly_score": anomaly_score
        })
        
        # Sonuç
        is_anomaly = anomaly_score > 0.5 or beta_1_spike
        
        result = {
            "beta_0": beta_0,  # Bağlı bileşen
            "beta_1": beta_1,  # Delik sayısı
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "crisis_warning": beta_1 > 5,
            "interpretation": self._interpret_topology(beta_0, beta_1, anomaly_score)
        }
        
        if result["crisis_warning"]:
            print(f"{Fore.RED}⚠️ TDA KRİZ UYARISI: β1={beta_1}{Style.RESET_ALL}", flush=True)
        
        return result
    
    def _interpret_topology(self, beta_0: int, beta_1: int, score: float) -> str:
        """Topoloji yorumu."""
        if beta_1 > 10:
            return "🔴 CİDDİ YAPI BOZULMASI - Çok sayıda delik tespit edildi"
        elif beta_1 > 5:
            return "🟠 YAPI BOZULMA BAŞLANGICI - Kriz öncesi sinyal"
        elif beta_0 > 3:
            return "🟡 PARÇALANMA - Piyasa bölünmüş durumda"
        elif score > 0.3:
            return "🟡 HAFİF ANORMALLİK - İzlemeye devam"
        else:
            return "🟢 NORMAL - Piyasa topolojisi sağlıklı"
    
    def generate_tda_report(self) -> str:
        """TDA raporu."""
        if not self.betti_history:
            return "Henüz analiz yok"
        
        latest = self.betti_history[-1]
        
        report = f"""
<topological_analysis>
📐 TOPOLOJİK VERİ ANALİZİ
════════════════════════════════════════

🔢 BETTİ SAYILARI:
  • β0 (Bağlı Bileşen): {latest['beta_0']}
  • β1 (Delik/Döngü): {latest['beta_1']}

📊 ANORMALLİK SKORU: {latest['anomaly_score']:.3f}

💡 YORUM:
  {self._interpret_topology(latest['beta_0'], latest['beta_1'], latest['anomaly_score'])}

📈 TARİHÇE (Son 10):
"""
        for b in self.betti_history[-10:]:
            report += f"  β0={b['beta_0']}, β1={b['beta_1']}\n"
        
        report += "</topological_analysis>\n"
        return report
