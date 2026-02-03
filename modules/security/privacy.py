"""
Federated Learning for Privacy-Preserving Training
Author: Erdinc Erdogan
Purpose: Enables distributed model training without sharing raw data using federated learning with differential privacy for strategy confidentiality.
References:
- Federated Learning (McMahan et al., 2017)
- Differential Privacy
- Secure Aggregation Protocol
Usage:
    learner = FederatedLearner(client_id='client_1')
    learner.initialize_model(global_params)
    result = learner.train_local(local_data, local_labels)
"""
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from colorama import Fore, Style


class FederatedLearner:
    """
    Federated Learning Client.
    
    Veriyi merkeze göndermeden yerel eğitim yapar,
    sadece model güncellemelerini (gradients) paylaşır.
    
    Avantajlar:
    - Strateji gizliliği
    - Veri mahremiyeti
    - Distributed alpha
    """
    
    def __init__(self, client_id: str = None, learning_rate: float = 0.01):
        """
        Args:
            client_id: Benzersiz client ID
            learning_rate: Öğrenme oranı
        """
        self.client_id = client_id or hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        self.learning_rate = learning_rate
        self.local_model = None
        self.gradient_history = []
        self.round_number = 0
    
    def initialize_model(self, model_params: Dict) -> None:
        """Model parametrelerini başlat."""
        self.local_model = {
            key: np.array(val) for key, val in model_params.items()
        }
        print(f"{Fore.CYAN}🔒 Federated Client {self.client_id} initialized{Style.RESET_ALL}", flush=True)
    
    def train_local(self, local_data: np.ndarray, local_labels: np.ndarray, epochs: int = 5) -> Dict:
        """
        Yerel veri ile eğit.
        
        Veri ASLA sunucuya gitmez.
        """
        if self.local_model is None:
            return {"error": "Model not initialized"}
        
        print(f"{Fore.CYAN}📊 Yerel eğitim: {len(local_data)} örnek{Style.RESET_ALL}", flush=True)
        
        # Basit gradient descent simülasyonu
        # Gerçekte: PyTorch/TensorFlow model
        
        gradients = {}
        
        for epoch in range(epochs):
            for key in self.local_model:
                # Simüle edilmiş gradient
                noise = np.random.randn(*self.local_model[key].shape) * 0.01
                grad = noise * (1 - epoch / epochs)  # Azalan gradient
                
                # Differential privacy ekle (gürültü)
                dp_noise = np.random.laplace(0, 0.001, self.local_model[key].shape)
                grad += dp_noise
                
                gradients[key] = grad
                self.local_model[key] -= self.learning_rate * grad
        
        # Gradient'ları kaydet (göndermek için)
        self.gradient_history.append({
            "round": self.round_number,
            "gradients": {k: v.tolist() for k, v in gradients.items()},
            "num_samples": len(local_data)
        })
        
        self.round_number += 1
        
        return {
            "client_id": self.client_id,
            "round": self.round_number,
            "gradients_computed": list(gradients.keys()),
            "num_samples": len(local_data),
            "privacy": "DIFFERENTIAL_PRIVACY_APPLIED"
        }
    
    def get_encrypted_gradients(self) -> Dict:
        """
        Şifrelenmiş gradientleri al.
        
        Secure Aggregation protokolü için.
        """
        if not self.gradient_history:
            return {"error": "No gradients computed"}
        
        latest = self.gradient_history[-1]
        
        # Basit "şifreleme" simülasyonu
        # Gerçekte: Homomorphic encryption veya secure aggregation
        encrypted = {
            "client_id": self.client_id,
            "round": latest["round"],
            "encrypted_payload": hashlib.sha256(str(latest["gradients"]).encode()).hexdigest(),
            "num_samples": latest["num_samples"]
        }
        
        return encrypted
    
    def apply_global_update(self, global_params: Dict) -> None:
        """
        Global model güncellemesini uygula.
        
        Aggregated gradients'tan hesaplanan yeni parametreler.
        """
        for key in global_params:
            if key in self.local_model:
                self.local_model[key] = np.array(global_params[key])
        
        print(f"{Fore.GREEN}✅ Global update uygulandı (Round {self.round_number}){Style.RESET_ALL}", flush=True)


class FederatedAggregator:
    """
    Federated Learning Server (Aggregator).
    
    Client gradient'larını toplar ve güvenli şekilde birleştirir.
    """
    
    def __init__(self):
        self.clients = {}
        self.global_model = None
        self.aggregation_history = []
    
    def register_client(self, client: FederatedLearner) -> None:
        """Client kaydet."""
        self.clients[client.client_id] = client
        print(f"{Fore.CYAN}➕ Client kayıt: {client.client_id}{Style.RESET_ALL}", flush=True)
    
    def initialize_global_model(self, model_params: Dict) -> None:
        """Global modeli başlat."""
        self.global_model = {
            key: np.array(val) for key, val in model_params.items()
        }
        
        # Tüm client'lara dağıt
        for client in self.clients.values():
            client.initialize_model(model_params)
    
    def aggregate_fedavg(self, client_updates: List[Dict]) -> Dict:
        """
        FedAvg (Federated Averaging) aggregation.
        
        Ağırlıklı ortalama: Her client'ın katkısı sample sayısına göre.
        """
        print(f"{Fore.CYAN}🔄 FedAvg Aggregation: {len(client_updates)} client{Style.RESET_ALL}", flush=True)
        
        if not client_updates:
            return {"error": "No updates to aggregate"}
        
        total_samples = sum(u.get("num_samples", 1) for u in client_updates)
        
        # Her key için ağırlıklı ortalama
        aggregated = {}
        
        for key in self.global_model:
            weighted_sum = np.zeros_like(self.global_model[key])
            
            for update in client_updates:
                weight = update.get("num_samples", 1) / total_samples
                grads = update.get("gradients", {}).get(key, np.zeros_like(self.global_model[key]))
                weighted_sum += weight * np.array(grads)
            
            # Global model güncelle
            self.global_model[key] -= 0.01 * weighted_sum
            aggregated[key] = self.global_model[key].tolist()
        
        # History'e ekle
        self.aggregation_history.append({
            "round": len(self.aggregation_history),
            "num_clients": len(client_updates),
            "total_samples": total_samples
        })
        
        return {
            "aggregated_params": aggregated,
            "num_clients": len(client_updates),
            "total_samples": total_samples
        }
    
    def broadcast_update(self) -> None:
        """Global güncellemeyi tüm client'lara yayınla."""
        for client in self.clients.values():
            client.apply_global_update(self.global_model)
    
    def run_round(self, local_data_per_client: Dict[str, Tuple]) -> Dict:
        """
        Bir federated learning round'u çalıştır.
        
        Args:
            local_data_per_client: {client_id: (X, y)}
        """
        updates = []
        
        # Her client yerel eğitim yapar
        for client_id, (X, y) in local_data_per_client.items():
            if client_id in self.clients:
                client = self.clients[client_id]
                result = client.train_local(X, y)
                
                updates.append({
                    "client_id": client_id,
                    "gradients": client.gradient_history[-1]["gradients"] if client.gradient_history else {},
                    "num_samples": result.get("num_samples", 0)
                })
        
        # Aggregate
        agg_result = self.aggregate_fedavg(updates)
        
        # Broadcast
        self.broadcast_update()
        
        return {
            "round": len(self.aggregation_history),
            "aggregation": agg_result
        }
    
    def generate_federated_report(self) -> str:
        """Federated learning raporu."""
        report = f"""
<federated_learning>
🔒 FEDERATED LEARNING RAPORU
════════════════════════════════════════

📊 GENEL:
  • Client Sayısı: {len(self.clients)}
  • Toplam Round: {len(self.aggregation_history)}

🔐 MAHREMİYET:
  • Differential Privacy: ✅ Aktif
  • Secure Aggregation: ✅ Simüle
  • Veri Paylaşımı: ❌ Yok

📈 SON ROUND'LAR:
"""
        for agg in self.aggregation_history[-5:]:
            report += f"  • Round {agg['round']}: {agg['num_clients']} client, {agg['total_samples']} sample\n"
        
        report += "</federated_learning>\n"
        return report
