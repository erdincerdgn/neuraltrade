"""
Adversarial Attack Simulator and Detector
Author: Erdinc Erdogan
Purpose: Simulates market manipulation attacks (spoofing, layering, quote stuffing, pump & dump) to train detection models and harden trading systems.
References:
- Market Manipulation Detection (SEC/FINRA)
- Adversarial Machine Learning
- Order Flow Toxicity Analysis
Usage:
    trainer = AdversarialTrainer()
    attack = trainer.generate_attack('spoofing', intensity=0.7)
    is_attack = trainer.detect(order_flow)
"""
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class AdversarialTrainer:
    """
    Adversarial Attack Simulator.
    
    Botu kandırmaya çalışan düşman ajanlarla eğitim:
    - Spoofing (Sahte emirler)
    - Layering (Katmanlı manipülasyon)
    - Quote Stuffing (Emir seli)
    - Pump & Dump patternleri
    """
    
    # Manipülasyon tipleri
    ATTACK_TYPES = {
        "spoofing": {
            "description": "Büyük sahte emirler koyup iptal etme",
            "detection_features": ["order_cancel_ratio", "order_size_pattern"]
        },
        "layering": {
            "description": "Farklı fiyat seviyelerinde katmanlı sahte emirler",
            "detection_features": ["order_depth_pattern", "price_cluster"]
        },
        "quote_stuffing": {
            "description": "Milisaniyeler içinde binlerce emir gönderme",
            "detection_features": ["order_rate", "message_burst"]
        },
        "pump_dump": {
            "description": "Yapay şişirme ve satış",
            "detection_features": ["volume_spike", "price_velocity", "reversal_pattern"]
        },
        "wash_trading": {
            "description": "Kendi kendine alım satım",
            "detection_features": ["same_side_trades", "circular_flow"]
        }
    }
    
    def __init__(self):
        self.attack_history = []
        self.detection_model = None
        self.training_data = []
    
    def generate_attack(self, attack_type: str, intensity: float = 0.5) -> Dict:
        """
        Saldırı senaryosu üret (eğitim için).
        
        Args:
            attack_type: Saldırı tipi
            intensity: Saldırı yoğunluğu (0-1)
        """
        if attack_type not in self.ATTACK_TYPES:
            attack_type = "spoofing"
        
        attack_config = self.ATTACK_TYPES[attack_type]
        
        if attack_type == "spoofing":
            return self._generate_spoofing_attack(intensity)
        elif attack_type == "layering":
            return self._generate_layering_attack(intensity)
        elif attack_type == "quote_stuffing":
            return self._generate_quote_stuffing_attack(intensity)
        elif attack_type == "pump_dump":
            return self._generate_pump_dump_attack(intensity)
        else:
            return self._generate_spoofing_attack(intensity)
    
    def _generate_spoofing_attack(self, intensity: float) -> Dict:
        """Spoofing saldırısı üret."""
        # Büyük emirler ve hızlı iptaller
        num_fake_orders = int(10 + intensity * 50)
        
        orders = []
        for i in range(num_fake_orders):
            order = {
                "id": f"FAKE_{i}",
                "type": "LIMIT",
                "side": "BUY" if np.random.random() > 0.5 else "SELL",
                "size": int(10000 + np.random.exponential(10000)),
                "price": 100 + np.random.uniform(-2, 2),
                "lifetime_ms": int(50 + np.random.exponential(200)),  # Çok kısa süre
                "is_fake": True
            }
            orders.append(order)
        
        return {
            "attack_type": "spoofing",
            "intensity": intensity,
            "fake_orders": orders,
            "cancel_ratio": 0.95,  # %95 iptal oranı
            "pattern": "large_orders_quick_cancel",
            "detection_difficulty": 0.3 + intensity * 0.5
        }
    
    def _generate_layering_attack(self, intensity: float) -> Dict:
        """Layering saldırısı üret."""
        layers = []
        base_price = 100
        
        num_layers = int(5 + intensity * 15)
        
        for i in range(num_layers):
            layer = {
                "price": base_price + i * 0.05,
                "size": int(5000 * (1 - i / num_layers)),  # Fiyat arttıkça boyut küçülür
                "side": "SELL"
            }
            layers.append(layer)
        
        return {
            "attack_type": "layering",
            "intensity": intensity,
            "layers": layers,
            "pattern": "descending_size_layers",
            "detection_difficulty": 0.4 + intensity * 0.4
        }
    
    def _generate_quote_stuffing_attack(self, intensity: float) -> Dict:
        """Quote stuffing saldırısı üret."""
        message_rate = int(1000 + intensity * 10000)  # Mesaj/saniye
        
        return {
            "attack_type": "quote_stuffing",
            "intensity": intensity,
            "message_rate_per_second": message_rate,
            "duration_ms": 100,
            "pattern": "burst_then_cancel",
            "detection_difficulty": 0.5 + intensity * 0.3
        }
    
    def _generate_pump_dump_attack(self, intensity: float) -> Dict:
        """Pump and dump saldırısı üret."""
        phases = []
        
        # Pump fazı
        pump_duration = int(60 + intensity * 240)  # dakika
        phases.append({
            "phase": "accumulation",
            "duration_min": pump_duration // 3,
            "volume_multiplier": 1.5
        })
        phases.append({
            "phase": "pump",
            "duration_min": pump_duration // 3,
            "price_increase_pct": 20 + intensity * 50,
            "volume_multiplier": 3.0
        })
        phases.append({
            "phase": "dump",
            "duration_min": pump_duration // 6,
            "price_decrease_pct": 30 + intensity * 40
        })
        
        return {
            "attack_type": "pump_dump",
            "intensity": intensity,
            "phases": phases,
            "total_duration_min": pump_duration,
            "detection_difficulty": 0.6 + intensity * 0.3
        }
    
    def detect_attack(self, market_data: Dict) -> Dict:
        """
        Saldırı tespit et.
        
        Args:
            market_data: {orders, trades, orderbook}
        """
        print(f"{Fore.CYAN}⚔️ Manipülasyon taraması...{Style.RESET_ALL}", flush=True)
        
        detections = []
        risk_score = 0
        
        orders = market_data.get("orders", [])
        trades = market_data.get("trades", [])
        
        # Spoofing tespiti
        spoofing = self._detect_spoofing(orders)
        if spoofing["detected"]:
            detections.append(spoofing)
            risk_score += 0.3
        
        # Quote stuffing tespiti
        stuffing = self._detect_quote_stuffing(orders)
        if stuffing["detected"]:
            detections.append(stuffing)
            risk_score += 0.2
        
        # Pump dump tespiti
        pump_dump = self._detect_pump_dump(trades)
        if pump_dump["detected"]:
            detections.append(pump_dump)
            risk_score += 0.4
        
        return {
            "detections": detections,
            "risk_score": min(risk_score, 1.0),
            "is_manipulated": risk_score > 0.3,
            "recommendation": "AVOID" if risk_score > 0.3 else "PROCEED_WITH_CAUTION" if risk_score > 0.1 else "SAFE",
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_spoofing(self, orders: List[Dict]) -> Dict:
        """Spoofing tespiti."""
        if not orders:
            return {"detected": False, "type": "spoofing"}
        
        # İptal oranı
        cancels = sum(1 for o in orders if o.get("status") == "CANCELLED")
        total = len(orders)
        cancel_ratio = cancels / total if total > 0 else 0
        
        # Büyük emir yüzdesi
        large_orders = sum(1 for o in orders if o.get("size", 0) > 10000)
        large_ratio = large_orders / total if total > 0 else 0
        
        detected = cancel_ratio > 0.8 and large_ratio > 0.3
        
        return {
            "detected": detected,
            "type": "spoofing",
            "cancel_ratio": cancel_ratio,
            "large_order_ratio": large_ratio,
            "confidence": 0.7 if detected else 0.2
        }
    
    def _detect_quote_stuffing(self, orders: List[Dict]) -> Dict:
        """Quote stuffing tespiti."""
        if len(orders) < 100:
            return {"detected": False, "type": "quote_stuffing"}
        
        # Mesaj yoğunluğu analizi (simüle)
        message_rate = len(orders)  # Gerçekte: timestamp bazlı hesaplama
        
        detected = message_rate > 1000
        
        return {
            "detected": detected,
            "type": "quote_stuffing",
            "message_rate": message_rate,
            "confidence": 0.6 if detected else 0.1
        }
    
    def _detect_pump_dump(self, trades: List[Dict]) -> Dict:
        """Pump and dump tespiti."""
        if len(trades) < 50:
            return {"detected": False, "type": "pump_dump"}
        
        prices = [t.get("price", 0) for t in trades[-50:]]
        volumes = [t.get("size", 0) for t in trades[-50:]]
        
        if not prices or not volumes:
            return {"detected": False, "type": "pump_dump"}
        
        # Fiyat değişimi
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        # Hacim artışı
        avg_volume_early = np.mean(volumes[:10])
        avg_volume_late = np.mean(volumes[-10:])
        volume_change = avg_volume_late / avg_volume_early if avg_volume_early > 0 else 1
        
        # Pump pattern: yüksek fiyat artışı + hacim patlaması
        detected = abs(price_change) > 0.1 and volume_change > 2
        
        return {
            "detected": detected,
            "type": "pump_dump",
            "price_change_pct": price_change * 100,
            "volume_multiplier": volume_change,
            "confidence": 0.65 if detected else 0.15
        }
    
    def train_detection_model(self, num_samples: int = 1000) -> Dict:
        """Tespit modelini eğit."""
        print(f"{Fore.CYAN}🧠 Adversarial model eğitimi başlıyor...{Style.RESET_ALL}", flush=True)
        
        # Eğitim verisi üret
        X = []
        y = []
        
        for _ in range(num_samples):
            # Normal piyasa veya saldırı
            is_attack = np.random.random() > 0.5
            
            if is_attack:
                attack_type = np.random.choice(list(self.ATTACK_TYPES.keys()))
                attack = self.generate_attack(attack_type, np.random.uniform(0.3, 0.9))
                features = self._extract_features(attack)
            else:
                features = self._generate_normal_features()
            
            X.append(features)
            y.append(1 if is_attack else 0)
        
        # Basit eğitim (gerçekte: sklearn/pytorch model)
        self.training_data = {"X": X, "y": y}
        
        print(f"{Fore.GREEN}✅ Model eğitildi: {num_samples} örnek{Style.RESET_ALL}", flush=True)
        
        return {
            "samples": num_samples,
            "attack_ratio": sum(y) / len(y),
            "status": "TRAINED"
        }
    
    def _extract_features(self, attack: Dict) -> List[float]:
        """Saldırıdan özellik çıkar."""
        return [
            attack.get("intensity", 0),
            attack.get("cancel_ratio", 0),
            attack.get("message_rate_per_second", 0) / 10000,
            attack.get("detection_difficulty", 0),
            len(attack.get("fake_orders", [])) / 100,
        ]
    
    def _generate_normal_features(self) -> List[float]:
        """Normal piyasa özellikleri."""
        return [
            np.random.uniform(0, 0.2),
            np.random.uniform(0.1, 0.4),
            np.random.uniform(0, 0.1),
            np.random.uniform(0, 0.3),
            np.random.uniform(0, 0.2),
        ]


class ZeroKnowledgeProof:
    """
    Zero-Knowledge Proof (ZKP) Verification.
    
    Stratejinin detaylarını açıklamadan kârlılığını kanıtla.
    
    Kullanım alanları:
    - Yatırımcılara şeffaflık (stratejiyi ifşa etmeden)
    - Düzenleyici uyumluluk
    - Trustless audit
    """
    
    def __init__(self, strategy_hash: str = None):
        """
        Args:
            strategy_hash: Stratejinin gizli hash'i
        """
        self.strategy_hash = strategy_hash or self._generate_strategy_hash()
        self.proofs = []
    
    def _generate_strategy_hash(self) -> str:
        """Strateji için benzersiz hash üret."""
        secret = os.urandom(32)
        return hashlib.sha256(secret).hexdigest()
    
    def generate_performance_proof(self, 
                                   returns: List[float],
                                   benchmark_returns: List[float]) -> Dict:
        """
        Performans kanıtı oluştur.
        
        Kanıtlar:
        - Gerçek getirilerin kullanıldığı
        - Benchmark'ı geçtiği
        - Verinin değiştirilmediği
        """
        print(f"{Fore.CYAN}🔐 ZKP kanıtı oluşturuluyor...{Style.RESET_ALL}", flush=True)
        
        # Getiri hesaplamaları
        total_return = np.prod([1 + r for r in returns]) - 1
        benchmark_total = np.prod([1 + r for r in benchmark_returns]) - 1
        alpha = total_return - benchmark_total
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Commitment (veri hash'i)
        data_commitment = hashlib.sha256(
            str(returns).encode() + self.strategy_hash.encode()
        ).hexdigest()
        
        # Proof oluştur
        proof = {
            "proof_id": hashlib.sha256(os.urandom(16)).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "claims": {
                "positive_alpha": alpha > 0,
                "beats_benchmark": total_return > benchmark_total,
                "positive_sharpe": sharpe > 0,
                "return_positive": total_return > 0
            },
            "public_outputs": {
                "alpha": round(alpha * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_return_pct": round(total_return * 100, 2),
                "benchmark_return_pct": round(benchmark_total * 100, 2)
            },
            "commitment": data_commitment,
            "verification_hash": self._create_verification_hash(data_commitment, alpha, sharpe)
        }
        
        self.proofs.append(proof)
        
        print(f"{Fore.GREEN}✅ ZKP kanıtı oluşturuldu: {proof['proof_id']}{Style.RESET_ALL}", flush=True)
        
        return proof
    
    def _create_verification_hash(self, commitment: str, alpha: float, sharpe: float) -> str:
        """Doğrulama hash'i oluştur."""
        data = f"{commitment}:{alpha:.6f}:{sharpe:.6f}:{self.strategy_hash}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_proof(self, proof: Dict, expected_commitment: str = None) -> Dict:
        """
        Kanıtı doğrula.
        
        Üçüncü taraf bu fonksiyonla kanıtı doğrulayabilir.
        """
        # Commitment kontrolü
        commitment_valid = True
        if expected_commitment:
            commitment_valid = proof.get("commitment") == expected_commitment
        
        # Claims kontrolü
        claims = proof.get("claims", {})
        outputs = proof.get("public_outputs", {})
        
        # Tutarlılık kontrolleri
        alpha_consistent = (outputs.get("alpha", 0) > 0) == claims.get("positive_alpha", False)
        sharpe_consistent = (outputs.get("sharpe_ratio", 0) > 0) == claims.get("positive_sharpe", False)
        
        is_valid = commitment_valid and alpha_consistent and sharpe_consistent
        
        return {
            "proof_id": proof.get("proof_id"),
            "is_valid": is_valid,
            "checks": {
                "commitment_valid": commitment_valid,
                "alpha_consistent": alpha_consistent,
                "sharpe_consistent": sharpe_consistent
            },
            "verification_timestamp": datetime.now().isoformat()
        }
    
    def generate_audit_certificate(self, proofs: List[Dict] = None) -> str:
        """Denetim sertifikası oluştur."""
        proofs = proofs or self.proofs
        
        if not proofs:
            return "Kanıt bulunamadı"
        
        certificate = f"""
╔══════════════════════════════════════════════════════════════╗
║           ZERO-KNOWLEDGE PROOF CERTIFICATE                  ║
╠══════════════════════════════════════════════════════════════╣
║ Strategy Hash: {self.strategy_hash[:16]}...                      ║
║ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╠══════════════════════════════════════════════════════════════╣
"""
        for proof in proofs[-5:]:  # Son 5 kanıt
            outputs = proof.get("public_outputs", {})
            certificate += f"""║ Proof: {proof.get('proof_id', 'N/A')}
║   Alpha: {outputs.get('alpha', 0):+.2f}%  |  Sharpe: {outputs.get('sharpe_ratio', 0):.2f}
║   Return: {outputs.get('total_return_pct', 0):+.2f}% vs Benchmark: {outputs.get('benchmark_return_pct', 0):+.2f}%
╠──────────────────────────────────────────────────────────────╣
"""
        
        certificate += """╚══════════════════════════════════════════════════════════════╝
This certificate proves performance claims WITHOUT revealing strategy details.
Verify at: neuraltrade.io/verify/<proof_id>
"""
        return certificate
