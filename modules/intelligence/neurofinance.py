"""
Neuro-Finance Biometric Risk Monitor
Author: Erdinc Erdogan
Purpose: Monitors trader biometrics (heart rate, HRV, sleep, stress) from wearables to detect impaired decision-making states and trigger trading safeguards.
References:
- Neurofinance (Lo & Repin, 2002)
- Heart Rate Variability (HRV) Analysis
- Cognitive Load Theory
Usage:
    monitor = BiometricMonitor()
    assessment = monitor.update_metrics({"heart_rate": 95, "sleep_score": 65, "hrv": 35})
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from colorama import Fore, Style


class HumanState(Enum):
    """İnsan durumu."""
    OPTIMAL = "optimal"           # En iyi karar alma
    FATIGUED = "fatigued"         # Yorgun
    STRESSED = "stressed"         # Stresli
    IMPAIRED = "impaired"         # Ciddi bozulma
    CRITICAL = "critical"         # Manuel müdahale engellenmeli


class BiometricMonitor:
    """
    Biometric Integration Monitor.
    
    Apple Watch, Oura Ring, Fitbit gibi cihazlardan
    biyometrik veri okuyarak insan faktörü riski ölçer.
    """
    
    # Normal aralıklar
    THRESHOLDS = {
        "heart_rate": {"min": 50, "max": 100, "stress": 110, "panic": 130},
        "hrv": {"min": 20, "optimal": 50},  # Heart Rate Variability
        "sleep_score": {"min": 70, "poor": 50, "critical": 30},
        "stress_level": {"low": 30, "medium": 60, "high": 80},
        "body_battery": {"low": 25, "medium": 50, "good": 75}
    }
    
    def __init__(self):
        self.current_metrics = {}
        self.history = []
        self.human_state = HumanState.OPTIMAL
        self.alerts = []
        self.override_active = False
    
    def update_metrics(self, metrics: Dict) -> Dict:
        """
        Biyometrik metrikleri güncelle.
        
        Args:
            metrics: {"heart_rate": 75, "sleep_score": 85, ...}
        """
        self.current_metrics = metrics
        self.current_metrics["timestamp"] = datetime.now().isoformat()
        self.history.append(self.current_metrics)
        
        # Durumu değerlendir
        assessment = self._assess_state()
        
        return assessment
    
    def _assess_state(self) -> Dict:
        """İnsan durumunu değerlendir."""
        risk_score = 0
        issues = []
        
        # Heart rate analizi
        hr = self.current_metrics.get("heart_rate", 70)
        if hr > self.THRESHOLDS["heart_rate"]["panic"]:
            risk_score += 40
            issues.append(f"PANİK: Nabız {hr}")
        elif hr > self.THRESHOLDS["heart_rate"]["stress"]:
            risk_score += 20
            issues.append(f"Stres: Nabız {hr}")
        
        # HRV analizi (düşük = stres)
        hrv = self.current_metrics.get("hrv", 50)
        if hrv < self.THRESHOLDS["hrv"]["min"]:
            risk_score += 25
            issues.append(f"Düşük HRV: {hrv}")
        
        # Uyku skoru
        sleep = self.current_metrics.get("sleep_score", 80)
        if sleep < self.THRESHOLDS["sleep_score"]["critical"]:
            risk_score += 35
            issues.append(f"Kritik uyku: {sleep}")
        elif sleep < self.THRESHOLDS["sleep_score"]["poor"]:
            risk_score += 15
            issues.append(f"Kötü uyku: {sleep}")
        
        # Stres seviyesi
        stress = self.current_metrics.get("stress_level", 30)
        if stress > self.THRESHOLDS["stress_level"]["high"]:
            risk_score += 30
            issues.append(f"Yüksek stres: {stress}")
        
        # Durum belirleme
        if risk_score >= 80:
            self.human_state = HumanState.CRITICAL
        elif risk_score >= 60:
            self.human_state = HumanState.IMPAIRED
        elif risk_score >= 40:
            self.human_state = HumanState.STRESSED
        elif risk_score >= 20:
            self.human_state = HumanState.FATIGUED
        else:
            self.human_state = HumanState.OPTIMAL
        
        # Kritik durumda alert
        if self.human_state in [HumanState.CRITICAL, HumanState.IMPAIRED]:
            alert = {
                "type": "HUMAN_RISK",
                "state": self.human_state.value,
                "risk_score": risk_score,
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            print(f"{Fore.RED}⚠️ İNSAN RİSKİ: {self.human_state.value}{Style.RESET_ALL}", flush=True)
        
        return {
            "state": self.human_state.value,
            "risk_score": risk_score,
            "issues": issues,
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Duruma göre öneriler."""
        recommendations = []
        
        if self.human_state == HumanState.CRITICAL:
            recommendations = [
                "❌ Manuel müdahale YASAKLANDI",
                "🔒 Yeni pozisyon açılamaz",
                "📉 Mevcut pozisyonlar küçültülüyor",
                "⏸️ Sadece stop-loss'lar aktif"
            ]
        elif self.human_state == HumanState.IMPAIRED:
            recommendations = [
                "⚠️ Büyük işlemler onay gerektirir",
                "📉 Risk limitleri %50 düşürüldü",
                "⏰ 2 saat sonra tekrar değerlendir"
            ]
        elif self.human_state == HumanState.STRESSED:
            recommendations = [
                "🧘 5 dakika mola önerilir",
                "📉 Risk limitleri %25 düşürüldü"
            ]
        elif self.human_state == HumanState.FATIGUED:
            recommendations = [
                "☕ Kahve molası?",
                "👀 Kararları gözden geçir"
            ]
        
        return recommendations
    
    def get_risk_multiplier(self) -> float:
        """Risk çarpanı: İnsan durumuna göre pozisyon boyutu ayarı."""
        multipliers = {
            HumanState.OPTIMAL: 1.0,
            HumanState.FATIGUED: 0.8,
            HumanState.STRESSED: 0.5,
            HumanState.IMPAIRED: 0.25,
            HumanState.CRITICAL: 0.0  # İşlem yapma
        }
        return multipliers.get(self.human_state, 0.5)
    
    def can_trade(self) -> bool:
        """İşlem yapılabilir mi?"""
        return self.human_state not in [HumanState.CRITICAL]
    
    def requires_confirmation(self) -> bool:
        """Onay gerekli mi?"""
        return self.human_state in [HumanState.IMPAIRED, HumanState.STRESSED]
    
    def simulate_wearable_data(self) -> Dict:
        """Wearable veri simülasyonu (test için)."""
        return {
            "heart_rate": np.random.randint(60, 100),
            "hrv": np.random.randint(20, 80),
            "sleep_score": np.random.randint(50, 95),
            "stress_level": np.random.randint(10, 70),
            "body_battery": np.random.randint(30, 100),
            "steps_today": np.random.randint(0, 15000)
        }
    
    def generate_biometric_report(self) -> str:
        """Biyometrik rapor."""
        m = self.current_metrics
        
        state_emoji = {
            HumanState.OPTIMAL: "🟢",
            HumanState.FATIGUED: "🟡",
            HumanState.STRESSED: "🟠",
            HumanState.IMPAIRED: "🔴",
            HumanState.CRITICAL: "⛔"
        }
        
        report = f"""
<biometric_monitor>
🧠 NEURO-FİNANS RAPORU
════════════════════════════════════════

📊 BİYOMETRİK VERİLER:
  • Nabız: {m.get('heart_rate', 'N/A')} bpm
  • HRV: {m.get('hrv', 'N/A')} ms
  • Uyku Skoru: {m.get('sleep_score', 'N/A')}
  • Stres: {m.get('stress_level', 'N/A')}

{state_emoji.get(self.human_state, '⚪')} DURUM: {self.human_state.value.upper()}

⚙️ SİSTEM AYARLARI:
  • Risk Çarpanı: {self.get_risk_multiplier()}
  • İşlem İzni: {'✅' if self.can_trade() else '❌'}
  • Onay Gerekli: {'✅' if self.requires_confirmation() else '❌'}

💡 ÖNERİLER:
"""
        for rec in self._get_recommendations():
            report += f"  {rec}\n"
        
        report += "</biometric_monitor>\n"
        return report
