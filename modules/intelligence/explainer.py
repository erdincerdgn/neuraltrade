"""
Explainable AI (XAI) Decision Explainer
Author: Erdinc Erdogan
Purpose: Provides transparency into model decisions using SHAP values, feature importance, and natural language explanations for trading signals.
References:
- SHAP (Shapley Additive Explanations)
- Feature Importance Analysis
- Partial Dependence Plots
Usage:
    explainer = DecisionExplainer()
    explanation = explainer.explain_decision(decision="BUY", features={"rsi": 25, "macd": 0.5})
"""
import os
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from colorama import Fore, Style


class DecisionExplainer:
    """
    Decision Explainer - Karar Şeffaflığı.
    
    Model bir karar verdiğinde, hangi girdinin kararı ne kadar
    etkilediğini matematiksel olarak gösterir.
    
    Yöntemler:
    - SHAP (Shapley Additive Explanations)
    - Feature Importance
    - Partial Dependence
    """
    
    def __init__(self):
        self.shap_available = False
        self._check_shap()
    
    def _check_shap(self):
        """SHAP kütüphanesini kontrol et."""
        try:
            import shap
            self.shap_available = True
            print(f"{Fore.GREEN}✅ SHAP yüklü{Style.RESET_ALL}", flush=True)
        except ImportError:
            print(f"{Fore.YELLOW}⚠️ SHAP yüklü değil, basit explainer kullanılacak{Style.RESET_ALL}", flush=True)
            self.shap_available = False
    
    def explain_decision(self,
                        decision: str,
                        features: Dict[str, float],
                        weights: Dict[str, float] = None) -> Dict:
        """
        Karar açıklaması oluştur.
        
        Args:
            decision: Karar (AL/SAT/BEKLE)
            features: Kullanılan özellikler ve değerleri
            weights: Özellik ağırlıkları
        """
        # Varsayılan ağırlıklar
        if weights is None:
            weights = self._get_default_weights()
        
        # Her özelliğin katkısını hesapla
        contributions = {}
        total_score = 0
        
        for feature, value in features.items():
            weight = weights.get(feature, 0.5)
            normalized_value = self._normalize_value(feature, value)
            contribution = normalized_value * weight
            contributions[feature] = {
                "value": value,
                "normalized": normalized_value,
                "weight": weight,
                "contribution": contribution,
                "direction": "BULLISH" if contribution > 0 else "BEARISH" if contribution < 0 else "NEUTRAL"
            }
            total_score += contribution
        
        # Öneme göre sırala
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]["contribution"]),
            reverse=True
        )
        
        # En etkili faktörler
        top_factors = [
            {"feature": k, **v}
            for k, v in sorted_contributions[:5]
        ]
        
        return {
            "decision": decision,
            "confidence_score": self._score_to_confidence(total_score),
            "total_score": total_score,
            "contributions": dict(sorted_contributions),
            "top_factors": top_factors,
            "explanation": self._generate_explanation(decision, top_factors)
        }
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Varsayılan özellik ağırlıkları."""
        return {
            # Teknik göstergeler
            "rsi": 0.8,
            "macd": 0.7,
            "sma_trend": 0.6,
            "volume_ratio": 0.5,
            "volatility": 0.6,
            "fvg_count": 0.4,
            
            # Temel veriler
            "pe_ratio": 0.5,
            "earnings_surprise": 0.7,
            "revenue_growth": 0.6,
            
            # Sentiment
            "news_sentiment": 0.6,
            "insider_activity": 0.7,
            "analyst_rating": 0.5,
            
            # Makro
            "fed_stance": 0.5,
            "market_trend": 0.6,
            "sector_momentum": 0.5
        }
    
    def _normalize_value(self, feature: str, value: float) -> float:
        """Değeri -1 ile 1 arasına normalize et."""
        normalization_rules = {
            "rsi": lambda v: (50 - v) / 50,  # RSI 30'da +0.4, 70'te -0.4
            "macd": lambda v: np.tanh(v / 10),
            "sma_trend": lambda v: v,  # Zaten -1/0/1
            "volume_ratio": lambda v: np.tanh((v - 1) * 2),
            "volatility": lambda v: -np.tanh(v * 10),  # Yüksek vol bearish
            "news_sentiment": lambda v: v,  # -1 ile 1 arası
            "earnings_surprise": lambda v: np.tanh(v * 5),
        }
        
        normalizer = normalization_rules.get(feature, lambda v: np.tanh(v))
        return normalizer(value)
    
    def _score_to_confidence(self, score: float) -> float:
        """Skoru güven yüzdesine çevir."""
        return min(max((np.tanh(score) + 1) / 2, 0.1), 0.95)
    
    def _generate_explanation(self, decision: str, top_factors: List[Dict]) -> str:
        """İnsan okunabilir açıklama oluştur."""
        explanation = f"Karar: {decision}\n\nEn etkili faktörler:\n"
        
        for i, factor in enumerate(top_factors[:3], 1):
            direction_emoji = "📈" if factor["direction"] == "BULLISH" else "📉" if factor["direction"] == "BEARISH" else "➡️"
            explanation += f"{i}. {direction_emoji} {factor['feature']}: {factor['value']:.2f} (Katkı: {factor['contribution']:+.3f})\n"
        
        return explanation
    
    def explain_with_shap(self, model, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        SHAP ile model açıklaması.
        
        Args:
            model: Eğitilmiş model
            X: Açıklanacak veri
            feature_names: Özellik isimleri
        """
        if not self.shap_available:
            return {"error": "SHAP yüklü değil"}
        
        import shap
        
        try:
            # SHAP explainer oluştur
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            # Feature importance
            importance = np.abs(shap_values.values).mean(axis=0)
            
            feature_importance = {
                name: float(imp)
                for name, imp in zip(feature_names, importance)
            }
            
            # Sırala
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return {
                "shap_values": shap_values.values.tolist() if hasattr(shap_values.values, 'tolist') else shap_values.values,
                "feature_importance": sorted_importance,
                "base_value": float(explainer.expected_value) if hasattr(explainer.expected_value, '__float__') else explainer.expected_value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_explanation_report(self, explanation: Dict) -> str:
        """Açıklama raporu oluştur."""
        report = f"""
<decision_explainer>
🔍 KARAR AÇIKLAMASI (XAI)
════════════════════════════════════════

📊 KARAR: {explanation['decision']}
📈 Güven: %{explanation['confidence_score']*100:.0f}
🎯 Skor: {explanation['total_score']:+.3f}

🏆 EN ETKİLİ FAKTÖRLER:
"""
        for factor in explanation['top_factors'][:5]:
            bar = self._create_contribution_bar(factor['contribution'])
            direction = "↑" if factor['direction'] == "BULLISH" else "↓" if factor['direction'] == "BEARISH" else "→"
            report += f"  {direction} {factor['feature']}: {factor['value']:.2f}\n"
            report += f"     {bar} ({factor['contribution']:+.3f})\n"
        
        report += f"""
💭 AÇIKLAMA:
{explanation['explanation']}

</decision_explainer>
"""
        return report
    
    def _create_contribution_bar(self, contribution: float, width: int = 20) -> str:
        """Katkı çubuğu oluştur."""
        normalized = int(contribution * width / 2)
        center = width // 2
        
        if normalized >= 0:
            bar = "─" * center + "█" * min(normalized, center)
        else:
            bar = "─" * max(center + normalized, 0) + "█" * min(-normalized, center) + "─" * center
        
        return f"[{bar[:width]}]"


class FeatureImportanceAnalyzer:
    """
    Feature Importance Analyzer.
    
    Model kararlarında hangi özelliklerin ne kadar etkili olduğunu analiz eder.
    """
    
    def __init__(self):
        self.importance_history = []
    
    def analyze_model_importance(self, model, feature_names: List[str]) -> Dict:
        """Model özellik önemini analiz et."""
        # Sklearn tree-based modeller için
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            importance = np.ones(len(feature_names)) / len(feature_names)
        
        feature_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, importance)
        }
        
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        self.importance_history.append({
            "timestamp": datetime.now().isoformat(),
            "importance": sorted_importance
        })
        
        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:5],
            "total_features": len(feature_names)
        }
    
    def permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str], n_repeats: int = 10) -> Dict:
        """Permutation importance hesapla."""
        try:
            from sklearn.inspection import permutation_importance as perm_imp
            
            result = perm_imp(model, X, y, n_repeats=n_repeats)
            
            importance = {
                name: {
                    "mean": float(result.importances_mean[i]),
                    "std": float(result.importances_std[i])
                }
                for i, name in enumerate(feature_names)
            }
            
            return {"permutation_importance": importance}
            
        except Exception as e:
            return {"error": str(e)}
