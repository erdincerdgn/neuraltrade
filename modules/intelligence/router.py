"""
Semantic Router and Adaptive RAG Strategy
Author: Erdinc Erdogan
Purpose: Classifies queries by type (trade/technical/general) and complexity to select optimal RAG strategy (fast/standard/deep analysis).
References:
- Semantic Query Classification
- Adaptive RAG Strategies
- Query Complexity Assessment
Usage:
    query_type = SemanticRouter.classify("What's the RSI strategy for AAPL?")
    strategy = AdaptiveRAG.get_strategy(AdaptiveRAG.assess(query))
"""
from typing import Dict
from ..core.base import QueryType, QueryComplexity


class SemanticRouter:
    """Sorguları sınıflandırır."""
    
    TRADE_KW = ["trade", "al", "sat", "strateji", "rsi", "macd", "forex"]
    GENERAL_KW = ["merhaba", "selam", "nasılsın"]
    
    @classmethod
    def classify(cls, q: str) -> QueryType:
        q = q.lower()
        if any(kw in q for kw in cls.GENERAL_KW):
            return QueryType.GENERAL
        if sum(1 for kw in cls.TRADE_KW if kw in q) >= 1:
            return QueryType.TRADE
        return QueryType.TECHNICAL


class AdaptiveRAG:
    """Sorgu karmaşıklığına göre strateji seçer."""
    
    @classmethod
    def assess(cls, q: str) -> QueryComplexity:
        q = q.lower()
        if any(x in q for x in ["nasıl etkilenir", "ilişkisi", "karşılaştır"]):
            return QueryComplexity.COMPLEX
        if any(x in q for x in ["nedir", "ne demek"]):
            return QueryComplexity.SIMPLE
        return QueryComplexity.MODERATE
    
    @classmethod
    def get_strategy(cls, c: QueryComplexity) -> Dict:
        return {
            QueryComplexity.SIMPLE: {"use_crag": False, "use_cross": False, "top_k": 3, "desc": "⚡ HIZLI"},
            QueryComplexity.MODERATE: {"use_crag": True, "use_cross": False, "top_k": 5, "desc": "📊 STANDART"},
            QueryComplexity.COMPLEX: {"use_crag": True, "use_cross": True, "top_k": 10, "desc": "🔬 AI LAB"}
        }.get(c)
