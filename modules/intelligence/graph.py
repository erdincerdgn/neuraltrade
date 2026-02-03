"""
Knowledge Graph and Property Graph for Trading
Author: Erdinc Erdogan
Purpose: Maintains economic relationships (interest rates → USD strength) and sector dependencies to provide graph-based context for trading decisions.
References:
- Knowledge Graph Patterns
- Property Graph Model
- Ripple Effect Analysis
Usage:
    context = KnowledgeGraph.get_graph_context("How does interest rate affect gold?")
    effects = PropertyGraph().get_ripple_effects("AAPL", event="semiconductor_shortage")
"""
import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document


class KnowledgeGraph:
    """Ekonomik ilişkileri tutan bilgi grafiği."""
    
    RELATIONS = {
        ("faiz_artisi", "causes", "usd_guclenme"): 0.9,
        ("faiz_artisi", "causes", "altin_dusus"): 0.7,
        ("enflasyon_artisi", "causes", "faiz_artisi"): 0.8,
        ("usd_guclenme", "causes", "emtia_dusus"): 0.7,
        ("rsi_oversold", "suggests", "olasi_toparlanma"): 0.6,
    }
    
    CONCEPT_MAPPING = {
        "faiz": "faiz_artisi", "enflasyon": "enflasyon_artisi",
        "dolar": "usd_guclenme", "rsi": "rsi_oversold"
    }
    
    @classmethod
    def get_graph_context(cls, query: str) -> str:
        q = query.lower()
        found = [c for kw, c in cls.CONCEPT_MAPPING.items() if kw in q]
        relations = [(s, r, t, c) for (s, r, t), c in cls.RELATIONS.items() if s in found]
        
        if not relations:
            return ""
        
        ctx = "\n🕸️ EKONOMİK İLİŞKİLER:\n"
        for s, r, t, c in relations:
            ctx += f"  • {s.replace('_', ' ').title()} → {t.replace('_', ' ').title()} ({int(c*100)}%)\n"
        return ctx


class PropertyGraph:
    """Sektör ilişkileri ve Ripple Effect analizi."""
    
    SECTOR_RELATIONS = {
        "AAPL": ["technology", "consumer_electronics", "TSMC", "QCOM", "semiconductors"],
        "MSFT": ["technology", "cloud", "AI", "enterprise"],
        "GOOGL": ["technology", "advertising", "AI", "cloud"],
        "NVDA": ["semiconductors", "AI", "gaming", "datacenter"],
        "TSLA": ["automotive", "energy", "batteries", "lithium"],
        "AMZN": ["ecommerce", "cloud", "logistics", "retail"],
    }
    
    RIPPLE_EFFECTS = {
        "semiconductor_shortage": {"AAPL": -0.8, "TSLA": -0.6, "NVDA": -0.5},
        "ai_boom": {"NVDA": 0.9, "MSFT": 0.7, "GOOGL": 0.7},
        "interest_rate_hike": {"growth_stocks": -0.6, "banks": 0.5},
        "oil_price_increase": {"energy": 0.7, "transportation": -0.5},
        "china_tension": {"AAPL": -0.4, "semiconductors": -0.6},
    }
    
    def __init__(self, llm=None):
        self.llm = llm
        self.extracted_relations = {}
    
    def extract_relations_from_doc(self, doc: Document) -> List[Tuple[str, str, str]]:
        """Döküman içeriğinden ilişkileri otomatik çıkar."""
        relations = []
        content = doc.page_content.lower()
        
        patterns = [
            (r"(\w+) increases? (\w+)", "increases"),
            (r"(\w+) decreases? (\w+)", "decreases"),
            (r"(\w+) affects? (\w+)", "affects"),
            (r"(\w+) leads? to (\w+)", "leads_to"),
            (r"(\w+) causes? (\w+)", "causes"),
        ]
        
        for pattern, relation in patterns:
            matches = re.findall(pattern, content)
            for source, target in matches[:5]:
                relations.append((source, relation, target))
        
        return relations
    
    def get_ripple_effects(self, ticker: str, event: str = None) -> str:
        """Bir hisse için ripple effect analizi yap."""
        effects = []
        
        if ticker in self.SECTOR_RELATIONS:
            sectors = self.SECTOR_RELATIONS[ticker]
            effects.append(f"📊 {ticker} Sektörleri: {', '.join(sectors[:3])}")
        
        for event_name, impacts in self.RIPPLE_EFFECTS.items():
            if ticker in impacts:
                impact = impacts[ticker]
                direction = "📈" if impact > 0 else "📉"
                effects.append(f"{direction} {event_name}: {impact:+.0%} etki")
        
        if effects:
            return "\n<property_graph>\n🕸️ İLİŞKİ ANALİZİ:\n" + "\n".join(effects) + "\n</property_graph>\n"
        return ""
