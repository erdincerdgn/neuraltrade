"""
Agentic Tool Use for LLM Trading
Author: Erdinc Erdogan
Purpose: Enables LLM to dynamically decide which tools (web search, calculations, document search, chart analysis) to use for answering trading queries.
References:
- ReAct Pattern (Reason + Act)
- LangChain Tool Selection
- Multi-Tool Agent Architectures
Usage:
    agent = AgenticToolUse(llm=llm)
    tools = agent.plan_tools("What's the RSI strategy for AAPL?")
"""
from typing import List


class AgenticToolUse:
    """LLM'in hangi araçları kullanacağına dinamik karar vermesi."""
    
    AVAILABLE_TOOLS = {
        "web_search": "Web'den güncel haber ara",
        "calculation": "Python REPL ile hesaplama yap", 
        "document_search": "Döküman havuzunda ara",
        "chart_analysis": "Grafik formasyonlarını analiz et",
        "fact_check": "Fiyat doğrulama yap"
    }
    
    def __init__(self, llm):
        self.llm = llm
    
    def plan_tools(self, query: str) -> List[str]:
        """Sorgu için hangi araçları kullanacağına karar ver."""
        prompt = f"""Aşağıdaki trading sorusu için hangi araçları kullanmalısın?

SORU: {query}

MEVCUT ARAÇLAR:
- web_search: Son haberler için
- calculation: Matematiksel hesaplamalar için
- document_search: Kitap stratejileri için
- chart_analysis: Grafik analizi için
- fact_check: Gerçek zamanlı fiyat kontrolü için

Sadece araç isimlerini virgülle ayırarak yaz:"""

        try:
            response = self.llm.invoke(prompt)
            tools = [t.strip().lower() for t in response.split(",")]
            return [t for t in tools if t in self.AVAILABLE_TOOLS]
        except:
            return ["document_search", "fact_check"]
