"""
Trade Rule Extractor from Documents
Author: Erdinc Erdogan
Purpose: Extracts actionable trading rules (entry, stop-loss, take-profit) from strategy documents and calculates automatic levels based on RSI.
References:
- LLM-Based Information Extraction
- Trading Rule Formalization
- Risk/Reward Level Calculation
Usage:
    extractor = TradeRuleExtractor(llm=llm)
    rules = extractor.extract_trade_rules(docs, ticker="AAPL", current_price=150.0)
"""
from typing import List
from langchain_core.documents import Document


class TradeRuleExtractor:
    """Dökümanlardan trade kuralları çıkarır."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_trade_rules(self, docs: List[Document], ticker: str, current_price: float) -> str:
        """Trade kurallarını dökümanlardan çıkar."""
        if not docs:
            return ""
        
        content = "\n".join([d.page_content[:500] for d in docs[:2]])
        
        prompt = f"""Aşağıdaki trading dökümanından {ticker} (şu an ${current_price:.2f}) için pratik trade kuralları çıkar.

DÖKÜMAN:
{content[:1000]}

Formatla:
- ENTRY KURALI: ...
- STOP-LOSS: ...
- TAKE-PROFIT: ...
- ÖZEL KOŞULLAR: ..."""

        try:
            response = self.llm.invoke(prompt)
            return f"\n📋 TRADE KURALLARI:\n{response}\n"
        except:
            return ""
    
    def calculate_levels(self, price: float, rsi: float) -> str:
        """RSI ve fiyata göre otomatik seviyeler hesapla."""
        if rsi < 30:
            entry = price * 0.995
            sl = price * 0.97
            tp = price * 1.03
            bias = "LONG (Aşırı Satım)"
        elif rsi > 70:
            entry = price * 1.005
            sl = price * 1.03
            tp = price * 0.97
            bias = "SHORT (Aşırı Alım)"
        else:
            entry = price
            sl = price * 0.98
            tp = price * 1.02
            bias = "NÖTR"
        
        return f"""
🎯 OTOMATİK SEVİYELER ({bias}):
  • Entry: ${entry:.2f}
  • Stop-Loss: ${sl:.2f} 
  • Take-Profit: ${tp:.2f}
"""
