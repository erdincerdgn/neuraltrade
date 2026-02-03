"""
Corrective RAG (C-RAG) Implementation
Author: Erdinc Erdogan
Purpose: Filters irrelevant documents using LLM-based relevance scoring with dynamic thresholds and web search fallback for improved RAG accuracy.
References:
- Corrective RAG (C-RAG) Pattern
- Dynamic Threshold Document Filtering
- Web Search Fallback for RAG
Usage:
    crag = CorrectiveRAG(llm=llm)
    filtered_docs = crag.filter_documents(query, docs, min_score=25)
"""
from typing import List, Tuple
from colorama import Fore, Style
from langchain_core.documents import Document


class CorrectiveRAG:
    """
    Döküman alaka kontrolü + Dinamik Eşik + Web Fallback.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_relevance(self, query: str, doc: Document) -> Tuple[bool, float]:
        """Dökümanın sorguyla alakalı olup olmadığını kontrol et."""
        prompt = f"""Bu döküman, aşağıdaki soruyla ALAKALI mı?
Sadece 0-100 arası bir skor yaz.

SORU: {query[:200]}
DÖKÜMAN: {doc.page_content[:300]}

ALAKA SKORU (0-100):"""

        try:
            response = self.llm.invoke(prompt)
            score = int(''.join(filter(str.isdigit, response[:5])))
            return score >= 25, min(score, 100)
        except:
            return True, 70
    
    def filter_documents(self, query: str, docs: List[Document], min_score: int = 25) -> List[Document]:
        """Alakasız dökümanları filtrele - DİNAMİK EŞİK."""
        if not docs:
            return docs
        
        # Tüm skorları topla
        scored_docs = []
        for doc in docs:
            is_relevant, score = self.check_relevance(query, doc)
            doc.metadata["relevance_score"] = score
            scored_docs.append((doc, score))
        
        # Dinamik eşik: Ortalama skorun %60'ı veya min_score
        avg_score = sum(s for _, s in scored_docs) / len(scored_docs) if scored_docs else 0
        dynamic_threshold = max(min_score, avg_score * 0.6)
        
        # Filtrele
        filtered = [doc for doc, score in scored_docs if score >= dynamic_threshold]
        
        print(f"{Fore.CYAN}    Dinamik Eşik: {dynamic_threshold:.0f} (Ort: {avg_score:.0f}){Style.RESET_ALL}", flush=True)
        
        # Fallback: Hiç döküman geçmediyse, en az ilk 2'yi döndür
        if not filtered and docs:
            return docs[:2]
        
        return filtered
    
    def web_search_fallback(self, ticker: str, query: str) -> str:
        """Web'den son haberleri çek (DuckDuckGo)."""
        print(f"{Fore.YELLOW}  → Web Fallback: {ticker} haberleri aranıyor...{Style.RESET_ALL}", flush=True)
        
        try:
            import urllib.request
            import urllib.parse
            import re
            
            search_query = urllib.parse.quote(f"{ticker} stock news today")
            url = f"https://html.duckduckgo.com/html/?q={search_query}"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8')
            
            titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
            
            if titles:
                news = [f"• {title.strip()}" for title in titles[:3]]
                result = "\n".join(news)
                print(f"{Fore.GREEN}  → Web Fallback: {len(news)} haber bulundu{Style.RESET_ALL}", flush=True)
                return f"\n🌐 SON HABERLER ({ticker}):\n{result}\n"
        except Exception as e:
            print(f"{Fore.RED}  → Web Fallback hatası: {e}{Style.RESET_ALL}", flush=True)
        
        return ""
