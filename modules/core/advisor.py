"""
AI Advisor Slim Edition - RAG-Enhanced Trading Advisor
Author: Erdinc Erdogan
Purpose: Provides hedge fund-level AI trading advice using Qdrant vector search and
LangChain with Ollama LLM for intelligent market analysis and recommendations.
References:
- Retrieval Augmented Generation (RAG)
- Qdrant Vector Database
- LangChain Framework
Usage:
    advisor = AIAdvisor()
    recommendation = advisor.analyze_trade(ticker="AAPL", tech_signals=signals)
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest

# Cross-Encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ============================================
# MODÜLER İMPORTLAR
# ============================================
from .base import QueryType, QueryComplexity
from .memory import SQLiteMemory

# RAG Sınıfları (inline - çünkü özele has)
from ..intelligence.corrective import CorrectiveRAG
from ..intelligence.reranker import CrossEncoderReranker
from ..intelligence.graph import KnowledgeGraph, PropertyGraph

# Intelligence
from ..intelligence.vision import VisionRAG
from ..intelligence.agentic import AgenticToolUse
from ..intelligence.orchestrator import ModelOrchestrator
from ..intelligence.router import SemanticRouter, AdaptiveRAG

# RAG Helper Sınıfları
from ..intelligence.repl import PythonREPL
from ..intelligence.extractor import TradeRuleExtractor


# ============================================
# MAIN AI ADVISOR CLASS
# ============================================
class AIAdvisor:
    """
    Hedge Fund AI Level Trading Advisor.
    40+ özellik modüler yapıda entegre.
    """
    
    def __init__(self, user_id: str = "default"):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama-service:11434")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-service:6333")
        self.user_id = user_id
        
        self.child_collection = "neural_trade_pro"
        self.parent_collection = "neural_trade_parent"
        
        # Core Components
        self.llm = OllamaLLM(model="llama3", base_url=self.ollama_host)
        self.dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.client = QdrantClient(url=self.qdrant_url)
        self.bi_encoder = Ranker(model_name="ms-marco-MultiBERT-L-12")
        
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.child_collection,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # AI Lab Components
        self.memory = SQLiteMemory()
        self.crag = CorrectiveRAG(self.llm)
        self.cross_encoder = CrossEncoderReranker()
        self.repl = PythonREPL()
        self.graph = KnowledgeGraph()
        self.rule_extractor = TradeRuleExtractor(self.llm)
        
        # Dynamic Intelligence
        self.agentic = AgenticToolUse(self.llm)
        self.vision = VisionRAG(self.ollama_host)
        
        # Hedge Fund AI
        self.property_graph = PropertyGraph(self.llm)
        self.orchestrator = ModelOrchestrator(self.ollama_host)

    def _batch_fetch_parents(self, docs: List[Document]) -> List[Document]:
        """Optimized parent fetch."""
        parent_ids = list(set(d.metadata.get("parent_id") for d in docs if d.metadata.get("parent_id")))
        if not parent_ids:
            return docs
        
        try:
            results = self.client.scroll(
                collection_name=self.parent_collection,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="metadata.parent_id", match=models.MatchAny(any=parent_ids))
                ]),
                limit=min(len(parent_ids), 100),
                with_payload=True
            )
            
            if not results[0]:
                return docs
            
            parent_map = {p.payload.get("metadata", {}).get("parent_id"): Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {})
            ) for p in results[0]}
            
            expanded = []
            seen = set()
            for doc in docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid in parent_map and pid not in seen:
                    expanded.append(parent_map[pid])
                    seen.add(pid)
            
            return expanded if expanded else docs
        except:
            return docs

    def _execute_calculation(self, query: str, tech_signals: str) -> str:
        """Python REPL ile finansal hesaplama yap."""
        calc_keywords = ["pip", "lot", "kaldıraç", "leverage", "pozisyon", "risk"]
        if not any(kw in query.lower() or kw in tech_signals.lower() for kw in calc_keywords):
            return ""
        
        print(f"{Fore.MAGENTA}  → Python REPL: Hesaplama yapılıyor...{Style.RESET_ALL}", flush=True)
        
        results = []
        results.append(self.repl.calculate_risk_reward(1.0850, 1.0800, 1.0950))
        results.append(self.repl.calculate_position_size(10000, 2, 50))
        
        if results:
            return f"\n🐍 PYTHON HESAPLAMALARI:\n  • " + "\n  • ".join(results) + "\n"
        return ""

    def _get_prompt(self, use_full: bool = True) -> str:
        if use_full:
            return """Sen AI Lab seviyesinde bir algoritmik trade danışmanısın.

<user_profile>
{personalization}
</user_profile>

<past_performance>
{past_performance}
</past_performance>

<graph_rag>
{graph_context}
</graph_rag>

<calculations>
{calculations}
</calculations>

<technical_data ticker="{ticker}">
{tech_signals}
Piyasa: {market_sentiment}
{fact_check}
</technical_data>

<strategies>
{context}
</strategies>

═══════════════════════════════════════════════════════════
🧠 CHAIN-OF-THOUGHT ANALİZ
═══════════════════════════════════════════════════════════

**ADIM 1 - TEKNİK ANALİZ:** Verileri analiz et.
**ADIM 2 - KİTAP STRATEJİLERİ:** Önerileri çıkar.
**ADIM 3 - KULLANICI PROFİLİ:** Risk toleransına göre ayarla.
**ADIM 4 - GEÇMİŞ PERFORMANS:** Önceki hatalardan ders al.
**ADIM 5 - EKONOMİK İLİŞKİLER:** Graph-RAG'dan çıkarımlar.
**ADIM 6 - HESAPLAMALAR:** Python REPL sonuçlarını değerlendir.
**ADIM 7 - RİSK DEĞERLENDİRMESİ:** 2+ risk belirt.
**ADIM 8 - NİHAİ KARAR:** 'AL 🟢', 'SAT 🔴' veya 'BEKLE 🟡'

ANALİZ:"""
        else:
            return """Trade danışmanısın.
<user_profile>{personalization}</user_profile>
<graph_rag>{graph_context}</graph_rag>
<calculations>{calculations}</calculations>
<technical_data ticker="{ticker}">{tech_signals}</technical_data>
Piyasa: {market_sentiment}
{fact_check}
<strategies>{context}</strategies>
AL/SAT/BEKLE öner:"""

    def analyze_trade(self, ticker: str, tech_signals: str, market_sentiment: str = "Nötr"):
        """
        Hedge Fund AI Level Trading Analizi.
        
        Pipeline: Semantic Router → Adaptive RAG → Hybrid Search → 
        Bi-Encoder → C-RAG → Cross-Encoder → Parent Fetch → 
        Trade Rules → Python REPL → Graph-RAG → Memory → LLM
        """
        base_query = f"{ticker} için {tech_signals} verileri ile hangi strateji kullanılmalı?"
        
        # 1. Router
        if SemanticRouter.classify(base_query) == QueryType.GENERAL:
            return "Merhaba! 👋 Ben NeuralTrade AI Lab. Trading sorunuz var mı? 📈"
        
        # 2. Adaptive RAG
        complexity = AdaptiveRAG.assess(base_query)
        strategy = AdaptiveRAG.get_strategy(complexity)
        
        print(f"\n{Fore.YELLOW}[AI] {strategy['desc']} - {complexity.value.upper()}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}[AI] {ticker} için AI LAB analiz başlatılıyor...{Style.RESET_ALL}", flush=True)
        
        # 3. Hybrid Search
        docs = self.vectorstore.as_retriever(search_kwargs={"k": strategy["top_k"]}).invoke(base_query)
        print(f"{Fore.GREEN}  → {len(docs)} döküman bulundu{Style.RESET_ALL}", flush=True)
        
        # 4. Bi-Encoder Reranking
        if len(docs) > 1:
            passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
            rerank = self.bi_encoder.rerank(RerankRequest(query=base_query, passages=passages))
            docs = [docs[r["id"]] for r in rerank[:7]]
            print(f"{Fore.GREEN}  → Bi-Encoder: {len(docs)} döküman{Style.RESET_ALL}", flush=True)
        
        # 5. C-RAG (Alaka Kontrolü)
        web_news = ""
        if strategy["use_crag"]:
            print(f"{Fore.MAGENTA}  → C-RAG: Alaka kontrolü...{Style.RESET_ALL}", flush=True)
            original_count = len(docs)
            docs = self.crag.filter_documents(base_query, docs)
            print(f"{Fore.GREEN}  → C-RAG: {len(docs)} alakalı döküman{Style.RESET_ALL}", flush=True)
            
            if len(docs) < original_count * 0.3:
                web_news = self.crag.web_search_fallback(ticker, base_query)
        
        # 6. Cross-Encoder Reranking
        if strategy["use_cross"] and self.cross_encoder.available:
            print(f"{Fore.MAGENTA}  → Cross-Encoder: Ultra hassas sıralama...{Style.RESET_ALL}", flush=True)
            docs = self.cross_encoder.rerank(base_query, docs, top_k=3)
            print(f"{Fore.GREEN}  → Cross-Encoder: {len(docs)} döküman{Style.RESET_ALL}", flush=True)
        else:
            docs = docs[:5]
        
        # 7. Parent Fetch
        docs = self._batch_fetch_parents(docs)
        
        # Context
        context = "\n\n---\n\n".join([f"[{d.metadata.get('category', '?')}] {d.page_content[:800]}" for d in docs[:3]])
        
        # 8. Trade Rule Extraction
        trade_rules = ""
        current_price = 0.0
        current_rsi = 40.0
        
        price_match = re.search(r'Fiyat:\s*([\d.]+)', tech_signals)
        rsi_match = re.search(r'RSI.*?:\s*([\d.]+)', tech_signals)
        if price_match:
            current_price = float(price_match.group(1))
        if rsi_match:
            current_rsi = float(rsi_match.group(1))
        
        if current_price > 0:
            trade_rules = self.rule_extractor.extract_trade_rules(docs, ticker, current_price)
            trade_rules += self.rule_extractor.calculate_levels(current_price, current_rsi)
            print(f"{Fore.GREEN}  → Trade Rules: Entry/Exit seviyeleri hesaplandı{Style.RESET_ALL}", flush=True)
        
        # 9. Python REPL
        calculations = self._execute_calculation(base_query, tech_signals)
        
        # 10. Graph-RAG
        graph_context = self.graph.get_graph_context(base_query)
        if graph_context:
            print(f"{Fore.MAGENTA}  → Graph-RAG: Ekonomik ilişkiler bulundu{Style.RESET_ALL}", flush=True)
        
        # 11. Memory
        personalization = self.memory.get_personalization_context(self.user_id)
        print(f"{Fore.CYAN}  → SQLite Memory: Profil yüklendi{Style.RESET_ALL}", flush=True)
        
        # 12. Fact-Check
        fact_check = ""
        if YFINANCE_AVAILABLE and ticker not in ["GENEL", "EURUSD"]:
            try:
                info = yf.Ticker(ticker).info
                price = info.get("regularMarketPrice", info.get("currentPrice"))
                if price:
                    change = ((price - info.get("previousClose", price)) / info.get("previousClose", price) * 100)
                    fact_check = f"\n✅ CANLI: {ticker} ${price:.2f} ({change:+.2f}%)\n"
                    print(f"{Fore.GREEN}  → Fact-Check: ${price:.2f}{Style.RESET_ALL}", flush=True)
            except:
                pass
        
        # Add context
        if web_news:
            context = web_news + "\n" + context
        if trade_rules:
            context = trade_rules + "\n" + context
        
        # 13. Backtest Learning
        past_performance = self.memory.analyze_past_performance(self.user_id)
        if past_performance:
            print(f"{Fore.CYAN}  → Backtest Learning: Geçmiş dersler yüklendi{Style.RESET_ALL}", flush=True)
        
        # 14. Agentic Tool Planning
        if complexity == QueryComplexity.COMPLEX:
            planned_tools = self.agentic.plan_tools(base_query)
            print(f"{Fore.MAGENTA}  → Agentic: Planlanan araçlar: {planned_tools}{Style.RESET_ALL}", flush=True)
        
        # 15. LLM Generation
        template = self._get_prompt(complexity == QueryComplexity.COMPLEX)
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = ({
            "context": lambda x: context,
            "ticker": lambda x: ticker,
            "tech_signals": lambda x: tech_signals,
            "market_sentiment": lambda x: market_sentiment,
            "fact_check": lambda x: fact_check,
            "personalization": lambda x: personalization,
            "graph_context": lambda x: graph_context,
            "calculations": lambda x: calculations,
            "past_performance": lambda x: past_performance
        } | prompt | self.llm | StrOutputParser())
        
        try:
            print(f"{Fore.YELLOW}  → LLM: AI Lab analiz...{Style.RESET_ALL}", flush=True)
            response = chain.invoke(base_query)
            
            action = "BEKLE"
            if "AL" in response.upper():
                action = "AL"
            elif "SAT" in response.upper():
                action = "SAT"
            
            self.memory.add_recommendation(self.user_id, ticker, action, 85)
            
            print(f"{Fore.GREEN}✅ AI Lab analiz tamamlandı{Style.RESET_ALL}", flush=True)
            return response
        except Exception as e:
            return f"❌ Hata: {str(e)}"


# ============================================
# HELPER FUNCTIONS
# ============================================
def set_user_preference(advisor: AIAdvisor, key: str, value: str):
    """Kullanıcı tercihini güncelle."""
    advisor.memory.update_preference(advisor.user_id, key, value)
    print(f"{Fore.GREEN}✅ Tercih güncellendi: {key} = {value}{Style.RESET_ALL}")
