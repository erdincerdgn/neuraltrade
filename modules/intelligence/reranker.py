"""
Cross-Encoder Document Reranker
Author: Erdinc Erdogan
Purpose: Reranks retrieved documents using cross-encoder models for ultra-precise relevance scoring in RAG pipelines.
References:
- Cross-Encoder Models (ms-marco-MiniLM)
- Sentence Transformers Library
- Two-Stage Retrieval Pattern
Usage:
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_docs = reranker.rerank(query, docs, top_k=3)
"""
from typing import List
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """Cross-Encoder ile ultra hassas sıralama."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
            except:
                self.available = False
        else:
            self.available = False
    
    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        """Dökümanları cross-encoder ile yeniden sırala."""
        if not self.available or not docs:
            return docs[:top_k]
        
        pairs = [[query, doc.page_content[:500]] for doc in docs]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
