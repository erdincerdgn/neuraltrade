"""
Enterprise RAG System Test Suite
Author: Erdinc Erdogan
Purpose: Tests FlashRank reranking, Parent-Document retrieval, metadata filtering, and hybrid search (Dense+Sparse) for knowledge base integration.
References:
- FlashRank Reranking (40% hallucination reduction)
- Parent-Document Retrieval (60% context quality improvement)
- Qdrant Hybrid Search
Usage:
    python test_rag.py
    # Validates RAG pipeline with Qdrant and Ollama
"""

import os
import re
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from colorama import Fore, Style, init
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangChain Modülleri
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# FlashRank Reranking (Doğrudan kullanım)
from flashrank import Ranker, RerankRequest

# Renkli loglama
init(autoreset=True)

def debug_log(message, color=Fore.CYAN):
    print(f"{color}[RAG] {message}{Style.RESET_ALL}", flush=True)

# ============================================
# YAPILANDIRMA
# ============================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-service:6333")
UNSTRUCTURED_URL = os.getenv("UNSTRUCTURED_API_URL", "http://unstructured-api:8000/general/v0/general")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-service:11434")

# Koleksiyon adları
COLLECTION_NAME = "neural_trade_pro"           # Child chunks (arama için)
PARENT_COLLECTION_NAME = "neural_trade_parent" # Parent docs (bağlam için)

# Embedding boyutu
DENSE_VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5

# Knowledge base
KNOWLEDGE_BASE_PATH = "/app/knowledge_base/books"

# ============================================
# METADATA EXTRACTOR
# ============================================
def extract_book_metadata(pdf_path: str) -> Dict:
    """
    PDF dosya adından metadata çıkarır.
    Örnek: currency-trading-for-dummies.pdf → category: Forex, author: Dummies
    """
    filename = os.path.basename(pdf_path).lower()
    
    # Kategori tespiti
    category = "General"
    if any(word in filename for word in ["currency", "forex", "fx"]):
        category = "Forex"
    elif any(word in filename for word in ["stock", "equity", "share"]):
        category = "Stocks"
    elif any(word in filename for word in ["crypto", "bitcoin", "blockchain"]):
        category = "Crypto"
    elif any(word in filename for word in ["option", "derivative", "future"]):
        category = "Derivatives"
    elif any(word in filename for word in ["technical", "analysis", "chart"]):
        category = "Technical Analysis"
    
    # Yazar tespiti
    author = "Unknown"
    if "dummies" in filename:
        author = "For Dummies Series"
    elif "art-of" in filename:
        author = "Expert Author"
    
    # Yıl (dosya adından veya varsayılan)
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else 2024
    
    return {
        "category": category,
        "author": author,
        "year": year,
        "source_file": os.path.basename(pdf_path),
        "indexed_at": datetime.now().isoformat()
    }


# ============================================
# PARENT-DOCUMENT RETRIEVAL
# ============================================
class ParentDocumentStore:
    """
    Parent-Document Retrieval için depolama.
    Küçük chunk'ları arar, ancak LLM'e parent document'ı gönderir.
    """
    
    def __init__(self, client: QdrantClient, dense_emb, sparse_emb):
        self.client = client
        self.dense_embeddings = dense_emb
        self.sparse_embeddings = sparse_emb
        self.parent_docs: Dict[str, Document] = {}  # parent_id -> full document
        
    def setup_collections(self, force_recreate: bool = False):
        """Child ve Parent koleksiyonlarını oluşturur."""
        
        for coll_name in [COLLECTION_NAME, PARENT_COLLECTION_NAME]:
            exists = self._collection_exists(coll_name)
            
            if exists and not force_recreate:
                debug_log(f"✅ Mevcut koleksiyon: {coll_name}")
                continue
                
            if exists:
                self.client.delete_collection(coll_name)
                debug_log(f"🗑️ Silindi: {coll_name}", Fore.YELLOW)
            
            self.client.create_collection(
                collection_name=coll_name,
                vectors_config={
                    "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                }
            )
            debug_log(f"✅ Oluşturuldu: {coll_name}", Fore.GREEN)
            
        return force_recreate or not self._collection_exists(COLLECTION_NAME)
    
    def _collection_exists(self, name: str) -> bool:
        try:
            info = self.client.get_collection(name)
            return info.points_count > 0
        except:
            return False
    
    def add_documents(self, pdf_path: str, child_chunk_size: int = 400, parent_chunk_size: int = 1500):
        """
        PDF'i hem child hem parent olarak indeksler.
        
        Strategy:
        - Child chunks (400 char): Arama için - daha hassas eşleşme
        - Parent chunks (1500 char): Bağlam için - daha geniş görüş
        """
        debug_log(f"📄 İşleniyor: {os.path.basename(pdf_path)}")
        
        # Metadata çıkar
        metadata = extract_book_metadata(pdf_path)
        
        # PDF'i parse et
        loader = UnstructuredAPIFileLoader(url=UNSTRUCTURED_URL, file_path=pdf_path, mode="elements")
        raw_docs = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in raw_docs])
        
        # Parent chunks oluştur (büyük parçalar)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "]
        )
        parent_chunks = parent_splitter.split_text(full_text)
        
        # Child chunks oluştur (küçük parçalar)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        child_docs = []
        parent_docs_list = []
        
        for i, parent_text in enumerate(parent_chunks):
            parent_id = str(uuid.uuid4())
            
            # Parent document
            parent_doc = Document(
                page_content=parent_text,
                metadata={**metadata, "doc_type": "parent", "parent_id": parent_id}
            )
            parent_docs_list.append(parent_doc)
            self.parent_docs[parent_id] = parent_doc
            
            # Child chunks (parent'a referans ile)
            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_doc = Document(
                    page_content=child_text,
                    metadata={**metadata, "doc_type": "child", "parent_id": parent_id}
                )
                child_docs.append(child_doc)
        
        debug_log(f"  → {len(parent_chunks)} parent, {len(child_docs)} child chunk")
        
        # Child'ları Qdrant'a ekle
        child_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Batch halinde ekle
        batch_size = 50
        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i:i+batch_size]
            child_store.add_documents(documents=batch)
        
        # Parent'ları da Qdrant'a ekle (metadata araması için)
        parent_store = QdrantVectorStore(
            client=self.client,
            collection_name=PARENT_COLLECTION_NAME,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        for i in range(0, len(parent_docs_list), batch_size):
            batch = parent_docs_list[i:i+batch_size]
            parent_store.add_documents(documents=batch)
        
        debug_log(f"  ✅ İndekslendi: {metadata['category']}", Fore.GREEN)
        
        return len(child_docs)


# ============================================
# PROFESSIONAL RAG PIPELINE
# ============================================
class ProfessionalRAG:
    """
    Enterprise-grade RAG sistemi.
    Features:
    - Hybrid Search (Dense + Sparse)
    - FlashRank Reranking  
    - Parent-Document Retrieval
    - Metadata Filtering
    """
    
    def __init__(self):
        debug_log("🚀 Professional RAG başlatılıyor...")
        
        # Embeddings
        self.dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Qdrant
        self.client = QdrantClient(url=QDRANT_URL)
        
        # Parent Document Store
        self.parent_store = ParentDocumentStore(
            self.client, 
            self.dense_embeddings, 
            self.sparse_embeddings
        )
        
        # LLM
        self.llm = OllamaLLM(model="llama3", base_url=OLLAMA_HOST)
        
        # FlashRank Reranker (Doğrudan kullanım)
        self.reranker = Ranker(model_name="ms-marco-MultiBERT-L-12")  # Hafif ve hızlı model
        
        debug_log("✅ Tüm bileşenler hazır", Fore.GREEN)
    
    def build_knowledge_base(self, force_recreate: bool = False):
        """Tüm kitapları indeksler."""
        
        needs_indexing = self.parent_store.setup_collections(force_recreate)
        
        if not needs_indexing:
            debug_log("📚 Mevcut knowledge base kullanılıyor")
            return
        
        books_path = Path(KNOWLEDGE_BASE_PATH)
        pdf_files = list(books_path.glob("*.pdf"))
        
        if not pdf_files:
            debug_log("❌ PDF bulunamadı!", Fore.RED)
            return
        
        debug_log(f"📚 {len(pdf_files)} kitap indeksleniyor...")
        
        total_chunks = 0
        for pdf_file in pdf_files:
            chunks = self.parent_store.add_documents(str(pdf_file))
            total_chunks += chunks
        
        debug_log(f"✅ Toplam {total_chunks} chunk indekslendi", Fore.GREEN)
    
    def get_retriever(self, category_filter: str = None, top_k: int = 10):
        """
        Reranking destekli retriever döndürür.
        Opsiyonel metadata filtreleme.
        """
        
        # Base retriever (Hybrid Search)
        child_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Metadata filter oluştur
        search_kwargs = {"k": top_k}
        
        if category_filter:
            search_kwargs["filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchValue(value=category_filter)
                    )
                ]
            )
            debug_log(f"🏷️ Kategori filtresi: {category_filter}")
        
        base_retriever = child_store.as_retriever(search_kwargs=search_kwargs)
        
        return base_retriever  # Reranking query() içinde yapılacak
    
    def expand_to_parent(self, docs: List[Document]) -> List[Document]:
        """
        Child chunk'ları parent document'lara genişletir.
        Daha geniş bağlam sağlar.
        """
        expanded = []
        seen_parents = set()
        
        for doc in docs:
            parent_id = doc.metadata.get("parent_id")
            
            if parent_id and parent_id not in seen_parents:
                # Parent'ı Qdrant'tan çek
                try:
                    results = self.client.scroll(
                        collection_name=PARENT_COLLECTION_NAME,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.parent_id",
                                    match=models.MatchValue(value=parent_id)
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=True
                    )
                    
                    if results[0]:
                        point = results[0][0]
                        parent_doc = Document(
                            page_content=point.payload.get("page_content", doc.page_content),
                            metadata=point.payload.get("metadata", doc.metadata)
                        )
                        expanded.append(parent_doc)
                        seen_parents.add(parent_id)
                except:
                    # Fallback: child'ın kendisini kullan
                    expanded.append(doc)
            else:
                if parent_id not in seen_parents:
                    expanded.append(doc)
        
        return expanded if expanded else docs
    
    def query(self, question: str, category: str = None, use_parent: bool = True) -> str:
        """
        Profesyonel RAG sorgusu.
        
        Pipeline:
        1. Hybrid Search (Top 10)
        2. FlashRank Reranking (Top 3)
        3. Parent Expansion (opsiyonel)
        4. LLM Generation
        """
        
        debug_log(f"🔍 Sorgu: {question[:50]}...")
        
        # 1. Retriever al (reranking dahil)
        retriever = self.get_retriever(category_filter=category)
        
        # 2. Dökümanları getir
        docs = retriever.invoke(question)
        debug_log(f"  → {len(docs)} döküman bulundu")
        
        # 3. FlashRank ile rerank
        if docs and len(docs) > 1:
            passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
            rerank_request = RerankRequest(query=question, passages=passages)
            rerank_results = self.reranker.rerank(rerank_request)
            
            # En iyi 3'ü seç
            top_indices = [r["id"] for r in rerank_results[:3]]
            docs = [docs[i] for i in top_indices]
            debug_log(f"  → {len(docs)} döküman reranked")
        
        # 3. Parent expansion
        if use_parent:
            docs = self.expand_to_parent(docs)
            debug_log(f"  → {len(docs)} parent document'a genişletildi")
        
        # 4. Bağlam oluştur
        context = "\n\n---\n\n".join([
            f"[{doc.metadata.get('category', 'Unknown')}] {doc.page_content}"
            for doc in docs
        ])
        
        # 5. LLM ile yanıt üret
        template = """Sen uzman bir finansal danışmansın. Aşağıdaki bağlamı kullanarak soruyu Türkçe yanıtla.
        
BAĞLAM:
{context}

SORU: {question}

ÖNEMLİ: 
- Sadece bağlamdaki bilgileri kullan
- Somut stratejiler ve örnekler ver
- Eğer bilmiyorsan "Bu konuda bilgim yok" de

CEVAP:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(question)
        
        return response


# ============================================
# ANA GİRİŞ NOKTASI
# ============================================
def run_professional_rag(force_recreate: bool = False, query: str = None, category: str = None):
    """Ana RAG fonksiyonu."""
    
    try:
        debug_log("=" * 50, Fore.YELLOW)
        debug_log("🚀 ENTERPRISE-GRADE RAG SİSTEMİ", Fore.YELLOW)
        debug_log("=" * 50, Fore.YELLOW)
        
        # RAG sistemi oluştur
        rag = ProfessionalRAG()
        
        # Knowledge base kur
        rag.build_knowledge_base(force_recreate)
        
        # Test sorgusu
        if query:
            debug_log(f"\n🔍 SORGULANIYOR: {query}", Fore.MAGENTA)
            if category:
                debug_log(f"🏷️ Kategori: {category}", Fore.MAGENTA)
            
            response = rag.query(query, category=category)
            
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"🎯 NEURALTRADE PRO RAG YANITI")
            print(f"{'='*60}{Style.RESET_ALL}")
            print(response)
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
        
        debug_log("✅ Enterprise RAG sistemi hazır!", Fore.GREEN)
        
        return rag
        
    except Exception as e:
        debug_log(f"❌ HATA: {str(e)}", Fore.RED)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    force = "--force" in sys.argv or "-f" in sys.argv
    
    # Kategori filtresi (opsiyonel)
    category = None
    for arg in sys.argv:
        if arg.startswith("--category="):
            category = arg.split("=")[1]
    
    test_query = "RSI ve Ichimoku göstergelerini kullanarak Forex piyasasında nasıl trade yapılır? Somut strateji örnekleri ver."
    
    run_professional_rag(
        force_recreate=force,
        query=test_query,
        category=category
    )