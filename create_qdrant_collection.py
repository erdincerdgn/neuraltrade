#!/usr/bin/env python3
"""Qdrant collection oluşturma scripti"""
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def create_collection():
    """Qdrant neural_trade_pro collection oluşturur"""
    try:
        # Qdrant bağlantısı
        client = QdrantClient(url="http://localhost:6333")
        
        collection_name = "neural_trade_pro"
        
        # Collection var mı kontrol et
        try:
            collections = client.get_collections()
            existing = [c.name for c in collections.collections]
            if collection_name in existing:
                print(f"✅ Collection '{collection_name}' zaten mevcut!")
                return True
        except Exception as e:
            print(f"ℹ️  Collection kontrolü yapılamadı: {e}")
        
        # Collection oluştur
        print(f"🔧 Collection '{collection_name}' oluşturuluyor...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # FastEmbed BAAI/bge-small-en-v1.5 boyutu
                distance=Distance.COSINE
            )
        )
        
        print(f"✅ Collection '{collection_name}' başarıyla oluşturuldu!")
        print(f"📊 Qdrant Dashboard: http://localhost:6333/dashboard")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    success = create_collection()
    sys.exit(0 if success else 1)
