"""
MongoDB Veritabanı İşlemleri
"""
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION


class MongoDBManager:
    """MongoDB bağlantı ve işlemlerini yönetir"""
    
    def __init__(self):
        """MongoDB bağlantısını başlat"""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db[MONGODB_COLLECTION]
            
            # Bağlantıyı test et
            self.client.admin.command('ping')
            print("✅ MongoDB bağlantısı başarılı")
            
            # İndeksler oluştur
            self._create_indexes()
            
        except ConnectionFailure as e:
            print(f"❌ MongoDB bağlantı hatası: {e}")
            raise
    
    def _create_indexes(self):
        """Verimli sorgular için indeksler oluştur"""
        # Kategori indeksi
        self.collection.create_index("category")
        # Source URL indeksi
        self.collection.create_index("source")
        # Görsel ve chunk bağlantıları için ek indeksler
        self.collection.create_index("image_hashes")
        self.collection.create_index([("category", 1), ("source", 1), ("metadata.start_index", 1)])
        # Text search indeksi
        self.collection.create_index([("content", "text")])
    
    def insert_document(self, 
                       content: str, 
                       embedding: List[float],
                       category: str,
                       source: str,
                       metadata: Optional[Dict] = None) -> str:
        """
        Tek bir doküman ekle
        
        Args:
            content: Doküman içeriği
            embedding: Embedding vektörü
            category: Kategori adı (örn: "warthog")
            source: Kaynak URL
            metadata: Ek metadata bilgileri
            
        Returns:
            Eklenen dokümanın ID'si
        """
        document = {
            "content": content,
            "embedding": embedding,
            "category": category,
            "source": source,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many_documents(self, documents: List[Dict]) -> List[str]:
        """
        Birden fazla doküman ekle
        
        Args:
            documents: Doküman listesi
            
        Returns:
            Eklenen dokümanların ID listesi
        """
        timestamp = datetime.utcnow()
        
        # Timestamp ekle
        for doc in documents:
            # Mongo ID'sini insert öncesinde kesinleştir. Qdrant payload'ları bu
            # gerçek ObjectId ile oluşur; tahmini/geçici ID yazılmasını önler.
            doc.setdefault("_id", ObjectId())
            doc["created_at"] = timestamp
            doc["updated_at"] = timestamp
        
        result = self.collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def search_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """
        Kategoriye göre dokümanları getir
        
        Args:
            category: Kategori adı
            limit: Maksimum sonuç sayısı
            
        Returns:
            Doküman listesi
        """
        cursor = self.collection.find(
            {"category": category},
            {"_id": 0}
        ).limit(limit)
        
        return list(cursor)
    
    def get_all_documents(self, category: Optional[str] = None) -> List[Dict]:
        """
        Tüm dokümanları veya belirli kategorideki dokümanları getir
        
        Args:
            category: Opsiyonel kategori filtresi
            
        Returns:
            Doküman listesi
        """
        query = {"category": category} if category else {}
        cursor = self.collection.find(query, {"_id": 0})
        return list(cursor)
    
    def delete_by_category(self, category: str) -> int:
        """
        Belirli bir kategorideki tüm dokümanları sil
        
        Args:
            category: Kategori adı
            
        Returns:
            Silinen doküman sayısı
        """
        result = self.collection.delete_many({"category": category})
        return result.deleted_count
    
    def clear_all(self) -> int:
        """
        Tüm dokümanları sil
        
        Returns:
            Silinen doküman sayısı
        """
        result = self.collection.delete_many({})
        return result.deleted_count
    
    def get_stats(self) -> Dict:
        """
        Veritabanı istatistiklerini getir
        
        Returns:
            İstatistik bilgileri
        """
        pipeline = [
            {
                "$group": {
                    "_id": "$category",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        category_counts = list(self.collection.aggregate(pipeline))
        total_count = self.collection.count_documents({})
        
        return {
            "total_documents": total_count,
            "categories": {item["_id"]: item["count"] for item in category_counts}
        }
    
    def close(self):
        """MongoDB bağlantısını kapat"""
        self.client.close()
        print("🔌 MongoDB bağlantısı kapatıldı")


# Test fonksiyonu
if __name__ == "__main__":
    # MongoDB bağlantısını test et
    db = MongoDBManager()
    
    print("\n📊 Veritabanı İstatistikleri:")
    stats = db.get_stats()
    print(f"Toplam doküman: {stats['total_documents']}")
    print(f"Kategoriler: {stats['categories']}")
    
    db.close()
