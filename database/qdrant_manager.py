"""
Qdrant Vector Database Yönetim Modülü
Embeddings ve similarity search için optimize edilmiş
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    QDRANT_MODE,
    QDRANT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_VECTOR_SIZE
)


class QdrantManager:
    """Qdrant vektör veritabanı yönetimi"""

    @staticmethod
    def make_point_id(*parts: object) -> str:
        """Stable UUID for idempotent MongoDB <-> Qdrant synchronization."""
        key = "::".join(str(part) for part in parts if part is not None)
        return str(uuid5(NAMESPACE_URL, f"ragora:{key}"))
    
    def __init__(self):
        """Qdrant client'ı başlat"""
        try:
            # MODE'a göre bağlantı türünü seç
            if QDRANT_MODE == "local":
                # Local file-based mode (servis gerektirmez)
                self.client = QdrantClient(path=str(QDRANT_PATH))
                print(f"✅ Qdrant (LOCAL MODE): {QDRANT_PATH}")
            else:
                # Server mode (network üzerinden)
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                print(f"✅ Qdrant (SERVER MODE): {QDRANT_HOST}:{QDRANT_PORT}")
            
            self.collection_name = QDRANT_COLLECTION
            
            # Collection yoksa oluştur
            self._ensure_collection()
        
        except Exception as e:
            print(f"❌ Qdrant bağlantısı başarısız: {e}")
            raise
    
    def _ensure_collection(self):
        """Collection'ın var olduğundan emin ol"""
        try:
            # Önce collection var mı kontrol et
            try:
                self.client.get_collection(self.collection_name)
                print(f"✅ Collection mevcut: {self.collection_name}")
                return
            except:
                # Collection yok, oluştur
                pass
            
            print(f"🔄 Qdrant collection oluşturuluyor: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection oluşturuldu: {self.collection_name}")
        
        except Exception as e:
            print(f"❌ Collection kontrolü hatası: {e}")
            raise
    
    def insert_vector(
        self,
        vector: List[float],
        mongodb_id: str,
        category: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Tek bir vektör ekle
        
        Args:
            vector: Embedding vektörü
            mongodb_id: MongoDB'deki doküman ID'si
            category: Kategori
            source: Kaynak URL
            metadata: Ek metadata
            
        Returns:
            Qdrant point ID
        """
        try:
            point_id = self.make_point_id(self.collection_name, mongodb_id, "text")
            
            payload = {
                "mongodb_id": mongodb_id,
                "category": category,
                "source": source
            }
            
            if metadata:
                payload.update(metadata)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            return point_id
        
        except Exception as e:
            print(f"❌ Vektör ekleme hatası: {e}")
            raise
    
    def insert_many_vectors(
        self,
        vectors: List[Dict]
    ) -> List[str]:
        """
        Toplu vektör ekleme
        
        Args:
            vectors: [{"vector": [...], "mongodb_id": "...", "category": "...", "source": "...", "metadata": {...}}]
            
        Returns:
            Qdrant point ID'leri
        """
        try:
            points = []
            point_ids = []
            
            for vec_data in vectors:
                point_id = vec_data.get("point_id") or self.make_point_id(
                    self.collection_name,
                    vec_data["mongodb_id"],
                    vec_data.get("start_index", ""),
                    "text"
                )
                point_ids.append(point_id)
                
                payload = {
                    "mongodb_id": vec_data["mongodb_id"],
                    "category": vec_data["category"],
                    "source": vec_data["source"]
                }
                
                # ⭐ Chunk content ve start_index ekle
                if "chunk_content" in vec_data:
                    payload["chunk_content"] = vec_data["chunk_content"]
                if "start_index" in vec_data:
                    payload["start_index"] = vec_data["start_index"]
                
                if "metadata" in vec_data:
                    payload.update(vec_data["metadata"])
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec_data["vector"],
                        payload=payload
                    )
                )
            
            # Toplu ekleme
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return point_ids
        
        except Exception as e:
            print(f"❌ Toplu vektör ekleme hatası: {e}")
            raise
    
    def search_similar(
        self,
        query_vector: List[float],
        category: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.65
    ) -> List[Tuple[Dict, float]]:
        """
        Benzer vektörleri ara
        
        Args:
            query_vector: Sorgu vektörü
            category: Opsiyonel kategori filtresi
            limit: Maksimum sonuç sayısı
            score_threshold: Minimum benzerlik skoru
            
        Returns:
            [(payload, score), ...] listesi
        """
        try:
            # Kategori filtresi
            query_filter = None
            if category:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category)
                        )
                    ]
                )
            
            # Arama yap - Qdrant 1.17.0 API'si
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Sonuçları formatla
            formatted_results = []
            for result in results.points:
                formatted_results.append((result.payload, result.score))
            
            return formatted_results
        
        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []
    
    def delete_by_category(self, category: str, collection_name: Optional[str] = None) -> int:
        """
        Kategoriye göre vektörleri sil
        
        Args:
            category: Silinecek kategori
            collection_name: Silme işleminin yapılacağı collection (verilmezse varsayılan collection kullanılır)
            
        Returns:
            Silinen vektör sayısı
        """
        try:
            target_collection = collection_name or self.collection_name

            # Önce kategoriyi say
            count_result = self.client.count(
                collection_name=target_collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category)
                        )
                    ]
                )
            )
            
            count = count_result.count
            
            if count > 0:
                # Sil
                self.client.delete(
                    collection_name=target_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="category",
                                match=MatchValue(value=category)
                            )
                        ]
                    )
                )
            
            return count
        
        except Exception as e:
            print(f"❌ Silme hatası: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Collection istatistiklerini al
        
        Returns:
            İstatistik bilgileri
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "total_vectors": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
        
        except Exception as e:
            print(f"❌ İstatistik alma hatası: {e}")
            return {}
    
    def delete_collection(self):
        """Collection'ı sil"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Collection silindi: {self.collection_name}")
        except Exception as e:
            print(f"❌ Collection silme hatası: {e}")
    
    def close(self):
        """Client'ı kapat"""
        # Qdrant client otomatik kapanıyor
        pass


# Test için
if __name__ == "__main__":
    print("🔄 Qdrant Manager Test...")
    
    try:
        qm = QdrantManager()
        
        # İstatistikleri göster
        stats = qm.get_stats()
        print(f"\n📊 Qdrant İstatistikleri:")
        print(f"  Toplam Vektör: {stats.get('total_vectors', 0)}")
        print(f"  Vektör Boyutu: {stats.get('vector_size', 0)}")
        print(f"  Distance Metrik: {stats.get('distance', 'N/A')}")
        
        qm.close()
        print("\n✅ Test başarılı!")
    
    except Exception as e:
        print(f"\n❌ Test başarısız: {e}")
