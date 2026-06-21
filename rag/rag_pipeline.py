"""
RAG (Retrieval-Augmented Generation) Pipeline
MongoDB (metadata) + Qdrant (vectors) + Llama ile TÃ¼rkÃ§e Soru-Cevap Sistemi
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from bson import ObjectId

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    MAX_CONTEXT_CHARS,
    TURKISH_QA_PROMPT
)
from database.mongodb_manager import MongoDBManager
from database.qdrant_manager import QdrantManager
from rag.context_safety import build_untrusted_context_block, sanitize_retrieved_text
from rag.model_utils import strip_thinking, with_no_think


class RAGSystem:
    """RAG tabanlÄ± soru-cevap sistemi"""
    
    def __init__(self):
        """RAG sistemini baÅŸlat"""
        print("ğŸ”„ RAG sistemi baÅŸlatÄ±lÄ±yor...")
        
        # Embedding modeli
        print("  - Embedding modeli yÃ¼kleniyor...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # MongoDB baÄŸlantÄ±sÄ± (metadata, images)
        print("  - MongoDB baÄŸlanÄ±lÄ±yor...")
        self.db = MongoDBManager()
        
        # Qdrant baÄŸlantÄ±sÄ± (vector search)
        print("  - Qdrant baÄŸlanÄ±lÄ±yor...")
        self.vector_db = QdrantManager()
        
        # Llama LLM (Ollama Ã¼zerinden)
        print("  - Llama LLM baÄŸlanÄ±lÄ±yor...")
        try:
            self.llm = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.0,
                num_predict=450,
            )
            print("âœ… RAG sistemi hazÄ±r")
        except Exception as e:
            print(f"âš ï¸ Ollama baÄŸlantÄ± hatasÄ±: {e}")
            print("  Ollama'nÄ±n Ã§alÄ±ÅŸtÄ±ÄŸÄ±ndan emin olun: ollama serve")
            print("  Model indirin: ollama pull llama3.1:8b")
            self.llm = None
        
        # Prompt ÅŸablonu
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=TURKISH_QA_PROMPT
        )
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        category: str = None,
        k: int = RETRIEVAL_K
    ) -> List[Tuple[Dict, float]]:
        """
        Sorguya en uygun chunk'larÄ± getir (Qdrant'tan direkt chunk content kullanarak)
        
        Args:
            query: KullanÄ±cÄ± sorusu
            category: Opsiyonel kategori filtresi
            k: Getirilecek chunk sayÄ±sÄ±
            
        Returns:
            (chunk_dict, skor) tuple'larÄ±nÄ±n listesi
        """
        # Sorgu iÃ§in embedding oluÅŸtur
        query_embedding = self.embeddings.embed_query(query)
        
        # Qdrant'tan benzer vektÃ¶rleri ara
        similar_vectors = self.vector_db.search_similar(
            query_vector=query_embedding,
            category=category,
            limit=max(k * 3, k),
            score_threshold=SIMILARITY_THRESHOLD
        )
        
        if not similar_vectors:
            print("âš ï¸ Ä°lgili dokÃ¼man bulunamadÄ±!")
            return []
        
        # Qdrant payload'Ä±ndan chunk content'i direkt al (MongoDB'ye gitme)
        scored_docs = []
        seen_keys = set()
        for payload, score in similar_vectors:
            # â­ Qdrant'tan chunk content'i al
            chunk_content = payload.get("chunk_content")
            
            # EÄŸer chunk_content yoksa (eski veri), MongoDB'den al
            if not chunk_content:
                mongodb_id = payload.get("mongodb_id")
                if mongodb_id:
                    doc = self.db.collection.find_one({"_id": ObjectId(mongodb_id)})
                    if doc:
                        chunk_content = doc.get("content", "")
            
            images = payload.get("images", [])
            if not images:
                mongodb_id = payload.get("mongodb_id")
                if mongodb_id:
                    try:
                        doc = self.db.collection.find_one({"_id": ObjectId(mongodb_id)})
                        if doc:
                            images = doc.get("images", [])
                    except Exception:
                        images = []

            if chunk_content:
                normalized_source = (payload.get("source") or "").rstrip("/")
                start_index = payload.get("start_index", 0)
                mongodb_id = payload.get("mongodb_id")
                content_fingerprint = chunk_content[:500].strip()
                dedupe_key = (normalized_source, start_index, content_fingerprint)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                # Chunk'Ä± mock document biÃ§iminde oluÅŸtur
                chunk_dict = {
                    "content": chunk_content,
                    "source": payload.get("source", "Bilinmiyor"),
                    "category": payload.get("category", "Bilinmiyor"),
                    "start_index": start_index,
                    "mongodb_id": mongodb_id,
                    "image_hashes": payload.get("image_hashes", []),
                    "images": images
                }
                scored_docs.append((chunk_dict, score))
                if len(scored_docs) >= k:
                    break
        
        return scored_docs
    
    def generate_answer(self, query: str, category: str = None) -> Dict:
        """
        Soru iÃ§in RAG pipeline'Ä± Ã§alÄ±ÅŸtÄ±r ve cevap Ã¼ret
        
        Args:
            query: KullanÄ±cÄ± sorusu
            category: Opsiyonel kategori filtresi
            
        Returns:
            Cevap bilgileri iÃ§eren dictionary
        """
        if not self.llm:
            return {
                "answer": "LLM servisi aktif deÄŸil. LÃ¼tfen Ollama'yÄ± baÅŸlatÄ±n.",
                "sources": [],
                "error": True
            }
        
        # 1. Ä°lgili dokÃ¼manlarÄ± getir
        print(f"\nğŸ” Soru: {query}")
        print("  DokÃ¼manlar aranÄ±yor...")
        
        relevant_docs = self.retrieve_relevant_documents(query, category)
        
        if not relevant_docs:
            return {
                "answer": "Bu sorunun cevabÄ±nÄ± mevcut dokÃ¼manlarda bulamadÄ±m. LÃ¼tfen sorunuzu yeniden formÃ¼le etmeyi deneyin.",
                "sources": [],
                "error": False
            }
        
        print(f"  âœ… {len(relevant_docs)} ilgili dokÃ¼man bulundu")
        
        # 2. BaÄŸlam oluÅŸtur
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(relevant_docs, 1):
            safe_content = sanitize_retrieved_text(doc.get("content", ""))
            context_parts.append(f"Dokuman {i} (Benzerlik: {score:.2f}):\n{safe_content}")
            sources.append({
                "source": doc.get("source", "Bilinmiyor"),
                "category": doc.get("category", "Bilinmiyor"),
                "similarity": float(score),
                "content_preview": safe_content[:200] + "...",
                "images": doc.get("images", [])  # GÃ¶rselleri ekle
            })
        
        context = build_untrusted_context_block(
            context_parts,
            max_chars=MAX_CONTEXT_CHARS,
            label="Technical Context",
        )
        
        # 3. LLM ile cevap Ã¼ret
        print("  ğŸ¤– Cevap oluÅŸturuluyor...")
        
        try:
            # Prompt'u formatla
            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )
            formatted_prompt = with_no_think(formatted_prompt)
            
            # LLM'den cevap al
            answer = self.llm.invoke(formatted_prompt)
            answer = strip_thinking(answer).strip()
            
            return {
                "answer": answer,
                "sources": sources,
                "error": False
            }
        
        except Exception as e:
            print(f"  âŒ LLM hatasÄ±: {e}")
            return {
                "answer": f"Cevap Ã¼retilirken bir hata oluÅŸtu: {str(e)}",
                "sources": sources,
                "error": True
            }
    
    def close(self):
        """BaÄŸlantÄ±larÄ± kapat"""
        self.db.close()


# Test fonksiyonu
if __name__ == "__main__":
    # RAG sistemini test et
    rag = RAGSystem()
    
    # Test sorularÄ±
    test_questions = [
        "Warthog robotunun bakÄ±m prosedÃ¼rleri nelerdir?",
        "Warthog bataryasÄ± nasÄ±l ÅŸarj edilir?",
        "Warthog'un gÃ¼venlik Ã¶zellikleri neler?",
    ]
    
    print("\n" + "="*60)
    print("ğŸ§ª RAG SÄ°STEMÄ° TEST")
    print("="*60)
    
    for question in test_questions:
        result = rag.generate_answer(question)
        
        print(f"\n{'='*60}")
        print(f"â“ SORU: {question}")
        print(f"{'='*60}")
        print(f"\nğŸ’¡ CEVAP:\n{result['answer']}")
        
        if result['sources']:
            print(f"\nğŸ“š KAYNAKLAR ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  {i}. {source['source']}")
                print(f"     Kategori: {source['category']}")
                print(f"     Benzerlik: {source['similarity']:.2%}")
        
        print("\n" + "="*60)
    
    rag.close()
