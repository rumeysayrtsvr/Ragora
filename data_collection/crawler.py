"""
Veri Toplama Modülü - Kategorilere Göre Web ve PDF İçerikleri + Görseller
CLIP ile görsel eşleştirme ve akıllı filtreleme
Otomatik URL keşfi ile recursive crawling
"""
import sys
from pathlib import Path
import re
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin, urlparse
import requests
import time
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    CATEGORIES, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    IMAGE_MAX_PER_PAGE,
    IMAGE_EMBEDDING_SIZE,
    QDRANT_IMAGE_COLLECTION,
    AUTO_DISCOVER_LINKS,
    MAX_CRAWL_DEPTH,
    MAX_PAGES_PER_CATEGORY,
    CRAWL_DELAY,
    SKIP_URL_PATTERNS,
    ALLOWED_URL_PATTERNS
)
from database.mongodb_manager import MongoDBManager
from database.qdrant_manager import QdrantManager
from data_collection.image_utils import ImageProcessor


class DataCollector:
    """Kategorilere göre web ve PDF verilerini toplar ve işler"""
    
    def __init__(self):
        """Veri toplayıcıyı başlat"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ]
        )
        
        print("🔄 Embedding modeli yükleniyor...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        print("✅ Embedding modeli yüklendi")
        
        # Image processor (CLIP)
        print("🔄 Image processor başlatılıyor...")
        try:
            self.image_processor = ImageProcessor()
        except Exception as e:
            print(f"⚠️ Image processor yüklenemedi: {e}")
            print("  Görsel işleme devre dışı")
            self.image_processor = None
        
        # Veritabanları
        self.db = MongoDBManager()
        self.vector_db = QdrantManager()  # Text vectors
        
        # Image vector database (aynı client'ı paylaş - local modda concurrent access yok)
        if self.image_processor:
            try:
                # Mevcut client'ı kullan (yeni client oluşturma)
                self.image_vector_db = self.vector_db.client
                
                # Image collection oluştur
                from qdrant_client.models import VectorParams, Distance
                collections = self.image_vector_db.get_collections().collections
                if QDRANT_IMAGE_COLLECTION not in [c.name for c in collections]:
                    self.image_vector_db.create_collection(
                        collection_name=QDRANT_IMAGE_COLLECTION,
                        vectors_config=VectorParams(
                            size=IMAGE_EMBEDDING_SIZE,
                            distance=Distance.COSINE
                        )
                    )
                    print(f"✅ Image collection oluşturuldu: {QDRANT_IMAGE_COLLECTION}")
            except Exception as e:
                print(f"⚠️ Image vector DB başlatılamadı: {e}")
                self.image_vector_db = None
        else:
            self.image_vector_db = None
    
    def discover_urls(
        self, 
        base_url: str, 
        max_depth: int = MAX_CRAWL_DEPTH,
        max_pages: int = MAX_PAGES_PER_CATEGORY
    ) -> List[str]:
        """
        Base URL'den başlayarak otomatik alt sayfaları keşfet (recursive crawling)
        
        Args:
            base_url: Başlangıç URL'i
            max_depth: Maksimum derinlik seviyesi
            max_pages: Maksimum sayfa sayısı
            
        Returns:
            Keşfedilen URL listesi
        """
        discovered_urls = set()
        visited_urls = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        
        # Base URL'in domain ve path'ini al
        base_parsed = urlparse(base_url)
        base_domain = f"{base_parsed.scheme}://{base_parsed.netloc}"
        base_path = base_parsed.path.rstrip('/')
        
        print(f"\n🔍 Otomatik URL keşfi başlatılıyor...")
        print(f"   Base URL: {base_url}")
        print(f"   Max depth: {max_depth}, Max pages: {max_pages}")
        
        while to_visit and len(discovered_urls) < max_pages:
            current_url, depth = to_visit.pop(0)
            
            # Zaten ziyaret edildiyse atla
            if current_url in visited_urls:
                continue
            
            # Depth limiti aşıldıysa atla
            if depth > max_depth:
                continue
            
            visited_urls.add(current_url)
            discovered_urls.add(current_url)
            
            print(f"   [{len(discovered_urls)}/{max_pages}] Depth {depth}: {current_url[:70]}...")
            
            try:
                # Sayfayı indir
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                
                # Linkleri parse et
                soup = BeautifulSoup(response.content, "lxml")
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Tam URL'ye çevir
                    full_url = urljoin(current_url, href)
                    
                    # URL'i normalize et (trailing slash, fragment kaldır)
                    parsed = urlparse(full_url)
                    normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
                    
                    # Filtreleme kuralları
                    # 1. Zaten ziyaret edildiyse atla
                    if normalized_url in visited_urls or normalized_url in [u for u, _ in to_visit]:
                        continue
                    
                    # 2. Aynı domain değilse atla
                    if not normalized_url.startswith(base_domain):
                        continue
                    
                    # 3. Aynı path içinde değilse atla (alt sayfalar olmalı)
                    if not parsed.path.startswith(base_path):
                        continue
                    
                    # 4. Skip patterns kontrolü
                    if any(pattern in normalized_url for pattern in SKIP_URL_PATTERNS):
                        continue
                    
                    # 5. Allowed patterns kontrolü
                    if ALLOWED_URL_PATTERNS:
                        if not any(pattern in normalized_url for pattern in ALLOWED_URL_PATTERNS):
                            continue
                    
                    # Queue'ya ekle
                    to_visit.append((normalized_url, depth + 1))
                
                # Server'a nazik ol
                time.sleep(CRAWL_DELAY)
            
            except Exception as e:
                print(f"   ⚠️ Hata ({current_url[:50]}...): {e}")
                continue
        
        print(f"✅ {len(discovered_urls)} URL keşfedildi\n")
        return list(discovered_urls)
    
    def clean_html(self, text: str) -> str:
        """
        HTML içeriğini temizle (script, style, nav, footer vb. kaldır)
        
        Args:
            text: Ham HTML metni
            
        Returns:
            Temizlenmiş metin
        """
        soup = BeautifulSoup(text, "lxml")
        
        # Gereksiz etiketleri kaldır
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        cleaned_text = soup.get_text(separator=" ")
        cleaned_text = " ".join(cleaned_text.split())
        
        return cleaned_text

    def extract_structured_text(self, html: str) -> str:
        """
        HTML içeriğini başlık ve paragraf bütünlüğünü koruyacak şekilde metne çevir.

        Headings markdown başlıkları olarak korunur, paragraflar ve liste öğeleri
        ayrı bloklara bölünür. Bu sayede chunk'lar cümle ve konu sınırlarına daha
        yakın oluşur.
        """
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        body = soup.body or soup
        blocks = []

        interesting_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]

        for element in body.find_all(interesting_tags):
            text = " ".join(element.get_text(" ", strip=True).split())
            if not text:
                continue

            if element.name and element.name.startswith("h") and len(element.name) == 2:
                level = int(element.name[1])
                level = max(1, min(level, 6))
                blocks.append(f"{'#' * level} {text}")
            elif element.name == "li":
                blocks.append(f"- {text}")
            else:
                blocks.append(text)

        if not blocks:
            cleaned_text = soup.get_text(separator=" ")
            return " ".join(cleaned_text.split())

        structured_text = "\n\n".join(blocks)
        structured_text = re.sub(r"\n{3,}", "\n\n", structured_text).strip()
        return structured_text
    
    def extract_and_process_images(
        self, 
        url: str, 
        category: str,
        max_images: int = IMAGE_MAX_PER_PAGE
    ) -> List[Dict]:
        """
        URL'den görselleri çıkar, indir, filtrele ve embed et
        
        Args:
            url: Web sayfası URL'i
            category: Kategori adı
            max_images: Maksimum görsel sayısı
            
        Returns:
            İşlenmiş görsel metadata listesi
        """
        if not self.image_processor:
            return []
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "lxml")
            processed_images = []
            seen_image_urls = set()
            
            # img tag'lerini bul
            for img in soup.find_all("img", limit=max_images * 3):  # Fazladan dene (filtrelenecek)
                if len(processed_images) >= max_images:
                    break

                # Lazy-load varyantlarını da dene
                candidate_sources = [
                    img.get("src"),
                    img.get("data-src"),
                    img.get("data-original"),
                    img.get("data-lazy-src"),
                ]

                # srcset/data-srcset varsa ilk URL'i al
                srcset = img.get("srcset") or img.get("data-srcset")
                if srcset:
                    first_srcset_url = srcset.split(",")[0].strip().split(" ")[0].strip()
                    candidate_sources.append(first_srcset_url)

                for img_src in candidate_sources:
                    if len(processed_images) >= max_images:
                        break

                    if not img_src:
                        continue

                    # Data URI veya boş kaynakları atla
                    if img_src.startswith("data:"):
                        continue

                    # Göreli URL'leri tam URL'ye çevir
                    full_url = urljoin(url, img_src)
                    normalized_url = full_url.split("?")[0]

                    # Aynı görsel URL'ini tekrar işlemeyelim
                    if normalized_url in seen_image_urls:
                        continue
                    seen_image_urls.add(normalized_url)

                    # Uzantı yerine ImageProcessor içindeki gerçek indirme+PIL doğrulamasına güven
                    print(f"    🖼️ İşleniyor: {full_url[:60]}...")
                    image_metadata = self.image_processor.process_image_url(
                        url=full_url,
                        category=category,
                        source_url=url
                    )

                    if image_metadata:
                        image_metadata.update(self._extract_image_context(img))
                        processed_images.append(image_metadata)
                        print(f"      ✅ Eklendi ({len(processed_images)}/{max_images})")
                        # Bir img tag'i için bir başarılı aday yeterli
                        break
            
            if processed_images:
                print(f"  🖼️ {len(processed_images)} görsel işlendi")
            
            return processed_images
        
        except Exception as e:
            print(f"  ⚠️ Görsel işleme hatası: {e}")
            return []

    def _extract_image_context(self, img) -> Dict:
        """Extract text around an image so it can be linked to the best chunk."""
        def clean(value: str) -> str:
            return " ".join((value or "").split())

        alt_text = clean(img.get("alt", ""))
        title_text = clean(img.get("title", ""))

        caption_text = ""
        figure = img.find_parent("figure")
        if figure:
            caption = figure.find("figcaption")
            if caption:
                caption_text = clean(caption.get_text(" ", strip=True))

        heading_text = ""
        for parent in img.parents:
            previous_heading = parent.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            if previous_heading:
                heading_text = clean(previous_heading.get_text(" ", strip=True))
                break

        nearby_parts = []
        for sibling in list(img.find_all_previous(["p", "li"], limit=2)):
            text = clean(sibling.get_text(" ", strip=True))
            if text:
                nearby_parts.append(text)
        for sibling in list(img.find_all_next(["p", "li"], limit=2)):
            text = clean(sibling.get_text(" ", strip=True))
            if text:
                nearby_parts.append(text)

        nearby_text = clean(" ".join(nearby_parts))[:1200]
        context_text = clean(" ".join([alt_text, title_text, caption_text, heading_text, nearby_text]))

        return {
            "alt_text": alt_text,
            "title_text": title_text,
            "caption": caption_text,
            "section_title": heading_text,
            "nearby_text": nearby_text,
            "context_text": context_text,
        }

    def _image_public_payload(self, img_data: Dict) -> Dict:
        """Return image metadata without vectors or large binary-like fields."""
        keys = [
            "image_hash", "url", "local_path", "category", "source_url", "width", "height",
            "aspect_ratio", "file_size", "alt_text", "title_text", "caption",
            "section_title", "nearby_text", "context_text",
        ]
        return {key: img_data.get(key) for key in keys if img_data.get(key) not in (None, "")}

    def _score_image_chunk(self, img_data: Dict, chunk_text: str, section_title: str = "") -> float:
        """Score how strongly an image's local text belongs to a chunk."""
        context = " ".join([
            img_data.get("alt_text", ""),
            img_data.get("title_text", ""),
            img_data.get("caption", ""),
            img_data.get("section_title", ""),
            img_data.get("nearby_text", ""),
        ]).strip().lower()

        if not context:
            return 0.0

        chunk_norm = (chunk_text or "").lower()
        title_norm = (section_title or "").lower()
        score = 0.0

        for field, weight in [
            ("caption", 4.0),
            ("alt_text", 3.0),
            ("title_text", 2.0),
            ("section_title", 3.0),
        ]:
            value = (img_data.get(field) or "").strip().lower()
            if value and (value in chunk_norm or value in title_norm):
                score += weight

        tokens = set(re.findall(r"[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ-]{3,}", context))
        if tokens:
            chunk_tokens = set(re.findall(r"[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ-]{3,}", chunk_norm))
            overlap = len(tokens & chunk_tokens)
            score += min(overlap / max(len(tokens), 1), 1.0) * 5.0

        return score

    def _select_images_for_chunk(
        self,
        chunk_text: str,
        section_title: str,
        processed_images: List[Dict],
        max_images: int = 6
    ) -> List[Dict]:
        """Select the most relevant images for a text chunk."""
        scored_images = []
        for img_data in processed_images or []:
            score = self._score_image_chunk(img_data, chunk_text, section_title)
            scored_images.append((score, img_data))

        scored_images.sort(key=lambda item: item[0], reverse=True)
        selected = [img for score, img in scored_images if score > 0][:max_images]
        return [self._image_public_payload(img) for img in selected]
    
    def process_url(self, url: str, category: str) -> List[Dict]:
        """
        Tek bir URL'den veri topla ve işle
        
        Args:
            url: İşlenecek URL
            category: Veri kategorisi
            
        Returns:
            İşlenmiş doküman listesi
        """
        try:
            print(f"\n🔹 İşleniyor: {url}")
            
            # PDF veya web içeriği yükle
            if url.endswith(".pdf"):
                loader = PyMuPDFLoader(url)
            else:
                loader = WebBaseLoader(url)
            
            docs = loader.load()
            
            if not docs:
                print("⚠️ Uyarı: Boş doküman!")
                return []
            
            # Görselleri çıkar, indir ve işle (PDF değilse)
            processed_images = []
            if not url.endswith(".pdf"):
                processed_images = self.extract_and_process_images(url, category)
            
            # HTML temizleme ve metadata ekleme
            for doc in docs:
                if not url.endswith(".pdf"):
                    doc.page_content = self.extract_structured_text(doc.page_content)
                doc.metadata["source"] = url
                doc.metadata["category"] = category
                doc.metadata["processed_images"] = processed_images  # İşlenmiş görseller
            
            # Chunk'lara böl — başlığa göre bölümle, sonra bölüm içinde cümle bazlı chunking
            all_chunks = []
            for doc in docs:
                text = doc.page_content or ""
                processed_images_for_doc = doc.metadata.get("processed_images", [])
                chunks = self._split_text_into_chunks_by_section(
                    text=text,
                    source=url,
                    category=category,
                    processed_images=processed_images_for_doc
                )
                all_chunks.extend(chunks)

            print(f"🧩 {len(all_chunks)} chunk oluşturuldu")

            return all_chunks
        
        except Exception as e:
            print(f"❌ Hata ({url}): {e}")
            return []

    def _split_text_into_chunks_by_section(
        self,
        text: str,
        source: str,
        category: str,
        processed_images: List[Dict]
    ) -> List[object]:
        """
        Bölümlere ayırıp her bölüm içinde cümle bazlı chunking uygular.

        Dönülen 'chunk' objeleri, orijinal pipeline ile uyumlu olacak şekilde
        `page_content` ve `metadata` attribute'larına sahiptir.
        """
        if not text:
            return []

        # Bölümleri başlık satırlarına göre ayır (markdown başlıkları üretildi)
        # Başlıklar '# ' veya '## ' ile başlıyorsa yeni bir bölüm başlat
        lines = text.splitlines()
        sections = []
        cur_title = None
        cur_body_lines = []

        def flush_section():
            if cur_title is None and not cur_body_lines:
                return
            title = cur_title or ""
            body = "\n".join(cur_body_lines).strip()
            sections.append((title, body))

        for ln in lines:
            if ln.startswith("# ") or ln.startswith("## ") or ln.startswith("### "):
                # yeni başlık
                if cur_title is not None or cur_body_lines:
                    flush_section()
                cur_title = ln.strip()
                cur_body_lines = []
            else:
                cur_body_lines.append(ln)

        # son bölümü ekle
        if cur_title is not None or cur_body_lines:
            title = cur_title or ""
            body = "\n".join(cur_body_lines).strip()
            sections.append((title, body))

        # Eğer hiç başlık yoksa tüm metni tek bölüm olarak al
        if not sections:
            sections = [("", text)]

        chunks = []

        class SimpleChunk:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        # Parametreler
        max_chars = CHUNK_SIZE
        overlap = CHUNK_OVERLAP

        for title, body in sections:
            section_text = (title + "\n\n" + body).strip() if title else body
            if not section_text:
                continue

            # sentence spans
            sentence_iter = list(re.finditer(r".+?(?:[.!?]|$)(?:\s+|$)", section_text, flags=re.DOTALL))
            spans = [(m.start(), m.end()) for m in sentence_iter if m.group(0).strip()]

            if not spans:
                # fallback: take whole section
                spans = [(0, len(section_text))]

            i = 0
            n = len(spans)
            while i < n:
                # expand j as far as fits in max_chars
                j = i
                while j < n and (spans[j][1] - spans[i][0]) <= max_chars:
                    j += 1

                if j == i:
                    # one sentence too long — force include
                    j = i + 1

                chunk_start = spans[i][0]
                chunk_end = spans[j - 1][1]
                chunk_text = section_text[chunk_start:chunk_end].strip()

                # compute absolute start index relative to full document
                # Need to find section offset in original text
                # We'll search for section_text in text to find base offset
                try:
                    base_offset = text.index(section_text)
                except ValueError:
                    base_offset = 0

                abs_start = base_offset + chunk_start

                related_images = self._select_images_for_chunk(
                    chunk_text=chunk_text,
                    section_title=title,
                    processed_images=processed_images
                )

                metadata = {
                    "source": source,
                    "category": category,
                    "start_index": abs_start,
                    "section_title": title,
                    "processed_images": processed_images,
                    "related_images": related_images,
                }

                chunks.append(SimpleChunk(page_content=chunk_text, metadata=metadata))

                # compute next i considering overlap
                overlap_threshold = chunk_end - overlap
                # find smallest k >= i+1 such that spans[k][0] >= overlap_threshold
                k = i + 1
                while k < n and spans[k][0] < overlap_threshold:
                    k += 1

                i = k

        return chunks
    
    def collect_category(self, category_key: str, save_to_db: bool = True) -> int:
        """
        Belirli bir kategori için tüm verileri topla
        
        Args:
            category_key: Kategori anahtarı (örn: "warthog")
            save_to_db: MongoDB'ye kaydet
            
        Returns:
            Toplanan doküman sayısı
        """
        if category_key not in CATEGORIES:
            print(f"❌ Geçersiz kategori: {category_key}")
            return 0
        
        category_info = CATEGORIES[category_key]
        print(f"\n{'='*60}")
        print(f"📦 Kategori: {category_info['name']}")
        print(f"📝 Açıklama: {category_info['description']}")
        
        # URL'leri belirle: auto_discover veya manuel liste
        urls_to_process = []
        
        if category_info.get('auto_discover', False) and AUTO_DISCOVER_LINKS:
            # Otomatik URL keşfi
            base_url = category_info.get('base_url')
            if base_url:
                print(f"🔍 Otomatik URL keşfi aktif")
                urls_to_process = self.discover_urls(
                    base_url=base_url,
                    max_depth=MAX_CRAWL_DEPTH,
                    max_pages=MAX_PAGES_PER_CATEGORY
                )
            else:
                print(f"⚠️ base_url yok, manuel URL listesi kullanılıyor")
                urls_to_process = category_info.get('urls', [])
        else:
            # Manuel URL listesi
            urls_to_process = category_info.get('urls', [])
        
        print(f"🔗 İşlenecek URL sayısı: {len(urls_to_process)}")
        print(f"{'='*60}")
        
        all_chunks = []
        
        # Her URL'i işle
        for idx, url in enumerate(urls_to_process, 1):
            print(f"\n[{idx}/{len(urls_to_process)}] İşleniyor: {url}")
            chunks = self.process_url(url, category_key)
            all_chunks.extend(chunks)
        
        print(f"\n✅ Toplam {len(all_chunks)} chunk toplandı")
        
        # Hibrit veritabanına kaydet (MongoDB + Qdrant)
        if save_to_db and all_chunks:
            print("\n💾 Hibrit veritabanına kaydediliyor...")
            
            # Önce mevcut kategori verilerini sil (güncelleme için)
            deleted_mongo = self.db.delete_by_category(category_key)
            deleted_qdrant = self.vector_db.delete_by_category(category_key)
            deleted_images = self.vector_db.delete_by_category(
                category_key,
                collection_name=QDRANT_IMAGE_COLLECTION
            )
            
            if deleted_mongo > 0:
                print(f"🗑️ MongoDB'den {deleted_mongo} doküman silindi")
            if deleted_qdrant > 0:
                print(f"🗑️ Qdrant'tan {deleted_qdrant} vektör silindi")
            if deleted_images > 0:
                print(f"🗑️ Qdrant image collection'dan {deleted_images} görsel silindi")
            
            # Yeni verileri hazırla
            documents_to_insert = []
            vectors_to_insert = []
            embeddings_cache = []  # Qdrant için embeddings cache
            
            for chunk in all_chunks:
                # Text embedding oluştur (sadece Qdrant için kullanılacak)
                embedding = self.embeddings.embed_query(chunk.page_content)
                embeddings_cache.append(embedding)
                
                related_images = chunk.metadata.get("related_images", [])
                
                # MongoDB için doküman - processed_images'ı ÇIKART (tekrarlamayı önle)
                clean_metadata = {
                    k: v for k, v in chunk.metadata.items() 
                    if k not in {"processed_images", "related_images"}
                }
                
                doc = {
                    "content": chunk.page_content,
                    "category": category_key,
                    "source": chunk.metadata.get("source", ""),
                    "image_count": len(related_images),
                    "image_hashes": [img.get("image_hash") for img in related_images if img.get("image_hash")],
                    "images": related_images,
                    "metadata": clean_metadata  # Temizlenmiş metadata (processed_images YOK)
                }
                documents_to_insert.append(doc)
            
            # MongoDB'ye toplu ekleme (embeddings olmadan - %75 daha az yer)
            inserted_ids = self.db.insert_many_documents(documents_to_insert)
            print(f"✅ MongoDB: {len(inserted_ids)} doküman kaydedildi (metadata only)")
            
            # Qdrant text vectors için hazırla (MongoDB ID'leri + chunk content ile)
            for chunk, doc, mongo_id, embedding in zip(all_chunks, documents_to_insert, inserted_ids, embeddings_cache):
                vectors_to_insert.append({
                    "vector": embedding,
                    "mongodb_id": str(mongo_id),
                    "point_id": self.vector_db.make_point_id(
                        self.vector_db.collection_name,
                        str(mongo_id),
                        chunk.metadata.get("start_index", 0),
                        "text"
                    ),
                    "category": category_key,
                    "source": doc["source"],
                    "chunk_content": chunk.page_content,  # ⭐ Chunk'ın kendisini sakla
                    "start_index": chunk.metadata.get("start_index", 0),  # ⭐ Chunk'ın başladığı yeri sakla
                    "metadata": {
                        "has_images": doc["image_count"] > 0,
                        "image_hashes": doc.get("image_hashes", []),
                        "images": doc.get("images", []),
                        "section_title": chunk.metadata.get("section_title", ""),
                        "content_length": len(doc["content"]),
                        "chunk_length": len(chunk.page_content)
                    }
                })
            
            # Qdrant'a text vektörleri kaydet
            qdrant_ids = self.vector_db.insert_many_vectors(vectors_to_insert)
            print(f"✅ Qdrant Text: {len(qdrant_ids)} vektör kaydedildi")
            
            # Gorselleri Qdrant image collection'a chunk baglantilariyla kaydet
            image_occurrences = {}
            for chunk, doc, mongo_id in zip(all_chunks, documents_to_insert, inserted_ids):
                source = chunk.metadata.get("source", "")
                for img_data in chunk.metadata.get("processed_images", []):
                    image_hash = img_data.get("image_hash")
                    image_url = img_data.get("url", "")
                    source_url = img_data.get("source_url", source)
                    if not image_hash:
                        continue

                    key = f"{image_hash}|{source_url}|{image_url}"
                    score = self._score_image_chunk(
                        img_data=img_data,
                        chunk_text=chunk.page_content,
                        section_title=chunk.metadata.get("section_title", "")
                    )
                    existing = image_occurrences.get(key)
                    if not existing or score > existing["score"]:
                        public_image = self._image_public_payload(img_data)
                        public_image.update({
                            "related_mongodb_id": str(mongo_id),
                            "related_source": source,
                            "related_start_index": chunk.metadata.get("start_index", 0),
                            "related_section_title": chunk.metadata.get("section_title", ""),
                            "related_chunk_preview": chunk.page_content[:500],
                        })
                        image_occurrences[key] = {
                            "score": score,
                            "image": img_data,
                            "payload": public_image,
                        }

            if image_occurrences and self.image_vector_db:
                from qdrant_client.models import PointStruct
                
                unique_processed_images = list(image_occurrences.values())
                print(f"\n🖼️ {len(unique_processed_images)} görsel-chunk bağlantısı Qdrant'a kaydediliyor...")
                
                image_points = []
                for occurrence in unique_processed_images:
                    img_data = occurrence["image"]
                    payload = occurrence["payload"]
                    image_points.append(
                        PointStruct(
                            id=self.vector_db.make_point_id(
                                QDRANT_IMAGE_COLLECTION,
                                payload.get("related_mongodb_id"),
                                payload.get("image_hash"),
                                payload.get("source_url"),
                                payload.get("url"),
                            ),
                            vector=img_data["embedding"],
                            payload=payload
                        )
                    )
                
                self.image_vector_db.upsert(
                    collection_name=QDRANT_IMAGE_COLLECTION,
                    points=image_points
                )
                print(f"✅ Qdrant Images: {len(image_points)} görsel-chunk bağlantısı kaydedildi")
        
        return len(all_chunks)
    
    def collect_all_categories(self, save_to_db: bool = True) -> Dict[str, int]:
        """
        Tüm kategorileri topla
        
        Args:
            save_to_db: MongoDB'ye kaydet
            
        Returns:
            Kategori başına toplanan doküman sayısı
        """
        results = {}
        
        for category_key in CATEGORIES.keys():
            count = self.collect_category(category_key, save_to_db)
            results[category_key] = count
        
        print("\n" + "="*60)
        print("📊 TOPLAMA ÖZETİ")
        print("="*60)
        for cat, count in results.items():
            print(f"  {cat}: {count} doküman")
        print(f"\nToplam: {sum(results.values())} doküman")
        print("="*60)
        
        return results
    
    def close(self):
        """Bağlantıları kapat"""
        self.db.close()
        self.vector_db.close()


# Ana çalıştırma
if __name__ == "__main__":
    collector = DataCollector()
    
    # Tüm kategorileri topla
    collector.collect_all_categories(save_to_db=True)
    
    # Veritabanı istatistiklerini göster
    print("\n📊 Veritabanı İstatistikleri:")
    
    mongo_stats = collector.db.get_stats()
    print(f"\nMongoDB:")
    print(f"  Toplam doküman: {mongo_stats['total_documents']}")
    print(f"  Kategoriler: {mongo_stats['categories']}")
    
    qdrant_stats = collector.vector_db.get_stats()
    print(f"\nQdrant:")
    print(f"  Toplam vektör: {qdrant_stats.get('total_vectors', 0)}")
    print(f"  Vektör boyutu: {qdrant_stats.get('vector_size', 0)}")
    
    collector.close()
