"""
Görsel İndirme, Filtreleme ve Embedding Modülü  
CLIP modeli ile görsel-text eşleştirme (Türkçe sorgu desteği)
"""
import sys
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import hashlib
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    CLIP_MODEL,
    IMAGE_MIN_WIDTH,
    IMAGE_MIN_HEIGHT,
    IMAGE_MAX_FILE_SIZE,
    IMAGE_MIN_ASPECT_RATIO,
    IMAGE_MAX_ASPECT_RATIO,
    IMAGE_DOWNLOAD_TIMEOUT,
    IMAGE_STORAGE_DIR
)


class ImageProcessor:
    """Görsel işleme ve embedding üretimi (CLIP + Türkçe çeviri)"""
    
    def __init__(self):
        """Image processor'ı başlat"""
        print("🔄 CLIP modeli yükleniyor (Türkçe çeviri destekli)...")
        
        # CLIP model ve processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        
        # Translator (Türkçe → İngilizce)
        self.translator = GoogleTranslator(source='tr', target='en')
        
        # Storage dizini oluştur
        IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ CLIP modeli hazır ({self.device}, Türkçe→EN çeviri aktif)")
    
    def is_valid_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Görselin geçerli olup olmadığını kontrol et
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            (geçerli_mi, sebep) tuple
        """
        width, height = image.size
        
        # Boyut kontrolü
        if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
            return False, f"Çok küçük: {width}x{height}"
        
        # Aspect ratio kontrolü
        aspect_ratio = width / height
        if aspect_ratio < IMAGE_MIN_ASPECT_RATIO:
            return False, f"Çok dar: {aspect_ratio:.2f}"
        if aspect_ratio > IMAGE_MAX_ASPECT_RATIO:
            return False, f"Çok geniş: {aspect_ratio:.2f}"
        
        # Format kontrolü
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, f"Desteklenmeyen format: {image.mode}"
        
        return True, "Geçerli"
    
    def download_image(
        self, 
        url: str,
        verify_ssl: bool = True
    ) -> Optional[Tuple[Image.Image, bytes]]:
        """
        URL'den görseli indir ve doğrula
        
        Args:
            url: Görsel URL'i
            verify_ssl: SSL doğrulaması
            
        Returns:
            (PIL Image, bytes) tuple veya None
        """
        try:
            # Gereksiz görselleri filtrele (URL bazlı)
            lower_url = url.lower()
            skip_patterns = [
                'logo', 'icon', 'favicon', 'badge',
                'avatar', 'thumbnail', 'sprite', 'pixel',
                '.svg', '.gif'  # SVG ve animated GIF'leri atla
            ]
            
            if any(pattern in lower_url for pattern in skip_patterns):
                return None

            # 'banner' kelimesi robot gorsellerinde de kullanilabildigi icin genel filtreleme yapma.
            # Sadece site chrome gorunümlerini elemek icin website_images banner'larini atla.
            if 'banner' in lower_url and 'website_images' in lower_url:
                return None
            
            # İndir
            response = requests.get(
                url,
                timeout=IMAGE_DOWNLOAD_TIMEOUT,
                verify=verify_ssl,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            # Boyut kontrolü
            if len(response.content) > IMAGE_MAX_FILE_SIZE:
                return None
            
            # PIL Image'a çevir
            image = Image.open(BytesIO(response.content))
            
            # RGB'ye çevir (RGBA varsa)
            if image.mode == 'RGBA':
                # Beyaz background ekle
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Alpha channel
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validasyon
            is_valid, reason = self.is_valid_image(image)
            if not is_valid:
                return None
            
            return image, response.content
        
        except Exception as e:
            return None
    
    def generate_image_hash(self, image_bytes: bytes) -> str:
        """
        Görsel için benzersiz hash oluştur
        
        Args:
            image_bytes: Görsel byte verisi
            
        Returns:
            SHA256 hash
        """
        return hashlib.sha256(image_bytes).hexdigest()[:16]
    
    def save_image(
        self, 
        image_bytes: bytes,
        category: str,
        image_hash: str
    ) -> Path:
        """
        Görseli diske kaydet
        
        Args:
            image_bytes: Görsel verisi
            category: Kategori adı
            image_hash: Görsel hash'i
            
        Returns:
            Kaydedilen dosya yolu
        """
        # Kategori dizini oluştur
        category_dir = IMAGE_STORAGE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya yolu
        file_path = category_dir / f"{image_hash}.jpg"
        
        # Kaydet
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return file_path
    
    def get_image_embedding(self, image: Image.Image) -> List[float]:
        """
        CLIP ile görsel embedding oluştur
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding vektörü (512-dim)
        """
        try:
            # RGB formatına çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Embedding al
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Güvenli normalizasyon
                if isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    # Eğer output object ise, tensoru çıkar
                    image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
                
                # Normalize et
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Numpy array'e çevir
            embedding = image_features.cpu().numpy()[0].tolist()
            
            return embedding
        
        except Exception as e:
            print(f"  ⚠️ Image embedding hatası: {e}")
            # Fallback: Sıfır vektörü döndür
            return [0.0] * 512
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        CLIP ile text embedding oluştur (Türkçe otomatik çevrilir)
        
        Args:
            text: Metin sorgusu (Türkçe veya İngilizce)
            
        Returns:
            Embedding vektörü (512-dim)
        """
        try:
            # Türkçe → İngilizce çeviri (CLIP İngilizce eğitilmiş)
            try:
                # Basit dil tespiti: Türkçe karakterler varsa çevir
                turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
                if any(char in text for char in turkish_chars):
                    text_en = self.translator.translate(text)
                    print(f"  🔄 Çeviri: '{text}' → '{text_en}'")
                else:
                    text_en = text
            except Exception as e:
                print(f"  ⚠️ Çeviri hatası, orijinal metin kullanılacak: {e}")
                text_en = text
            
            # Preprocess
            inputs = self.processor(text=[text_en], return_tensors="pt", padding=True).to(self.device)
            
            # Embedding al
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                # Güvenli normalizasyon
                if isinstance(outputs, torch.Tensor):
                    text_features = outputs
                else:
                    # Eğer output object ise, tensoru çıkar
                    text_features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
                
                # Normalize et
                text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Numpy array'e çevir
            embedding = text_features.cpu().numpy()[0].tolist()
            
            return embedding
        
        except Exception as e:
            print(f"  ⚠️ Text embedding hatası: {e}")
            # Fallback: Sıfır vektörü döndür (arama yapılmayacak ama hata kalmayacak)
            return [0.0] * 512
    
    def process_image_url(
        self,
        url: str,
        category: str,
        source_url: str
    ) -> Optional[Dict]:
        """
        URL'den görseli indir, filtrele, embed et ve kaydet
        
        Args:
            url: Görsel URL'i
            category: Kategori
            source_url: Kaynak sayfa URL'i
            
        Returns:
            Görsel metadata dict veya None
        """
        # İndir
        result = self.download_image(url)
        if not result:
            return None
        
        image, image_bytes = result
        
        # Hash oluştur
        image_hash = self.generate_image_hash(image_bytes)
        
        # Diske kaydet
        file_path = self.save_image(image_bytes, category, image_hash)
        
        # Embedding oluştur
        embedding = self.get_image_embedding(image)
        
        # Metadata
        metadata = {
            "image_hash": image_hash,
            "url": url,
            "local_path": str(file_path),
            "category": category,
            "source_url": source_url,
            "width": image.size[0],
            "height": image.size[1],
            "aspect_ratio": image.size[0] / image.size[1],
            "file_size": len(image_bytes),
            "embedding": embedding
        }
        
        return metadata


# Test için
if __name__ == "__main__":
    print("🧪 Image Processor Test...")
    
    processor = ImageProcessor()
    
    # Test URL (örnek)
    test_url = "https://docs.clearpathrobotics.com/assets/images/warthog.jpg"
    
    result = processor.process_image_url(
        url=test_url,
        category="test",
        source_url="https://example.com"
    )
    
    if result:
        print(f"\n✅ Görsel işlendi:")
        print(f"  Hash: {result['image_hash']}")
        print(f"  Boyut: {result['width']}x{result['height']}")
        print(f"  Embedding boyutu: {len(result['embedding'])}")
    else:
        print("\n❌ Görsel işlenemedi (filtrelendi)")
