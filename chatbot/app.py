"""
Streamlit Chatbot Arayüzü
RAG Destekli Türkçe Soru-Cevap Sistemi + Vision Desteği
"""
import sys
from pathlib import Path
import html
import streamlit as st
from datetime import datetime
from PIL import Image

# Ensure project root is importable regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CHATBOT_TITLE, CHATBOT_WELCOME, CATEGORIES
from rag.rag_pipeline import RAGSystem
from rag.vision_rag import VisionRAGSystem


# Sayfa yapılandırması
st.set_page_config(
    page_title=CHATBOT_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Stil
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .welcome-text {
        font-size: 1.1rem;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #000 !important;
    }
    .chat-message strong {
        color: #000 !important;
        font-size: 1.1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        color: #000 !important;
    }
    .assistant-message {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #000 !important;
    }
    .source-box {
        background-color: #fff8e1;
        padding: 0.8rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #000 !important;
        border: 1px solid #ffecb3;
        line-height: 1.55;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .source-box, .source-box * {
        color: #111827 !important;
    }
    .source-box strong {
        color: #000 !important;
        font-weight: 700;
    }
    .source-box a {
        color: #0b57d0 !important;
        text-decoration: underline;
    }
    div[data-testid="stExpander"] {
        border-color: #3a3f4b !important;
    }
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] label {
        color: inherit;
    }
    .stats-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #000 !important;
        border: 1px solid #c8e6c9;
    }
    /* Do not force all Markdown text to black; dark themes need inherited colors. */
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """RAG sistemini başlat (cache ile)"""
    return RAGSystem()


@st.cache_resource
def initialize_vision_rag(_rag_system):
    """Vision RAG sistemini başlat (cache ile)
    
    Args:
        _rag_system: Mevcut RAG sistemi (underscore ile başlar = cache'de ignore edilir)
    """
    return VisionRAGSystem(text_rag=_rag_system)


def _image_ref_from_item(item):
    """Extract a renderable image reference from a URL/path string or image metadata dict."""
    if isinstance(item, dict):
        return (
            item.get("local_path")
            or item.get("image_path")
            or item.get("url")
            or item.get("image_url")
            or item.get("src")
        )
    return item


def _image_caption_from_item(item, fallback: str = "") -> str:
    if not isinstance(item, dict):
        return fallback

    caption = item.get("caption") or item.get("title") or fallback
    category = item.get("category")
    if category and category not in caption:
        caption = f"{caption} - {category}" if caption else category
    return caption


def _render_image(image_ref: str, caption: str = ""):
    """Render a local image path or remote image URL."""
    image_ref = _image_ref_from_item(image_ref)
    if not image_ref:
        return

    try:
        image_path = Path(str(image_ref))
        if image_path.exists():
            st.image(str(image_path), width='stretch', caption=caption)
        else:
            st.image(str(image_ref), width='stretch', caption=caption)
    except Exception:
        if caption:
            st.caption(caption)


def _collect_source_images(sources: list, limit: int = 3) -> list:
    """Pick a few distinct source images to show directly with the answer."""
    collected = []
    seen = set()

    for source in sources or []:
        for image_item in source.get("images", []) or []:
            image_ref = _image_ref_from_item(image_item)
            key = str(image_ref)
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append({
                "image_ref": key,
                "caption": _image_caption_from_item(
                    image_item,
                    source.get("category", "Kaynak gorsel"),
                ),
            })
            if len(collected) >= limit:
                return collected

    return collected


def _render_inline_images(sources: list = None, similar_images: list = None):
    """Show the strongest visual evidence directly below the assistant answer."""
    visual_items = []

    for img_data in (similar_images or [])[:3]:
        visual_items.append({
            "image_ref": img_data.get("image_path") or img_data.get("image_url"),
            "caption": (
                f"{img_data.get('category', 'Benzer gorsel')} "
                f"({float(img_data.get('similarity', 0.0)):.2f})"
            ),
        })

    if not visual_items:
        visual_items = _collect_source_images(sources or [], limit=3)

    visual_items = [item for item in visual_items if item.get("image_ref")]
    if not visual_items:
        return

    st.markdown("**İlgili Görseller**")
    cols = st.columns(min(3, len(visual_items)))
    for idx, item in enumerate(visual_items):
        with cols[idx % len(cols)]:
            _render_image(item["image_ref"], item.get("caption", ""))


def format_chat_message(role: str, content: str, sources: list = None, similar_images: list = None):
    """Chat mesajını formatla"""
    message_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{icon} {html.escape(role.upper())}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Keep the message body outside the HTML wrapper so Markdown tables,
    # fenced code blocks and simple diagrams render correctly.
    st.markdown(content or "")

    if role == "assistant":
        _render_inline_images(sources=sources, similar_images=similar_images)
    
    # Benzer görselleri göster (eğer varsa)
    if similar_images and role == "assistant":
        with st.expander(f"🖼️ Benzer Görseller ({len(similar_images)})"):
            cols = st.columns(3)
            for idx, img_data in enumerate(similar_images):
                with cols[idx % 3]:
                    try:
                        # Local path'ten görseli göster
                        _render_image(img_data.get("image_path") or img_data.get("image_url"))
                        
                        st.caption(f"""
                        **Kategori:** {img_data.get('category', 'Bilinmiyor')}  
                        **Benzerlik:** {float(img_data.get('similarity', 0.0)):.2f}  
                        **Boyut:** {img_data.get('width', '?')}×{img_data.get('height', '?')}  
                        [Kaynak]({img_data.get('source_url', '#')})
                        """)
                    except Exception as e:
                        st.caption(f"Görsel yüklenemedi: {str(e)}")
    
    # Kaynakları göster
    if sources and role == "assistant":
        with st.expander("📚 Kaynaklar"):
            for i, source in enumerate(sources, 1):
                source_url = html.escape(str(source.get("source", "Bilinmiyor")))
                category = html.escape(str(source.get("category", "Bilinmiyor")))
                preview = html.escape(str(source.get("content_preview", "Onizleme yok")))
                st.markdown(f"""
                <div class="source-box">
                    <strong>{i}. Kaynak:</strong> {source_url}<br>
                    <strong>Kategori:</strong> {category}<br>
                    <strong>Benzerlik:</strong> {source.get('similarity', 0):.1%}<br>
                    <strong>Önizleme:</strong> {preview}
                </div>
                """, unsafe_allow_html=True)
                
                # Görselleri göster
                images = source.get('images', [])
                if images:
                    st.markdown("**🖼️ Görseller:**")
                    cols = st.columns(min(3, len(images)))
                    for idx, image_item in enumerate(images[:6]):  # Maksimum 6 görsel
                        with cols[idx % 3]:
                            try:
                                caption = _image_caption_from_item(image_item, f"Görsel {idx+1}")
                                _render_image(image_item, caption=caption)
                            except:
                                st.caption(f"Görsel {idx+1} yüklenemedi")


def main():
    """Ana uygulama"""
    
    # Başlık
    st.markdown(f'<div class="main-header">{CHATBOT_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-text">{CHATBOT_WELCOME}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Görsel Yükleme
        st.subheader("🖼️ Görsel Analizi")
        uploaded_file = st.file_uploader(
            "Görsel yükle",
            type=["jpg", "jpeg", "png", "webp"],
            help="Robot, aksesuar veya parça görseli yükleyin"
        )
        
        if uploaded_file is not None:
            # Görseli önizle
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görsel", width='stretch')
            
            # Session'a kaydet
            st.session_state.uploaded_image = uploaded_file.getvalue()
            st.success("✅ Görsel yüklendi! Şimdi soru sorabilirsiniz.")
        else:
            # Görsel silinirse session'dan kaldır
            if "uploaded_image" in st.session_state:
                del st.session_state.uploaded_image
        
        st.divider()
        
        # Kategori seçimi
        category_options = {
            "Tümü": None,
            **{CATEGORIES[k]["name"]: k for k in CATEGORIES.keys()}
        }
        
        selected_category_name = st.selectbox(
            "Kategori Filtresi",
            options=list(category_options.keys()),
            help="Belirli bir kategoride arama yapın"
        )
        
        selected_category = category_options[selected_category_name]
        
        st.divider()
        
        # İstatistikler
        st.header("📊 İstatistikler")
        
        try:
            rag = initialize_rag()
            stats = rag.db.get_stats()
            
            st.markdown(f"""
            <div class="stats-box">
                <strong>📄 Toplam Doküman:</strong> {stats['total_documents']}<br><br>
                <strong>📁 Kategoriler:</strong><br>
                {'<br>'.join([f"  • {cat}: {count} doküman" for cat, count in stats['categories'].items()])}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"İstatistikler yüklenemedi: {e}")
        
        st.divider()
        
        # Örnek sorular
        st.header("💡 Örnek Sorular")
        example_questions = [
            "Warthog robotunun bakım prosedürleri nelerdir?",
            "Husky bataryası nasıl şarj edilir?",
            "Dingo'un güvenlik özellikleri neler?",
            "Warthog nasıl taşınır?",
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{hash(q)}", width='stretch'):
                st.session_state.example_question = q
        
        st.divider()
        
        # Temizle butonu
        if st.button("🗑️ Sohbeti Temizle", width='stretch'):
            st.session_state.messages = []
            st.rerun()
    
    # Chat geçmişini başlat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # RAG sistemlerini başlat
    try:
        rag = initialize_rag()
        vision_rag = initialize_vision_rag(rag)
    except Exception as e:
        st.error(f"❌ RAG sistemi başlatılamadı: {e}")
        st.info("Lütfen MongoDB ve Ollama'nın çalıştığından emin olun.")
        st.stop()
    
    # Chat geçmişini göster
    for message in st.session_state.messages:
        format_chat_message(
            message["role"],
            message["content"],
            message.get("sources"),
            message.get("similar_images")
        )
    
    # Örnek sorudan gelen input
    if "example_question" in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
    else:
        user_input = None
    
    # Chat input
    if prompt := (user_input or st.chat_input("Sorunuzu buraya yazın...")):
        # Görsel var mı kontrol et
        has_image = "uploaded_image" in st.session_state
        
        # Kullanıcı mesajını ekle
        message_data = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        }
        
        if has_image:
            message_data["has_image"] = True
        
        st.session_state.messages.append(message_data)
        
        # Kullanıcı mesajını göster
        if has_image:
            st.markdown("### 🖼️ Görsel ile Soru")
        format_chat_message("user", prompt)
        
        # Cevap üret
        if has_image:
            # Vision RAG kullan
            with st.spinner("🤖 Görsel analiz ediliyor..."):
                try:
                    result = vision_rag.analyze_image_with_context(
                        image_bytes=st.session_state.uploaded_image,
                        question=prompt,
                        category=selected_category
                    )
                    
                    # Assistant mesajını ekle
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "similar_images": result.get("similar_images", []),
                        "timestamp": datetime.now(),
                        "has_image": True
                    })
                    
                    # Assistant mesajını göster
                    format_chat_message(
                        "assistant",
                        result["answer"],
                        result.get("sources", []),
                        result.get("similar_images", [])
                    )
                    
                    # Hata varsa uyar
                    if result.get("error"):
                        st.warning("⚠️ Görsel analizi sırasında bir sorun oluştu.")
                    else:
                        # Başarılı analiz sonrası görseli temizle
                        if st.button("🗑️ Görseli Temizle"):
                            del st.session_state.uploaded_image
                            st.rerun()
                
                except Exception as e:
                    error_msg = f"Görsel analizi hatası: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
        else:
            # Normal text RAG kullan
            with st.spinner("🤖 Cevap üretiliyor..."):
                try:
                    result = rag.generate_answer(prompt, category=selected_category)
                    
                    # Text-to-image search de yap (CLIP ile)
                    similar_images = []
                    try:
                        similar_images = vision_rag.search_similar_images(
                            query_text=prompt,
                            category=selected_category,
                            limit=3
                        )
                    except:
                        pass  # Görsel arama başarısız olsa bile text cevabı göster
                    
                    # Assistant mesajını ekle
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "similar_images": similar_images,
                        "timestamp": datetime.now()
                    })
                    
                    # Assistant mesajını göster
                    format_chat_message(
                        "assistant",
                        result["answer"],
                        result.get("sources", []),
                        similar_images
                    )
                    
                    # Hata varsa uyar
                    if result.get("error"):
                        st.warning("⚠️ Cevap üretilirken bir sorun oluştu.")
                
                except Exception as e:
                    error_msg = f"Bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })


if __name__ == "__main__":
    main()
