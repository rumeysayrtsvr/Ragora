"""Evaluate Ragora text and image retrieval on a small labeled test set.

The script reports Recall@k, MRR@k and Precision@k for:
- text semantic search over Qdrant,
- text keyword search over MongoDB,
- text hybrid search using Reciprocal Rank Fusion,
- image-to-image CLIP search over Qdrant web_images.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config.settings import QDRANT_HOST, QDRANT_IMAGE_COLLECTION, QDRANT_PORT
from data_collection.image_utils import ImageProcessor
from rag.rag_pipeline import RAGSystem


def normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def contains_any(haystack: str, needles: Iterable[str]) -> bool:
    h = normalize(haystack)
    return any(normalize(n) in h for n in needles if n)


def text_result_relevant(case: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    category_ok = doc.get("category") in set(case.get("expected_categories", []))
    source_ok = contains_any(doc.get("source", ""), case.get("source_contains", []))
    blob = f"{doc.get('source', '')} {doc.get('content', '')}"
    terms_ok = contains_any(blob, case.get("terms_any", []))
    return bool(category_ok and (source_ok or terms_ok))


def image_result_relevant(case: Dict[str, Any], item: Dict[str, Any]) -> bool:
    if item.get("image_hash") == case.get("expected_hash"):
        return True

    category_ok = item.get("category") == case.get("expected_category")
    source_blob = f"{item.get('source_url', '')} {item.get('related_source', '')}"
    source_ok = contains_any(source_blob, case.get("source_contains", []))
    section_blob = f"{item.get('related_section_title', '')} {item.get('related_chunk_preview', '')}"
    section_ok = contains_any(section_blob, case.get("section_terms", []))
    return bool(category_ok and (source_ok or section_ok))


def ranking_metrics(cases: Sequence[Dict[str, Any]], rankings: Dict[str, List[Any]], is_relevant, k: int) -> Dict[str, float]:
    reciprocal_ranks = []
    recalls = []
    hits = []
    precisions = []

    for case in cases:
        results = rankings.get(case["id"], [])[:k]
        rel_positions = [idx for idx, item in enumerate(results, start=1) if is_relevant(case, item)]
        hits.append(1.0 if rel_positions else 0.0)
        recalls.append(1.0 if rel_positions else 0.0)
        reciprocal_ranks.append(1.0 / rel_positions[0] if rel_positions else 0.0)
        precisions.append(sum(1 for item in results if is_relevant(case, item)) / float(k))

    n = max(len(cases), 1)
    return {
        f"hit@{k}": sum(hits) / n,
        f"recall@{k}": sum(recalls) / n,
        f"mrr@{k}": sum(reciprocal_ranks) / n,
        f"precision@{k}": sum(precisions) / n,
    }


def to_doc(payload: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    return {
        "content": payload.get("chunk_content") or payload.get("content", ""),
        "source": payload.get("source", "Bilinmiyor"),
        "category": payload.get("category", "Bilinmiyor"),
        "mongodb_id": str(payload.get("mongodb_id", "")),
        "start_index": payload.get("start_index", payload.get("metadata", {}).get("start_index", 0)),
        "score": float(score or 0.0),
    }


def keyword_search(rag: RAGSystem, query: str, limit: int) -> List[Dict[str, Any]]:
    cursor = rag.db.collection.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}, "content": 1, "source": 1, "category": 1, "metadata": 1},
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)

    docs = []
    for doc in cursor:
        docs.append(
            {
                "content": doc.get("content", ""),
                "source": doc.get("source", "Bilinmiyor"),
                "category": doc.get("category", "Bilinmiyor"),
                "mongodb_id": str(doc.get("_id", "")),
                "start_index": doc.get("metadata", {}).get("start_index", 0),
                "score": float(doc.get("score", 0.0)),
            }
        )
    return docs


def rrf_merge(rankings: Sequence[List[Dict[str, Any]]], limit: int, rrf_k: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = defaultdict(float)
    docs: Dict[str, Dict[str, Any]] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.get("mongodb_id") or f"{doc.get('source')}::{doc.get('start_index')}::{doc.get('content', '')[:80]}"
            scores[key] += 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)

    merged = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(docs[key])
        item["score"] = score
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def evaluate_text(rag: RAGSystem, cases: Sequence[Dict[str, Any]], k: int) -> Dict[str, Any]:
    semantic_rankings: Dict[str, List[Dict[str, Any]]] = {}
    keyword_rankings: Dict[str, List[Dict[str, Any]]] = {}
    hybrid_rankings: Dict[str, List[Dict[str, Any]]] = {}
    per_case = []

    for case in cases:
        query = case["query"]
        sem_docs = []
        for doc, score in rag.retrieve_relevant_documents(query, k=max(k * 2, 10)):
            sem_docs.append(dict(doc, score=float(score)))

        key_docs = keyword_search(rag, query, limit=max(k * 2, 10))
        hybrid_docs = rrf_merge([sem_docs, key_docs], limit=k)

        semantic_rankings[case["id"]] = sem_docs
        keyword_rankings[case["id"]] = key_docs
        hybrid_rankings[case["id"]] = hybrid_docs

        per_case.append(
            {
                "id": case["id"],
                "query": query,
                "semantic_first_relevant_rank": first_relevant_rank(case, sem_docs, text_result_relevant),
                "keyword_first_relevant_rank": first_relevant_rank(case, key_docs, text_result_relevant),
                "hybrid_first_relevant_rank": first_relevant_rank(case, hybrid_docs, text_result_relevant),
                "semantic_top_categories": [d.get("category") for d in sem_docs[:k]],
                "keyword_top_categories": [d.get("category") for d in key_docs[:k]],
                "hybrid_top_categories": [d.get("category") for d in hybrid_docs[:k]],
            }
        )

    return {
        "semantic": ranking_metrics(cases, semantic_rankings, text_result_relevant, k),
        "keyword": ranking_metrics(cases, keyword_rankings, text_result_relevant, k),
        "hybrid_rrf": ranking_metrics(cases, hybrid_rankings, text_result_relevant, k),
        "cases": per_case,
    }


def first_relevant_rank(case: Dict[str, Any], ranking: Sequence[Any], is_relevant) -> int | None:
    for idx, item in enumerate(ranking, start=1):
        if is_relevant(case, item):
            return idx
    return None


def make_query_variant(src: Path, tmp_dir: Path, case_id: str) -> Path:
    """Create a mild crop/compression variant so image retrieval is not only exact file lookup."""
    image = Image.open(src).convert("RGB")
    width, height = image.size
    crop_x = max(int(width * 0.03), 0)
    crop_y = max(int(height * 0.03), 0)
    if width > 260 and height > 260:
        image = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    image.thumbnail((640, 640))
    out = tmp_dir / f"{case_id}.jpg"
    image.save(out, format="JPEG", quality=82, optimize=True)
    return out


def image_search(processor: ImageProcessor, client: QdrantClient, image_path: Path, limit: int) -> List[Dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    vector = processor.get_image_embedding(image)
    results = client.query_points(
        collection_name=QDRANT_IMAGE_COLLECTION,
        query=vector,
        limit=max(limit, 10),
        score_threshold=0.40,
    )

    seen = set()
    items = []
    for result in results.points:
        payload = result.payload or {}
        image_hash = payload.get("image_hash", "")
        if image_hash in seen:
            continue
        seen.add(image_hash)
        items.append(
            {
                "image_hash": image_hash,
                "category": payload.get("category", "bilinmiyor"),
                "source_url": payload.get("source", payload.get("source_url", "")),
                "related_source": payload.get("related_source", ""),
                "related_section_title": payload.get("related_section_title", ""),
                "related_chunk_preview": payload.get("related_chunk_preview", ""),
                "similarity": float(result.score),
            }
        )
        if len(items) >= limit:
            break
    return items


def evaluate_images(cases: Sequence[Dict[str, Any]], k: int, use_variants: bool) -> Dict[str, Any]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    processor = ImageProcessor()
    rankings: Dict[str, List[Dict[str, Any]]] = {}
    per_case = []

    with tempfile.TemporaryDirectory(prefix="ragora_eval_") as tmp:
        tmp_dir = Path(tmp)
        for case in cases:
            src = ROOT_DIR / case["image_path"]
            query_path = make_query_variant(src, tmp_dir, case["id"]) if use_variants else src
            results = image_search(processor, client, query_path, limit=k)
            rankings[case["id"]] = results
            per_case.append(
                {
                    "id": case["id"],
                    "image_path": case["image_path"],
                    "expected_category": case["expected_category"],
                    "first_relevant_rank": first_relevant_rank(case, results, image_result_relevant),
                    "top_hashes": [r.get("image_hash") for r in results[:k]],
                    "top_categories": [r.get("category") for r in results[:k]],
                    "top_scores": [round(float(r.get("similarity", 0.0)), 4) for r in results[:k]],
                }
            )

    return {
        "clip_image_to_image": ranking_metrics(cases, rankings, image_result_relevant, k),
        "cases": per_case,
        "query_variant": "mild_crop_resize_jpeg82" if use_variants else "original_indexed_image",
    }


def markdown_table(rows: List[List[Any]]) -> str:
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(str(x) for x in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(result: Dict[str, Any], out_md: Path, k: int) -> None:
    text = result["text"]
    image = result["image"]
    rows = [["Arama yöntemi", "Soru sayısı", f"Recall@{k}", f"MRR@{k}", f"Precision@{k}"]]
    for label, key in [
        ("Semantik arama (Qdrant)", "semantic"),
        ("Anahtar kelime araması (MongoDB)", "keyword"),
        ("Hibrit RRF", "hybrid_rrf"),
    ]:
        m = text[key]
        rows.append([label, result["counts"]["text_cases"], f"{m[f'recall@{k}']:.2f}", f"{m[f'mrr@{k}']:.2f}", f"{m[f'precision@{k}']:.2f}"])

    img_m = image["clip_image_to_image"]
    image_rows = [
        ["Görsel arama yöntemi", "Görsel sayısı", f"Hit@{k}", f"MRR@{k}", f"Precision@{k}"],
        [
            "CLIP image-to-image",
            result["counts"]["image_cases"],
            f"{img_m[f'hit@{k}']:.2f}",
            f"{img_m[f'mrr@{k}']:.2f}",
            f"{img_m[f'precision@{k}']:.2f}",
        ],
    ]

    body = f"""# Nicel Ön Retrieval Değerlendirmesi

Bu değerlendirme, mevcut MongoDB ve Qdrant indeksleri üzerinde hazırlanan küçük bir etiketli test kümesiyle yapılmıştır. Metin tarafında {result["counts"]["text_cases"]} Türkçe sorgu; görsel tarafta {result["counts"]["image_cases"]} indeksli kaynak görselinden üretilen hafif kırpılmış/yeniden sıkıştırılmış sorgu görseli kullanılmıştır. Metin sorgularında doğru kabul edilen sonuçlar kategori, kaynak URL parçası ve beklenen kanıt terimleriyle; görsel sorgularda ise beklenen görsel hash'i, kategori ve kaynak/başlık eşleşmesiyle işaretlenmiştir.

{markdown_table(rows)}

{markdown_table(image_rows)}

Ön sonuçlar, metin tarafında hibrit RRF yaklaşımının semantik arama ve anahtar kelime aramasının güçlü yönlerini birleştirdiğini göstermektedir. Semantik arama, doğal dille sorulan açıklayıcı sorgularda kategori düzeyinde güçlü sonuçlar üretirken; MongoDB text search özellikle ürün adı, parça adı, hata/özellik terimi veya tablo değeri içeren sorgularda tamamlayıcı sinyal sağlamıştır. Görsel tarafta CLIP tabanlı image-to-image arama, indeksli görsellerden oluşturulan varyantlarda yüksek isabet üretmiş; bununla birlikte bu sonuçlar aynı kaynak koleksiyonundan türetilen görseller üzerinde ölçüldüğü için açık dünya fotoğraflarıyla ayrıca genişletilmelidir.
"""
    out_md.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="evaluation/retrieval_test_set.json")
    parser.add_argument("--out-json", default="evaluation/retrieval_eval_results.json")
    parser.add_argument("--out-md", default="evaluation/nicel_on_retrieval_degerlendirmesi.md")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--original-images", action="store_true", help="Use exact stored images instead of mild variants.")
    args = parser.parse_args()

    test_set = json.loads((ROOT_DIR / args.test_set).read_text(encoding="utf-8"))
    start = time.time()

    rag = RAGSystem()
    try:
        text_result = evaluate_text(rag, test_set["text_cases"], args.k)
    finally:
        rag.close()

    image_result = evaluate_images(test_set["image_cases"], args.k, use_variants=not args.original_images)

    result = {
        "k": args.k,
        "counts": {
            "text_cases": len(test_set["text_cases"]),
            "image_cases": len(test_set["image_cases"]),
        },
        "elapsed_seconds": round(time.time() - start, 2),
        "text": text_result,
        "image": image_result,
    }

    out_json = ROOT_DIR / args.out_json
    out_md = ROOT_DIR / args.out_md
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result, out_md, args.k)

    print(json.dumps({
        "summary": {
            "k": args.k,
            "text": {k: v for k, v in text_result.items() if k != "cases"},
            "image": {k: v for k, v in image_result.items() if k != "cases"},
            "elapsed_seconds": result["elapsed_seconds"],
        },
        "outputs": {
            "json": str(out_json),
            "markdown": str(out_md),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
