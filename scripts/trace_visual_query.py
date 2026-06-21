"""Trace a visual query from Qdrant image search to MongoDB chunks.

Usage inside Docker:
  python scripts/trace_visual_query.py /app/data/images/warthog/8fb5e49b0b815168.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.vision_rag import VisionRAGSystem  # noqa: E402


def clip(text: str, length: int = 260) -> str:
    text = " ".join((text or "").split())
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace visual retrieval through Qdrant and MongoDB.")
    parser.add_argument("image_path", help="Image path visible from the runtime/container.")
    parser.add_argument("--question", default="Bu görseldeki ürün nedir?", help="Optional user question.")
    parser.add_argument("--limit", type=int, default=5, help="Number of visual matches to inspect.")
    parser.add_argument("--threshold", type=float, default=0.40, help="Image similarity threshold.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    vision = VisionRAGSystem()

    try:
        similar_images = vision.search_similar_images_by_image(
            image_path=str(image_path),
            limit=args.limit,
            score_threshold=args.threshold,
        )

        print("=" * 90)
        print("QDRANT IMAGE MATCHES -> related_mongodb_id")
        print("=" * 90)

        if not similar_images:
            print("No image matches found.")
            return 1

        for idx, img in enumerate(similar_images, 1):
            print(f"\n[{idx}] score={float(img.get('similarity', 0.0)):.4f}")
            print(f"    category: {img.get('category')}")
            print(f"    image_hash: {img.get('image_hash')}")
            print(f"    image_url: {img.get('image_url')}")
            print(f"    related_mongodb_id: {img.get('related_mongodb_id')}")
            print(f"    related_source: {img.get('source_url')}")

        linked_docs = vision._retrieve_chunks_linked_to_images(similar_images, limit=args.limit)

        print("\n" + "=" * 90)
        print("MONGODB CHUNKS FETCHED BY related_mongodb_id")
        print("=" * 90)

        if not linked_docs:
            print("No MongoDB chunks could be fetched from related_mongodb_id.")
            return 1

        for idx, (doc, score) in enumerate(linked_docs, 1):
            print(f"\n[{idx}] score={float(score):.4f}")
            print(f"    mongodb_id: {doc.get('mongodb_id')}")
            print(f"    category: {doc.get('category')}")
            print(f"    source: {doc.get('source')}")
            print(f"    start_index: {doc.get('start_index')}")
            print(f"    image_count: {len(doc.get('images') or [])}")
            print(f"    content: {clip(doc.get('content', ''))}")

        result = vision.analyze_image_with_context(
            image_path=str(image_path),
            question=args.question,
        )

        print("\n" + "=" * 90)
        print("FINAL VISUAL RAG ANSWER")
        print("=" * 90)
        print(clip(result.get("answer", ""), 1200))
        return 0 if not result.get("error") else 1
    finally:
        vision.close()


if __name__ == "__main__":
    raise SystemExit(main())
