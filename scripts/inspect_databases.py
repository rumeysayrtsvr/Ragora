"""Inspect MongoDB and Qdrant state for the RAG database."""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient
from database.mongodb_manager import MongoDBManager
from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_IMAGE_COLLECTION,
)


def qdrant_collection_info(client, name):
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    return {
        "name": name,
        "points": int(info.points_count or 0),
        "vector_size": getattr(vectors, "size", None),
        "distance": str(getattr(vectors, "distance", "")),
    }


def main():
    mongo = MongoDBManager()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("=== MongoDB ===")
    total_docs = mongo.collection.count_documents({})
    print(f"docs: {total_docs}")
    print(f"docs_with_images: {mongo.collection.count_documents({'image_count': {'$gt': 0}})}")

    mongo_categories = Counter(
        doc.get("category", "unknown")
        for doc in mongo.collection.find({}, {"category": 1})
    )
    for category, count in sorted(mongo_categories.items()):
        print(f"  {category}: {count}")

    print("\n=== Qdrant ===")
    for collection in [QDRANT_COLLECTION, QDRANT_IMAGE_COLLECTION]:
        print(json.dumps(qdrant_collection_info(qdrant, collection), ensure_ascii=False))

    print("\n=== Sample Mongo Chunk ===")
    sample_doc = mongo.collection.find_one({"image_count": {"$gt": 0}})
    if sample_doc:
        sample_doc["_id"] = str(sample_doc["_id"])
        print(json.dumps({
            "_id": sample_doc.get("_id"),
            "category": sample_doc.get("category"),
            "source": sample_doc.get("source"),
            "image_count": sample_doc.get("image_count"),
            "image_hashes": sample_doc.get("image_hashes", [])[:5],
            "content_preview": sample_doc.get("content", "")[:500],
        }, ensure_ascii=False, indent=2))

    print("\n=== Sample Image Payload ===")
    points, _ = qdrant.scroll(
        collection_name=QDRANT_IMAGE_COLLECTION,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if points:
        print(json.dumps(points[0].payload, ensure_ascii=False, indent=2)[:3000])

    mongo.close()


if __name__ == "__main__":
    main()
