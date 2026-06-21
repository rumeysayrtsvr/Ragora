"""Verify MongoDB <-> Qdrant referential integrity.

Checks:
- Qdrant text collection payload.mongodb_id exists in MongoDB.
- Qdrant image collection payload.related_mongodb_id exists in MongoDB.
- Mongo documents with images are summarized as a coverage warning.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    MONGODB_COLLECTION,
    MONGODB_DB_NAME,
    MONGODB_URI,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_IMAGE_COLLECTION,
    QDRANT_PORT,
)


def scroll_payload_ids(client: QdrantClient, collection_name: str, payload_key: str) -> list[str]:
    ids: list[str] = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            value = payload.get(payload_key)
            if value:
                ids.append(str(value))

        if next_offset is None:
            break

    return ids


def classify_ids(collection, ids: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    found: list[str] = []
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []

    for mongo_id in sorted(set(ids)):
        try:
            obj_id = ObjectId(mongo_id)
        except Exception as exc:
            invalid.append((mongo_id, str(exc)))
            continue

        if collection.count_documents({"_id": obj_id}, limit=1):
            found.append(mongo_id)
        else:
            missing.append(mongo_id)

    return found, missing, invalid


def pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 100.0


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    mongo = MongoClient(MONGODB_URI)
    collection = mongo[MONGODB_DB_NAME][MONGODB_COLLECTION]

    print_section("QDRANT VE MONGODB SENKRONIZASYON RAPORU")
    print(f"Rapor tarihi: {datetime.now().isoformat(timespec='seconds')}")

    checks = [
        (QDRANT_COLLECTION, "mongodb_id", "Text vectors"),
        (QDRANT_IMAGE_COLLECTION, "related_mongodb_id", "Image vectors"),
    ]

    report_lines = [
        "QDRANT ve MONGODB SENKRONIZASYON RAPORU",
        "=" * 80,
        f"Rapor tarihi: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    has_failure = False
    image_reference_ids: list[str] = []

    for collection_name, payload_key, label in checks:
        print_section(f"{label}: {collection_name}.{payload_key}")
        qdrant_ids = scroll_payload_ids(qdrant, collection_name, payload_key)
        counts = Counter(qdrant_ids)
        found, missing, invalid = classify_ids(collection, qdrant_ids)
        unique_total = len(set(qdrant_ids))

        if collection_name == QDRANT_IMAGE_COLLECTION:
            image_reference_ids = qdrant_ids

        print(f"Qdrant payload ID toplam: {len(qdrant_ids)}")
        print(f"Qdrant payload ID unique: {unique_total}")
        print(f"Duplicate referans: {len(qdrant_ids) - unique_total}")
        print(f"MongoDB'de bulunan: {len(found)} ({pct(len(found), unique_total):.1f}%)")
        print(f"MongoDB'de bulunamayan: {len(missing)}")
        print(f"Geçersiz ObjectId: {len(invalid)}")

        report_lines.extend([
            f"{label}: {collection_name}.{payload_key}",
            f"  Qdrant payload ID toplam: {len(qdrant_ids)}",
            f"  Qdrant payload ID unique: {unique_total}",
            f"  Duplicate referans: {len(qdrant_ids) - unique_total}",
            f"  MongoDB'de bulunan: {len(found)} ({pct(len(found), unique_total):.1f}%)",
            f"  MongoDB'de bulunamayan: {len(missing)}",
            f"  Geçersiz ObjectId: {len(invalid)}",
            "",
        ])

        if missing:
            has_failure = True
            print("\nMongoDB'de bulunamayan ilk 20 ID:")
            for mongo_id in missing[:20]:
                print(f"  - {mongo_id}")
            report_lines.append("  Bulunamayan ID'ler:")
            report_lines.extend(f"    {mongo_id}" for mongo_id in missing)

        if invalid:
            has_failure = True
            print("\nGeçersiz formatlı ilk 10 ID:")
            for mongo_id, error in invalid[:10]:
                print(f"  - {mongo_id}: {error}")

        duplicate_hotspots = [item for item in counts.most_common(10) if item[1] > 1]
        if duplicate_hotspots:
            print("\nEn sık tekrar eden referanslar:")
            for mongo_id, count in duplicate_hotspots:
                print(f"  - {mongo_id}: {count}")

    print_section("Mongo Görsel Karşılığı")
    image_ref_set = set(image_reference_ids)
    mongo_image_docs = list(collection.find(
        {"image_count": {"$gt": 0}},
        {"_id": 1, "image_count": 1, "image_hashes": 1},
    ))
    missing_image_payload_docs = [
        str(doc["_id"]) for doc in mongo_image_docs if str(doc["_id"]) not in image_ref_set
    ]
    print(f"Mongo image_count > 0 doküman: {len(mongo_image_docs)}")
    print(f"Qdrant image payload referansı olmayan Mongo dokümanı: {len(missing_image_payload_docs)}")

    if missing_image_payload_docs:
        print("Not: Bu bir referans bütünlüğü hatası değildir; web_images koleksiyonu her")
        print("Mongo chunk'ını değil, embed edilen benzersiz/best-match görselleri tutar.")
        print("İlk 20 eksik görsel referansı:")
        for mongo_id in missing_image_payload_docs[:20]:
            print(f"  - {mongo_id}")

    report_path = ROOT_DIR / "sync_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print_section("SONUÇ")
    if has_failure:
        print("Senkronizasyon hatası bulundu. Detaylar sync_report.txt dosyasına yazıldı.")
        return 1

    print("Tüm Qdrant referansları MongoDB ile senkronize.")
    print(f"Detaylı rapor: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
