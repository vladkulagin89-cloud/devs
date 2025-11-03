# qdrant_probe.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
import qdrant_client
from qdrant_client.models import Filter

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLL = os.getenv("QDRANT_COLLECTION", "zendesk_6m")

client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL)

# 1) список коллекций
print("collections:", [c.name for c in client.get_collections().collections])

# 2) инфо по коллекции
try:
    info = client.get_collection(COLL)
    print("collection:", COLL, "status:", info.status)
except Exception as e:
    print("get_collection error:", e)

# 3) первые 5 по scroll
pts, next_page = client.scroll(collection_name=COLL, limit=5, with_payload=True, scroll_filter=Filter(must=[]))
print("sample count:", len(pts))
for p in pts:
    pl = p.payload or {}
    print(json.dumps({
        "ticket_id": pl.get("ticket_id"),
        "created_at": pl.get("created_at"),
        "created_ts": pl.get("created_ts"),
        "tags": pl.get("tags"),
        "app_versions": pl.get("app_versions"),
        "text_len": len((pl.get("text") or "")),
    }, ensure_ascii=False))

# 4) соберём топ уникальных тегов из первых 200
tags = {}
points, nextp = client.scroll(collection_name=COLL, limit=200, with_payload=True)
for p in points:
    for t in (p.payload or {}).get("tags") or []:
        tags[t] = tags.get(t, 0) + 1
print("tag histogram (first 200):", sorted(tags.items(), key=lambda x: -x[1])[:20])
