# index_to_qdrant.py — индексация tickets.jsonl / tickets_chunks.jsonl в Qdrant
import os, json, uuid, re
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "zendesk_6m")
INPUT_PATH        = os.getenv("INPUT_PATH", "tickets.jsonl")
EMB_MODEL         = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small")
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", "256"))

# ---------- helpers ----------

def detect_platform(text: str) -> str | None:
    t = (text or "").lower()
    if any(k in t for k in [" ios ", "iphone", "ipad", "iOS", " ios,", " iOS "]):
        return "ios"
    if any(k in t for k in ["android", "samsung", "pixel", "huawei", "xiaomi"]):
        return "android"
    return None

def extract_versions(text: str):
    # ловим 26.0.1, 7.2.2, 7.3 и т.п.
    return list({m.group(0) for m in re.finditer(r"\b\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\b", text or "")})

def iso_to_ts(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def pull_comments(rec: dict) -> list[str]:
    out: list[str] = []

    # формат 1: "comments": [ "...", { "body": "...", ... }, ... ]
    cmts = rec.get("comments") or []
    if isinstance(cmts, list):
        for c in cmts:
            if isinstance(c, str):
                body = c.strip()
            elif isinstance(c, dict):
                body = (c.get("body")
                        or c.get("text")
                        or c.get("comment")
                        or "")
                body = str(body).strip()
            else:
                body = ""
            if body:
                out.append(body)

    # формат 2: "audits": [ { "events": [ {"type":"Comment","body":"..."} ] }, ... ]
    if not out:
        audits = rec.get("audits") or []
        for a in audits:
            events = a.get("events") or []
            for ev in events:
                t = (ev.get("type") or ev.get("event_type") or "").lower()
                if t in ("comment", "note"):
                    body = (ev.get("body") or ev.get("html_body") or ev.get("text") or "")
                    body = str(body).strip()
                    if body:
                        out.append(body)

    return out

def build_text_from_ticket(rec: dict) -> str:
    # если уже готовый чанк
    if rec.get("text"):
        return rec["text"]

    parts = []
    subj = rec.get("subject")
    desc = rec.get("description") or rec.get("body")
    di   = rec.get("diagnostics") or rec.get("diagnostics_info") or rec.get("diagnostics_information")

    if subj: parts.append(f"Subject: {subj}")
    if desc: parts.append(f"Description: {desc}")
    if di:   parts.append(f"Diagnostics: {di}")

    cmts = pull_comments(rec)
    if cmts:
        parts.append("Comments:\n" + "\n---\n".join(cmts))

    return "\n".join(parts)

# ---------- main ----------

def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {in_path.resolve()}")

    client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL)
    embedder = SentenceTransformer(EMB_MODEL)

    # создаём коллекцию при отсутствии
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception:
        dim = embedder.get_sentence_embedding_dimension()
        client.recreate_collection(QDRANT_COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def embed_passage(text: str):
        return embedder.encode([f"passage: {text}"], normalize_embeddings=True)[0]

    batch, total = [], 0
    for rec in tqdm(read_jsonl(in_path), desc="Indexing"):
        ticket_id  = rec.get("ticket_id") or rec.get("id")
        created_at = rec.get("created_at") or rec.get("created") or "2025-01-01T00:00:00Z"
        tags       = list(rec.get("tags") or [])

        text = build_text_from_ticket(rec)

        app_versions = rec.get("app_versions")
        if not app_versions:
            app_versions = extract_versions(text)

        platform = detect_platform(text)
        if platform and platform not in tags:
            tags.append(platform)

        payload = {
            "kind": "ticket",
            "ticket_id": ticket_id,
            "created_at": created_at,
            "created_ts": iso_to_ts(created_at),
            "tags": tags,
            "app_versions": app_versions,
            "text": text,
        }

        vec = embed_passage(text)
        batch.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

        if len(batch) >= BATCH_SIZE:
            client.upsert(QDRANT_COLLECTION, points=batch)
            total += len(batch)
            batch.clear()

    if batch:
        client.upsert(QDRANT_COLLECTION, points=batch)
        total += len(batch)

    print(f"Готово. Проиндексировано: {total} записей → коллекция {QDRANT_COLLECTION}")

if __name__ == "__main__":
    main()
