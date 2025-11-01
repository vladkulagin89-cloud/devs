# index_to_qdrant.py (fixed)
import os, json, uuid
from tqdm import tqdm
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

COLL = "zendesk_6m"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 1) клиент
client = (qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
          if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL))

# 2) эмбеддер
model = SentenceTransformer("intfloat/multilingual-e5-small")
VECTOR_SIZE = model.get_sentence_embedding_dimension()

# 3) коллекция (без deprecated recreate_collection)
if not client.collection_exists(COLL):
    client.create_collection(
        collection_name=COLL,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

def e5_embed(texts):
    # e5 требует префикс "passage: "
    return model.encode([f"passage: {t}" for t in texts],
                        show_progress_bar=False, normalize_embeddings=True)

def id_from_chunk(chunk_id: str) -> str:
    # детерминированный UUID5 из текстового chunk_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"zendesk://{chunk_id}"))

BATCH = 64
buf_t, buf_m, buf_id = [], [], []

total = 0
with open("corpus.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Batches", unit="batch"):
        rec = json.loads(line)
        buf_t.append(rec["text"])
        # переносим метаданные в payload
        buf_m.append(rec["meta"] | {"ticket_id": rec["ticket_id"]})
        buf_id.append(rec["chunk_id"])

        if len(buf_t) == BATCH:
            vecs = e5_embed(buf_t)
            points = [
                PointStruct(
                    id=id_from_chunk(buf_id[j]),                # UUID вместо строки "123#0"
                    vector=vecs[j].tolist(),
                    payload={"text": buf_t[j], **buf_m[j]},
                )
                for j in range(len(buf_t))
            ]
            client.upsert(COLL, points=points)
            total += len(points)
            buf_t, buf_m, buf_id = [], [], []

# хвост
if buf_t:
    vecs = e5_embed(buf_t)
    points = [
        PointStruct(
            id=id_from_chunk(buf_id[j]),
            vector=vecs[j].tolist(),
            payload={"text": buf_t[j], **buf_m[j]},
        )
        for j in range(len(buf_t))
    ]
    client.upsert(COLL, points=points)
    total += len(points)

print(f"indexed points: {total}")
