# rag_answer.py (без QueryPoints, совместимо со старыми qdrant-client)
import os, math, json, re
from typing import Optional, List, Tuple
from datetime import datetime, timezone

import requests
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer, util as sbert_util
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLL           = os.getenv("QDRANT_COLLECTION", "zendesk_6m")

LMSTUDIO_BASE  = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
MODEL          = os.getenv("MODEL", "openai/gpt-oss-20b")

DATE_FROM      = os.getenv("DATE_FROM")
DATE_FROM_TS   = int(datetime.fromisoformat(DATE_FROM).replace(tzinfo=timezone.utc).timestamp()) if DATE_FROM else None

TOP_K          = int(os.getenv("TOP_K", "6"))
DECAY_LAM      = float(os.getenv("DECAY_LAM", "0.03"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
SCORE_MIN      = float(os.getenv("SCORE_MIN", "0.30"))
SIM_MIN        = float(os.getenv("SIM_MIN", "0.45"))

client = (qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
          if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL))
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
sim_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(text: str):
    return embedder.encode([f"query: {text}"], normalize_embeddings=True)[0]

def recency_decay(created_at_iso: str, lam: float = DECAY_LAM) -> float:
    try:
        t = datetime.fromisoformat((created_at_iso or "").replace("Z", "+00:00"))
    except Exception:
        t = datetime.now(timezone.utc)
    days = (datetime.now(timezone.utc) - t).days
    return math.exp(-lam * max(0, days))

def iso_to_ts(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str: return None
    try: return int(datetime.fromisoformat(iso_str.replace("Z","+00:00")).timestamp())
    except Exception: return None

def extract_version_hint(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,2}\.\d{1,2})(?:\.\d{1,2})?\b", text)
    return m.group(1) if m else None

def extract_platform(text: str) -> Optional[str]:
    tt = text.lower()
    if "android" in tt: return "android"
    if "ios" in tt or "iphone" in tt or "ipad" in tt: return "ios"
    return None

def build_qdrant_filter(platform: Optional[str]) -> Optional[Filter]:
    must = []
    if platform:
        must.append(FieldCondition(key="tags", match=MatchValue(value=platform)))
    if DATE_FROM_TS is not None:
        must.append(FieldCondition(key="created_ts", range=Range(gte=DATE_FROM_TS)))
    return Filter(must=must) if must else None

def sort_and_score(hits) -> List[Tuple[float, str, dict]]:
    out = []
    for h in hits:
        p = h.payload
        if DATE_FROM_TS is not None:
            ts = p.get("created_ts")
            if ts is None:
                ts = iso_to_ts(p.get("created_at"))
            if ts is not None and ts < DATE_FROM_TS:
                continue
        s = h.score * recency_decay(p.get("created_at","2025-01-01T00:00:00Z"))
        out.append((s, p.get("text",""), p))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def ask_llm_strict(user_text: str, ticket_text: Optional[str], context: str) -> str:
    sys = (
        "Ты L2/L3 ассистент саппорта. Отвечай строго по фактам из данного тикета и контекста ниже. "
        "Не придумывай. Если данных недостаточно, напиши: ‘Needs verification (insufficient context)’. "
        "В конце добавь источники в формате [#]."
    )
    ticket_part = f"Текст тикета:\n{ticket_text}\n\n" if ticket_text else ""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"{ticket_part}Запрос:\n{user_text}\n\nRAG-контекст:\n{context}\n\nДобавь источники [#]."}
        ],
        "temperature": 0.1, "max_tokens": 800
    }
    r = requests.post(f"{LMSTUDIO_BASE}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def confidence(chunks: List[Tuple[float,str,dict]]) -> float:
    if not chunks: return 0.0
    top3 = [s for s,_,_ in chunks[:3]]
    return sum(top3)/len(top3)

def add_for_review(question: str, chunks: List[Tuple[float,str,dict]], reason: str, ticket_id: Optional[int]=None):
    rec = {
        "reason": reason,
        "question": question,
        "ticket_id": ticket_id,
        "chunks": [{"ticket_id": p.get("ticket_id"), "created_at": p.get("created_at")} for s,_,p in chunks],
        "ts": datetime.now(timezone.utc).isoformat()
    }
    with open("for_review.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")

# -------- retrieve_base: БЕЗ QueryPoints --------
def retrieve_base(query_text: str, *, version_hint: Optional[str], platform: Optional[str], limit: int=64):
    vec = embed_query(query_text)
    qfilter = build_qdrant_filter(platform)

    # пробуем современный вызов с kwargs
    hits = None
    if hasattr(client, "query_points"):
        try:
            resp = client.query_points(
                collection_name=COLL,
                query=vec,
                limit=limit,
                with_payload=True,
                query_filter=qfilter
            )
            hits = resp.points
        except Exception:
            hits = None

    # если нет query_points или он не сработал — падём на старый search
    if hits is None:
        hits = client.search(
            collection_name=COLL,
            query_vector=vec,
            limit=limit,
            with_payload=True,
            query_filter=qfilter
        )

    # жёсткий фильтр по версии
    if version_hint:
        filtered = []
        for h in hits:
            versions = h.payload.get("app_versions") or []
            if any(str(v).startswith(version_hint) for v in versions):
                filtered.append(h)
        if filtered:
            hits = filtered

    return sort_and_score(hits)

def build_context(chunks: List[Tuple[float,str,dict]]) -> str:
    blocks = []
    for i, (s, txt, p) in enumerate(chunks, 1):
        head = f"[{i}] ticket:{p.get('ticket_id')} created:{p.get('created_at')} score:{round(s,3)}"
        blocks.append(head + "\n" + txt + "\n")
    return "\n---\n".join(blocks)

def route_answer(question: str, *, version_hint: Optional[str]=None, platform: Optional[str]=None) -> str:
    chunks = retrieve_base(question, version_hint=version_hint, platform=platform, limit=64)
    if not chunks:
        add_for_review(question, [], "no_candidates")
        return "Нужна верификация админа (нет релевантных кандидатов). Добавлено в For review."

    conf = confidence(chunks)
    top_score = chunks[0][0]
    if top_score < SCORE_MIN or conf < CONF_THRESHOLD:
        add_for_review(question, chunks, f"low_confidence top={round(top_score,3)} conf={round(conf,3)}")
        return f"Нужна верификация админа (confidence={conf:.2f}). Запрос добавлен в For review."

    ctx = build_context(chunks)
    ans = ask_llm_strict(question, ticket_text=None, context=ctx)
    return ans + f"\n\n(confidence={conf:.2f}, top_score={top_score:.2f})"

def route_ticket(ticket_text: str, *, user_prompt: Optional[str]=None, ticket_id: Optional[int]=None) -> str:
    version_hint = extract_version_hint(ticket_text)
    platform = extract_platform(ticket_text)

    chunks = retrieve_base(ticket_text, version_hint=version_hint, platform=platform, limit=64)
    if not chunks:
        add_for_review(ticket_text, [], "no_candidates", ticket_id)
        return "Нужна верификация админа (нет релевантных кандидатов). Добавлено в For review."

    conf = confidence(chunks)
    top_score, top_text, _ = chunks[0]

    # проверка смысловой близости тикет↔топ-чанк
    t_vec = sim_embedder.encode(ticket_text, normalize_embeddings=True)
    c_vec = sim_embedder.encode(top_text[:2000], normalize_embeddings=True)
    sim = float(sbert_util.cos_sim(t_vec, c_vec)[0][0])

    if top_score < SCORE_MIN or conf < CONF_THRESHOLD or sim < SIM_MIN:
        reason = f"low_conf_or_match top={round(top_score,3)} conf={round(conf,3)} sim={round(sim,3)}"
        add_for_review(ticket_text, chunks, reason, ticket_id)
        return f"Нужна верификация админа ({reason}). Добавлено в For review."

    ctx = build_context(chunks)
    prompt = user_prompt or "Дай краткое и точное решение по этому тикету, опираясь только на контекст."
    ans = ask_llm_strict(prompt, ticket_text=ticket_text, context=ctx)
    return ans + f"\n\n(confidence={conf:.2f}, top_score={top_score:.2f}, match={sim:.2f})"

# --- debug ---
def debug_ticket_flow(ticket_text: str):
    version_hint = extract_version_hint(ticket_text)
    platform = extract_platform(ticket_text)
    chunks = retrieve_base(ticket_text, version_hint=version_hint, platform=platform, limit=10)
    print("== RETRIEVED for ticket ==")
    for i, (s, txt, p) in enumerate(chunks, 1):
        print(f"[{i}] t:{p.get('ticket_id')} date:{p.get('created_at')} score:{round(s,3)} vers:{p.get('app_versions')}")
        print((txt[:200] + " ...").replace("\n"," "))
        print("-"*60)
    return chunks

if __name__ == "__main__":
    sample_ticket = (
        "Subject: Приложение вылетает при открытии фото. "
        "Diagnostics: iPhone 15 Pro | 26.0.1 | 7.2.2 | en_US. "
        "Описание: после выбора фото экран черный, затем краш."
    )
    debug_ticket_flow(sample_ticket)
    print(route_ticket(sample_ticket, user_prompt="Дай пошаговое решение."))
    q = "Платёж прошёл дважды, как запросить возврат?"
    print(route_answer(q))