# rag_answer.py — RAG с форматированием ответа под стиль Vlad/Prisma
import os, math, json, re
from typing import Optional, List, Tuple
from datetime import datetime, timezone
from pathlib import Path

import requests
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, Range, MatchAny
from sentence_transformers import SentenceTransformer, util as sbert_util
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------- ENV ----------
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLL           = os.getenv("QDRANT_COLLECTION", "zendesk_6m")

LMSTUDIO_BASE  = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
MODEL          = os.getenv("MODEL", "openai/gpt-oss-20b")

EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-small")
RERANK_MODEL   = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DATE_FROM      = os.getenv("DATE_FROM")
DATE_FROM_TS   = int(datetime.fromisoformat(DATE_FROM).replace(tzinfo=timezone.utc).timestamp()) if DATE_FROM else None

TOP_K          = int(os.getenv("TOP_K", "12"))
DECAY_LAM      = float(os.getenv("DECAY_LAM", "0.02"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
SCORE_MIN      = float(os.getenv("SCORE_MIN", "0.20"))
SIM_MIN        = float(os.getenv("SIM_MIN", "0.40"))
MAX_CONTEXT_TOKENS       = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
MAX_OUTPUT_TOKENS        = int(os.getenv("MAX_OUTPUT_TOKENS", "256"))
SNIPPET_CHARS_PER_CHUNK  = int(os.getenv("SNIPPET_CHARS_PER_CHUNK", "700"))

ENABLE_RULE_TEMPLATES = os.getenv("ENABLE_RULE_TEMPLATES", "1") == "1"

# ---------- Clients / Models ----------
client = (qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
          if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL))

embedder    = SentenceTransformer(EMB_MODEL_NAME)
sim_embedder= SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross       = CrossEncoder(RERANK_MODEL)

# ---------- Utilities ----------
def iso_to_ts(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str: return None
    try: return int(datetime.fromisoformat(iso_str.replace("Z","+00:00")).timestamp())
    except Exception: return None

def recency_decay(created_at_iso: str, lam: float, kind: Optional[str]) -> float:
    if kind == "sop":  # SOP не старим
        return 1.0
    try:
        t = datetime.fromisoformat((created_at_iso or "").replace("Z", "+00:00"))
    except Exception:
        t = datetime.now(timezone.utc)
    days = (datetime.now(timezone.utc) - t).days
    return math.exp(-lam * max(0, days))

# простая евристика платформы
def extract_platform(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["iphone","ipad","ios","ipados","apple"]): return "ios"
    if any(k in t for k in ["android","samsung","pixel","xiaomi","huawei"]): return "android"
    return None

def extract_version_hint(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,2}\.\d{1,2})(?:\.\d{1,2})?\b", text)
    return m.group(1) if m else None

def embed_query(text: str):
    return embedder.encode([f"query: {text}"], normalize_embeddings=True)[0]

# ---------- Retrieval ----------
_TAG_CACHE = None
def _collect_known_tags():
    global _TAG_CACHE
    if _TAG_CACHE is not None:
        return _TAG_CACHE
    try:
        pts, _ = client.scroll(collection_name=COLL, limit=500, with_payload=True)
        s = set()
        for p in pts:
            for t in (p.payload or {}).get("tags") or []:
                s.add(str(t))
        _TAG_CACHE = s
    except Exception:
        _TAG_CACHE = set()
    return _TAG_CACHE

def build_qdrant_filter(platform: Optional[str]) -> Optional[Filter]:
    must = []
    if platform:
        known = _collect_known_tags()
        variants = [platform, platform.lower(), platform.upper(), platform.capitalize()]
        if any(v in known for v in variants):
            must.append(FieldCondition(key="tags", match=MatchAny(any=variants)))
    if DATE_FROM_TS is not None:
        must.append(FieldCondition(key="created_ts", range=Range(gte=DATE_FROM_TS)))
    return Filter(must=must) if must else None

def sort_and_score(hits, query_text: str) -> List[Tuple[float, str, dict]]:
    out = []
    for h in hits:
        p = h.payload or {}
        if DATE_FROM_TS is not None:
            ts = p.get("created_ts") or iso_to_ts(p.get("created_at"))
            if ts is not None and ts < DATE_FROM_TS:
                continue
        base  = float(h.score)
        decay = recency_decay(p.get("created_at", "2025-01-01T00:00:00Z"), DECAY_LAM, p.get("kind"))
        s = base * decay
        out.append((s, p.get("text", ""), p))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def rerank_with_crossencoder(query, cand_list):
    if not cand_list: return cand_list
    pairs = [(query, txt) for _, txt, _ in cand_list]
    ce = cross.predict(pairs)  # выше — лучше
    rescored = []
    for (base_s, txt, p), ce_s in zip(cand_list, ce):
        rescored.append((0.5*base_s + 0.5*float(ce_s), txt, p))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored[:5]

def build_context(chunks: List[Tuple[float,str,dict]]) -> str:
    blocks = []
    for i, (s, txt, p) in enumerate(chunks, 1):
        head = f"[{i}] kind:{p.get('kind')} title:{p.get('title')} ticket:{p.get('ticket_id')} created:{p.get('created_at')} score:{round(s,3)}"
        blocks.append(head + "\n" + txt[:SNIPPET_CHARS_PER_CHUNK] + "\n")
    ctx = "\n---\n".join(blocks)
    approx = max(1, len(ctx)//4)
    if approx > MAX_CONTEXT_TOKENS:
        ctx = ctx[:MAX_CONTEXT_TOKENS*4]
    return ctx

def confidence(chunks: List[Tuple[float,str,dict]]) -> float:
    if not chunks: return 0.0
    top3 = [s for s,_,_ in chunks[:3]]
    return sum(top3)/len(top3)

def retrieve_base(query_text: str, *, version_hint: Optional[str], platform: Optional[str], limit: int=None):
    if limit is None:
        limit = TOP_K
    qfilter = build_qdrant_filter(platform)
    vec = embed_query(query_text)

    hits = None
    if hasattr(client, "query_points"):
        try:
            resp = client.query_points(collection_name=COLL, query=vec, limit=limit, with_payload=True, query_filter=qfilter)
            hits = resp.points
        except Exception:
            hits = None
    if hits is None:
        hits = client.search(collection_name=COLL, query_vector=vec, limit=limit, with_payload=True, query_filter=qfilter)

    # необязательный префильтр по версии
    if version_hint:
        filtered = []
        for h in hits:
            versions = h.payload.get("app_versions") or []
            if any(str(v).startswith(version_hint) for v in versions):
                filtered.append(h)
        if filtered:
            hits = filtered

    cands = sort_and_score(hits, query_text)
    if len(cands) >= 3:
        cands = rerank_with_crossencoder(query_text, cands)
    return cands

# ---------- Answer formatting ----------
def make_assistant_prefill(platform: Optional[str]) -> str:
    # «праймим» стиль ответа — приветствие, структура, ветвление по магазину
    play_steps = (
        "1. Open Google Play.\n"
        "2. Tap profile icon → Payments & subscriptions.\n"
        "3. Subscriptions → select Prisma → Cancel and Request refund."
    )
    apple_steps = (
        "1. Go to reportaproblem.apple.com\n"
        "2. Sign in with Apple ID.\n"
        "3. Find the Prisma payment → Request a refund → follow the steps."
    )
    web_steps = (
        "Please send the email used at purchase and the purchase receipt. "
        "I'll check it and help with the refund from our side."
    )

    # если угадываем платформу — подсвечиваем соответствующую ветку
    target = ""
    if platform == "android":
        target = f"If purchased through Google Play:\n{play_steps}\n\n"
    elif platform == "ios":
        target = f"If purchased through Apple App Store:\n{apple_steps}\n\n"
    else:
        target = ""

    return (
        "You are Vlad from the Prisma support team. Always reply in English, "
        "polite and structured, with this format:\n\n"
        "Hello <name>,\n\n"
        "My name is Vlad, I’m from the Prisma support team. Thank you for reaching out.\n\n"
        "The refund process depends on where the subscription was purchased. "
        "Provide clear steps for Apple App Store and Google Play; if purchased on our website, "
        "ask for the email used at purchase and the receipt.\n\n"
        + target +
        "If purchased on our website:\n" + web_steps + "\n\n"
        "Finish with: 'Let me know once you try the steps or if you’re not sure which platform was used. I’ll help further.'\n\n"
        "Sign with: 'Best regards,\nVlad'"
    )

def ask_llm(user_text: str, context: str, assistant_prefill: Optional[str]=None) -> str:
    system = (
        "You are a level-headed L2/L3 support assistant. Use only the facts from the RAG context when giving product-specific details. "
        "If the context is insufficient for a definitive answer, provide the best general guidance and clearly state what extra info is needed. "
        "Keep responses concise, structured, and professional."
    )
    messages = [{"role": "system", "content": system}]
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    messages.append({"role": "user", "content": f"User message:\n{user_text}\n\nRAG context:\n{context}"})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT_TOKENS
    }
    try:
        r = requests.post(f"{LMSTUDIO_BASE}/chat/completions", json=payload, timeout=60)
        data = r.json()
        if r.status_code != 200:
            return f"LM Studio http {r.status_code}: {data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LM Studio недоступен: {e}"

# ---------- Public API ----------
def route_ticket(ticket_text: str, *, user_prompt: Optional[str]=None, ticket_id: Optional[int]=None) -> str:
    version_hint = extract_version_hint(ticket_text)
    platform     = extract_platform(ticket_text)

    # RAG candidates
    chunks = retrieve_base(ticket_text, version_hint=version_hint, platform=platform, limit=TOP_K)
    conf = confidence(chunks) if chunks else 0.0
    top_score = chunks[0][0] if chunks else 0.0

    # если совсем пусто — всё равно попробуем выдать полезный ответ по шаблону
    ctx = build_context(chunks) if chunks else "No exact matches. Provide general policy steps."

    # мягкий порог — позволяем отвечать даже при низком conf, но без «уверенных» утверждений
    local_thr = CONF_THRESHOLD - (0.1 if (chunks and chunks[0][2].get("kind") == "sop") else 0.0)
    assistant_prefill = make_assistant_prefill(platform)

    prompt = user_prompt or "Compose the full reply in Vlad/Prisma format described above."
    answer = ask_llm(prompt + "\n\nUser ticket:\n" + ticket_text, ctx, assistant_prefill=assistant_prefill)

    suffix = f"\n\n(confidence={conf:.2f}, top_score={top_score:.2f})"
    return answer + suffix

def route_answer(question: str, *, version_hint: Optional[str]=None, platform: Optional[str]=None) -> str:
    # общий вопрос (не конкретный тикет)
    chunks = retrieve_base(question, version_hint=version_hint, platform=platform, limit=TOP_K)
    ctx = build_context(chunks) if chunks else "No exact matches."
    assistant_prefill = None  # тут не навязываем шаблон ответа
    answer = ask_llm(question, ctx, assistant_prefill=assistant_prefill)
    conf = confidence(chunks) if chunks else 0.0
    top_score = chunks[0][0] if chunks else 0.0
    return answer + f"\n\n(confidence={conf:.2f}, top_score={top_score:.2f})"
