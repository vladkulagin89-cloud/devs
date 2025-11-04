# log_utils.py — единый формат логов и хелперы

import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

BASE_DIR   = Path(__file__).parent
LOGS_DIR   = BASE_DIR / "logs"
ANSWERS_FP   = LOGS_DIR / "answers.jsonl"
REVIEWS_FP   = LOGS_DIR / "reviews.jsonl"
APPROVALS_FP = LOGS_DIR / "approvals.jsonl"
TRAINER_FP   = LOGS_DIR / "trainer_queue.jsonl"
OVERRIDES_FP = LOGS_DIR / "overrides.jsonl"
CALIB_FP     = BASE_DIR / "calibration.json"

def ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _append_jsonl(path: Path, rec: Dict[str, Any]):
    ensure_logs_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_rag_chunk_summaries(chunks: List[Any]) -> List[Dict[str, Any]]:
    """Принимает [(score, text, payload), ...] ИЛИ список payload-ов."""
    out = []
    if not chunks:
        return out
    for ch in chunks:
        if isinstance(ch, tuple) and len(ch) == 3:
            score, _text, payload = ch
            p = payload or {}
            out.append({
                "ticket_id": p.get("ticket_id"),
                "kind": p.get("kind"),
                "score": float(score),
                "created_at": p.get("created_at"),
                "title": p.get("title"),
            })
        elif isinstance(ch, dict):
            out.append({
                "ticket_id": ch.get("ticket_id"),
                "kind": ch.get("kind"),
                "score": ch.get("score"),
                "created_at": ch.get("created_at"),
                "title": ch.get("title"),
            })
    return out

def log_record(*, file: str, id: str, created_at: str, language: str, platform: Optional[str],
               user_text: str, rag_chunks: List[Dict[str, Any]], answer: str,
               confidence: float, top_score: float, reason: str, tags: List[str],
               agent_id: Optional[str] = None):
    if file == "answers":
        fp = ANSWERS_FP
    elif file == "reviews":
        fp = REVIEWS_FP
    elif file == "approvals":
        fp = APPROVALS_FP
    else:
        raise ValueError("file must be answers|reviews|approvals")
    rec = {
        "id": id,
        "created_at": created_at,
        "language": language,
        "platform": platform,
        "user_text": user_text,
        "rag_chunks": rag_chunks,
        "answer": answer,
        "confidence": float(confidence),
        "top_score": float(top_score),
        "reason": reason,
        "tags": tags or [],
    }
    if agent_id:
        rec["agent_id"] = agent_id
    _append_jsonl(fp, rec)

def log_review(record: Dict[str, Any]):
    _append_jsonl(REVIEWS_FP, record)

def log_approval(record: Dict[str, Any]):
    _append_jsonl(APPROVALS_FP, record)

def log_override(item: Dict[str, Any]):
    """Сохраняем обучающий оверрайд (паттерн/примеры → эталонный ответ)."""
    _append_jsonl(OVERRIDES_FP, item)

def load_overrides_from_disk() -> List[Dict[str, Any]]:
    return list(_iter_jsonl(OVERRIDES_FP)) or []

def enqueue_training(item: Dict[str, Any]):
    _append_jsonl(TRAINER_FP, item)

def load_calibration_threshold(default: float) -> float:
    try:
        with CALIB_FP.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("threshold", default))
    except Exception:
        return default
