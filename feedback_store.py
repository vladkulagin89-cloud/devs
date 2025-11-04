# feedback_store.py — хранение и поиск оверрайдов/фидбэка

from typing import List, Dict, Any, Optional
import re
from log_utils import log_override, enqueue_training, load_overrides_from_disk, now_iso

_OV_CACHE: Optional[List[Dict[str, Any]]] = None

def _load_cache():
    global _OV_CACHE
    if _OV_CACHE is None:
        _OV_CACHE = load_overrides_from_disk()
    return _OV_CACHE

def load_overrides() -> List[Dict[str, Any]]:
    return list(_load_cache())

def refresh_overrides():
    global _OV_CACHE
    _OV_CACHE = load_overrides_from_disk()

def register_feedback(*, user_text: str, correct_answer: str,
                      contains: Optional[List[str]] = None,
                      regex: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      platform: Optional[str] = None,
                      language: Optional[str] = None):
    """
    Регистрируем обучающий пример (оверрайд).
    contains — список подстрок, если все встречаются → матч
    regex — регулярка для матча
    """
    rec: Dict[str, Any] = {
        "created_at": now_iso(),
        "contains": contains or [],
        "regex": regex or None,
        "answer": correct_answer,
        "tags": tags or [],
        "platform": platform,
        "language": language,
    }
    log_override(rec)

    enqueue_training({
        "created_at": now_iso(),
        "input": user_text,
        "target": correct_answer,
        "contains": contains or [],
        "regex": regex or None,
        "tags": tags or [],
        "platform": platform,
        "language": language,
        "source": "feedback_register",
    })

    refresh_overrides()

def _match_by_contains(text: str, arr: List[str]) -> bool:
    t = text.lower()
    return all(s.lower() in t for s in arr)

def _match_by_regex(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None
    except re.error:
        return False

def match_override(user_text: str) -> Optional[Dict[str, Any]]:
    """Возвращает {'answer': ...} если сработал оверрайд, иначе None."""
    ovs = _load_cache()
    if not ovs:
        return None
    for ov in ovs:
        c_ok = (not ov.get("contains")) or _match_by_contains(user_text, ov.get("contains", []))
        r_ok = (not ov.get("regex")) or _match_by_regex(user_text, ov.get("regex"))
        if c_ok and r_ok:
            return {"answer": ov.get("answer", "")}
    return None
