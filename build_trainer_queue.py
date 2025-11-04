# build_trainer_queue.py — из approvals.jsonl -> trainer_queue.jsonl
import os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOGS_DIR      = Path(os.getenv("LOGS_DIR", "logs"))
APPROVALS     = LOGS_DIR / os.getenv("APPROVALS_LOG", "approvals.jsonl")
TRAINER_QUEUE = LOGS_DIR / os.getenv("TRAINER_QUEUE", "trainer_queue.jsonl")

def iter_jsonl(path: Path):
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

def build():
    out = []
    for rec in iter_jsonl(APPROVALS):
        ans = (rec.get("answer") or "").strip()
        user_text = (rec.get("user_text") or "").strip()
        if not ans or not user_text:
            continue
        item = {
            "id": rec.get("id"),
            "created_at": rec.get("created_at"),
            "language": rec.get("language"),
            "platform": rec.get("platform"),
            "input_text": user_text,
            "target_answer": ans,
            "rag_chunks": rec.get("rag_chunks") or [],
            "confidence": rec.get("confidence", 0.0),
            "top_score": rec.get("top_score", 0.0),
            "reason": rec.get("reason", ""),
            "tags": rec.get("tags") or [],
            "agent_id": rec.get("agent_id")
        }
        out.append(item)

    TRAINER_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with TRAINER_QUEUE.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"trainer_queue.jsonl written: {len(out)} items → {TRAINER_QUEUE}")

if __name__ == "__main__":
    build()
