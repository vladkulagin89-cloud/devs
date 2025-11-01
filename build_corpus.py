import re, json
from tqdm import tqdm
from datetime import datetime, timezone

def to_ts(iso_str):
    try:
        return int(datetime.fromisoformat(iso_str.replace("Z","+00:00")).timestamp())
    except Exception:
        return None

def normalize_record(rec):
    t = rec["ticket"]; comments = rec.get("comments", [])
    public_comments = [c for c in comments if c.get("public")]
    parts = []
    if t.get("subject"): parts.append(f"Subject: {t['subject']}")
    if t.get("description"): parts.append(t["description"])
    for c in public_comments:
        body = (c.get("body") or "").strip()
        if body: parts.append(f"Comment: {body}")
    full = "\n\n".join(parts)
    meta = {
        "id": t["id"],
        "created_at": t["created_at"],
        "created_ts": to_ts(t["created_at"]),   # ← добавили
        "updated_at": t["updated_at"],
        "status": t.get("status"),
        "priority": t.get("priority"),
        "tags": t.get("tags", []),
        "via": (t.get("via") or {}).get("channel"),
        "app_versions": extract_version(full),
    }
    return {"id": t["id"], "text": full, "meta": meta}


def extract_version(text):
    return re.findall(r"\b\d{1,2}\.\d{1,2}\.\d{1,2}\b", text)



def chunk_text(text, max_chars=4000, overlap=500):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars - overlap
    return chunks

def main():
    out=[]
    with open("tickets.jsonl","r",encoding="utf-8") as f:
        for line in tqdm(f, desc="normalize"):
            rec = normalize_record(json.loads(line))
            for idx, ch in enumerate(chunk_text(rec["text"])):
                out.append({"chunk_id": f"{rec['id']}#{idx}", "ticket_id": rec["id"], "text": ch, "meta": rec["meta"]})
    with open("corpus.jsonl","w",encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(f"done → corpus.jsonl ({len(out)} chunks)")

if __name__ == "__main__":
    main()
