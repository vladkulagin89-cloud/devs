# chat_cli.py — CLI поверх RAG: /ask, /ticket, /search
import os, sys, json, shlex
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# грузим .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from rag_answer import route_answer, route_ticket
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLL = os.getenv("QDRANT_COLLECTION", "zendesk_6m")

client = (qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
          if QDRANT_API_KEY else qdrant_client.QdrantClient(QDRANT_URL))

HELP = """
Доступные команды:

/ask <текст вопроса>
  Вопрос к базе тикетов (RAG + LM Studio).
  Примеры:
    /ask Почему недоступны AI-функции в Иллинойсе?
    /ask Приложение вылетает на iOS 26.0.1 при открытии фото

/ticket [--id 123456] [--prompt "короткое ТЗ"] 
        (далее вставь текст тикета, завершить строкой /end)
  Ответ строго по этому тикету (LLM опирается только на тикет + RAG).
  Пример:
    /ticket --id 256920 --prompt "Дай пошаговое решение"
    Subject: App crashes on opening photo
    Diagnostics: iPhone 15 Pro | 26.0.1 | 7.2.2
    Описание: при выборе фото экран чёрный и вылет.
/end

/search [--version 7.2.2] [--tag ios] [--from 2025-10-01] [--to 2025-11-01] [--limit 50] [--json]
  Поиск тикетов по метаданным (без LLM).
  Примеры:
    /search --version 7.2.2
    /search --tag ios --from 2025-10-20 --to 2025-11-03 --limit 20
    /search --version 7.2 --json   (подбор по префиксу версии, постфильтр)
"""

def parse_kv(args):
    out = {}
    it = iter(args)
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:]
            val = next(it, None)
            out[key] = val
    return out

def _to_ts(iso_date: str, end: bool=False) -> int:
    dt = datetime.fromisoformat(iso_date)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return int(dt.timestamp())

def cmd_ask(line_rest: str):
    question = line_rest.strip()
    if not question:
        print("Уточни вопрос после /ask")
        return
    print(route_answer(question))

def cmd_ticket(args: list[str]):
    opts = parse_kv(args)
    ticket_id = None
    if "id" in opts:
        try:
            ticket_id = int(opts["id"])
        except:
            pass
    user_prompt = opts.get("prompt")

    print("Вставь текст тикета. Заверши строкой /end")
    buf = []
    for line in sys.stdin:
        if line.strip() == "/end":
            break
        buf.append(line)
    ticket_text = "".join(buf).strip()
    if not ticket_text:
        print("Пустой тикет — нечего обрабатывать.")
        return
    print(route_ticket(ticket_text, user_prompt=user_prompt, ticket_id=ticket_id))

def cmd_search(args: list[str]):
    # /search [--version X] [--tag ios|android] [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--limit N] [--json]
    opts = parse_kv(args)
    version = opts.get("version")
    tag = opts.get("tag")
    frm = opts.get("from")
    to  = opts.get("to")
    limit = int(opts.get("limit", "50"))
    as_json = ("--json" in args) or (opts.get("json") is not None)

    must = []
    if tag:
        must.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
    if frm:
        must.append(FieldCondition(key="created_ts", range=Range(gte=_to_ts(frm))))
    if to:
        must.append(FieldCondition(key="created_ts", range=Range(lte=_to_ts(to, end=True))))
    if version:
        must.append(FieldCondition(key="app_versions", match=MatchAny(any=[version])))

    flt = Filter(must=must) if must else None

    out = []
    next_page = None
    fetched = 0
    while fetched < limit:
        resp = client.scroll(collection_name=COLL, scroll_filter=flt, with_payload=True, limit=min(256, limit - fetched), offset=next_page)
        points, next_page = resp
        if not points:
            break
        for p in points:
            payload = p.payload
            out.append({
                "ticket_id": payload.get("ticket_id"),
                "created_at": payload.get("created_at"),
                "versions": payload.get("app_versions"),
                "tags": payload.get("tags"),
            })
            fetched += 1
            if fetched >= limit:
                break

    # постфильтр по префиксу версии, если указана укороченная (например "7.2")
    if version and "." in version and not any(ch.isalpha() for ch in version):
        parts = version.split(".")
        if len(parts) == 2:  # префикс 7.2
            pref = version + "."
            out = [r for r in out if any((v or "").startswith(pref) for v in (r.get("versions") or []))]

    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if not out:
            print("Ничего не найдено.")
            return
        print(f"Найдено: {len(out)} (показаны первые {min(limit, len(out))})")
        for r in out[:limit]:
            print(f"- ticket:{r['ticket_id']}  date:{r['created_at']}  versions:{r.get('versions')}  tags:{r.get('tags')}")

def main():
    print("RAG CLI запущен. Введи команду (help — подсказка).")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            print("bye")
            break
        if line == "help":
            print(HELP)
            continue

        if line.startswith("/ask"):
            cmd_ask(line[len("/ask"):])
        elif line.startswith("/ticket"):
            args = shlex.split(line)[1:]
            cmd_ticket(args)
        elif line.startswith("/search"):
            args = shlex.split(line)[1:]
            cmd_search(args)
        else:
            # по умолчанию — как /ask
            cmd_ask(line)

if __name__ == "__main__":
    main()
