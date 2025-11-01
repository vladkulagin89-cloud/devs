# export_zendesk_windowed_adaptive.py
import os, time, json, requests
from urllib.parse import urlencode
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()
SUB   = os.environ["ZENDESK_SUBDOMAIN"]
EMAIL = os.environ["ZENDESK_EMAIL"]
TOKEN = os.environ["ZENDESK_API_TOKEN"]
DATE_FROM = os.environ["DATE_FROM"]   # 'YYYY-MM-DD'
DATE_TO   = os.environ["DATE_TO"]     # 'YYYY-MM-DD' (верхняя граница исключительная по ISO, мы включим +1 день)

AUTH = (f"{EMAIL}/token", TOKEN)
BASE = f"https://{SUB}.zendesk.com/api/v2"
OUTPUT = "tickets.jsonl"

# Порог, при котором дробим окно дальше
SPLIT_THRESHOLD = 900   # безопасно ниже лимита ~1000
PER_PAGE = 100

def backoff(r):
    if r.status_code == 429:
        retry = int(r.headers.get("Retry-After", "2"))
        time.sleep(retry)
        return True
    return False

def search_count(frm, to):
    q = f'type:ticket created>{frm} created<{to}'
    while True:
        r = requests.get(f"{BASE}/search/count.json", params={"query": q}, auth=AUTH, timeout=60)
        if backoff(r):
            continue
        r.raise_for_status()
        return r.json().get("count", 0)

def month_windows(d_from, d_to_exclusive):
    cur = datetime.fromisoformat(d_from).replace(day=1)
    end = datetime.fromisoformat(d_to_exclusive)
    while cur < end:
        nxt = min(cur + relativedelta(months=1), end)
        yield cur.date().isoformat(), nxt.date().isoformat()
        cur = nxt

def week_windows(frm, to):
    start = datetime.fromisoformat(frm)
    end   = datetime.fromisoformat(to)
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        yield cur.date().isoformat(), nxt.date().isoformat()
        cur = nxt

def day_windows(frm, to):
    start = datetime.fromisoformat(frm)
    end   = datetime.fromisoformat(to)
    cur = start
    while cur < end:
        nxt = cur + timedelta(days=1)
        yield cur.date().isoformat(), min(nxt, end).date().isoformat()
        cur = nxt

def search_ids_window(frm, to):
    ids, page = [], 1
    while True:
        q = f'type:ticket created>{frm} created<{to}'
        params = {"query": q, "sort_by": "created_at", "sort_order": "asc", "page": page, "per_page": PER_PAGE}
        r = requests.get(f"{BASE}/search.json?{urlencode(params)}", auth=AUTH, timeout=60)
        if backoff(r):
            continue
        # 422 при переполнении окна — вернёмся вверх и раздробим
        if r.status_code == 422:
            raise RuntimeError("422-too-many")
        r.raise_for_status()
        data = r.json()
        for hit in data.get("results", []):
            if hit.get("result_type") == "ticket":
                ids.append(hit["id"])
        if not data.get("next_page"):
            break
        page += 1
        time.sleep(0.05)
    return ids

def adaptive_collect_ids(frm, to):
    """
    Возвращает список id для окна [frm, to) c адаптивным дроблением:
    месяц -> недели -> дни, пока count <= SPLIT_THRESHOLD
    """
    cnt = search_count(frm, to)
    if cnt <= SPLIT_THRESHOLD:
        try:
            return search_ids_window(frm, to)
        except RuntimeError:
            # если всё равно 422, дробим дальше
            pass

    span_days = (datetime.fromisoformat(to) - datetime.fromisoformat(frm)).days
    if span_days > 14:
        # дробим месяц на недели
        out = []
        for wfrm, wto in week_windows(frm, to):
            out += adaptive_collect_ids(wfrm, wto)
        return out
    elif span_days > 1:
        # дробим неделю на дни
        out = []
        for dfrm, dto in day_windows(frm, to):
            out += adaptive_collect_ids(dfrm, dto)
        return out
    else:
        # один день всё ещё большой — крайний случай: оставляем как есть и пробуем вытащить
        return search_ids_window(frm, to)

def get_ticket(tid):
    while True:
        r = requests.get(f"{BASE}/tickets/{tid}.json", auth=AUTH, timeout=60)
        if backoff(r):
            continue
        r.raise_for_status()
        return r.json()["ticket"]

def get_comments(tid):
    while True:
        r = requests.get(f"{BASE}/tickets/{tid}/comments.json?include=events", auth=AUTH, timeout=60)
        if backoff(r):
            continue
        r.raise_for_status()
        return r.json().get("comments", [])

def main():
    # включим полный последний день
    date_from = DATE_FROM
    date_to_excl = (datetime.fromisoformat(DATE_TO) + timedelta(days=1)).date().isoformat()

    seen = set()
    total_ids = 0
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for mfrm, mto in month_windows(date_from, date_to_excl):
            print(f"month {mfrm}..{mto}")
            ids = adaptive_collect_ids(mfrm, mto)
            print(f"  ids in month window: {len(ids)}")
            for tid in ids:
                if tid in seen:
                    continue
                seen.add(tid)
                t = get_ticket(tid)
                c = get_comments(tid)
                f.write(json.dumps({"ticket": t, "comments": c}, ensure_ascii=False) + "\n")
                total_ids += 1
                if total_ids % 100 == 0:
                    print(f"...exported {total_ids}")
                    time.sleep(0.02)
    print(f"done → {OUTPUT}  total_exported:{total_ids}")

if __name__ == "__main__":
    main()
