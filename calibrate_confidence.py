# calibrate_confidence.py — учим логистическую регрессию по логам и сохраняем calibration.json
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

LOG_DIR = Path("logs")
ANS = LOG_DIR / "answers.jsonl"   # положительные примеры
REV = LOG_DIR / "reviews.jsonl"   # отрицательные примеры
OUT = Path("calibration.json")

def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z","+00:00"))

def _days_ago(iso: str) -> float:
    try:
        t = _parse_dt(iso)
        return max(0.0, (datetime.now(timezone.utc) - t).days)
    except Exception:
        return 365.0

def _feat(rec: Dict[str,Any]) -> Tuple[List[float], int]:
    top_score   = float(rec.get("top_score", 0.0))
    conf        = float(rec.get("confidence", 0.0))  # старый эвристический может быть <0, ок
    chunks      = rec.get("rag_chunks") or []
    created_at  = chunks[0].get("created_at") if chunks else None
    rec_days    = _days_ago(created_at) if created_at else 365.0
    is_sop_top  = 1.0 if (chunks and (chunks[0].get("kind") == "sop")) else 0.0

    # фичи: top_score, conf (как proxy mean_top3), recency, is_sop_top
    x = [top_score, conf, -rec_days/30.0, is_sop_top]
    return x, 0

def _read_jsonl(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def main():
    pos = _read_jsonl(ANS)
    neg = _read_jsonl(REV)

    X, y = [], []
    for r in pos:
        x,_ = _feat(r)
        X.append(x); y.append(1)
    for r in neg:
        x,_ = _feat(r)
        X.append(x); y.append(0)

    if len(X) < 20:
        print("Мало данных для калибровки (нужно хотя бы ~20 записей).")
        return

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    # подберём порог по F1
    probs = clf.predict_proba(X)[:,1]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.3, 0.8, 26):
        pred = (probs >= thr).astype(int)
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum())
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    coef = clf.coef_[0].tolist()
    inter = float(clf.intercept_[0])

    OUT.write_text(json.dumps({
        "coef": coef,                          # порядок признаков: [top_score, conf, -recency/30, is_sop_top]
        "intercept": inter,
        "threshold": float(best_thr),
        "f1_on_train": float(best_f1),
        "n_pos": int((y==1).sum()),
        "n_neg": int((y==0).sum())
    }, ensure_ascii=False, indent=2))
    print(f"Сохранено в {OUT} (threshold={best_thr:.2f}, F1={best_f1:.3f})")

if __name__ == "__main__":
    main()
