# paste_ticket.py — вставь текст тикета, получи ответ, затем: approve/decline/modify
import sys, argparse, uuid
from typing import List
from rag_answer import route_ticket_meta, detect_language, extract_platform
from log_utils import log_record, build_rag_chunk_summaries
from feedback_store import register_feedback

def read_block(prompt: str) -> str:
    print(prompt)
    print("(введите /end на отдельной строке для завершения)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "/end":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def gen_id() -> str:
    return str(uuid.uuid4())

def pretty_print_answer(ans: str, meta: dict):
    print("\n=== ОТВЕТ ===\n")
    print(ans)
    print(f"\n(confidence={meta.get('confidence', 0):.2f}, top_score={meta.get('top_score', 0):.2f})\n")

def auto_contains(user_text: str, k: int = 3) -> List[str]:
    import re
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9_'-]{5,}", user_text.lower())
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    return uniq[:k]

def log_to_jsonl(kind: str, *, log_id: str, ticket_text: str, answer: str, meta: dict):
    rag_chunks = build_rag_chunk_summaries(meta.get("chunks") or [])
    lang   = meta.get("language") or detect_language(ticket_text)
    plat   = meta.get("platform") or extract_platform(ticket_text)
    reason = meta.get("reason", "ok")

    log_record(
        file=kind,  # "answers" | "reviews" | "approvals"
        id=log_id,
        created_at=meta.get("created_at") or "",
        language=lang,
        platform=plat,
        user_text=ticket_text,
        rag_chunks=rag_chunks,
        answer=answer,
        confidence=float(meta.get("confidence", 0.0)),
        top_score=float(meta.get("top_score", 0.0)),
        reason=reason,
        tags=[],
        agent_id="vlad",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help="ticket_id (опц.)")
    parser.add_argument("--prompt", type=str, default=None, help="доп. промпт (опц.)")
    args = parser.parse_args()

    ticket_text = read_block("Вставьте текст тикета.")
    if not ticket_text:
        print("Пусто. Нечего обрабатывать.", file=sys.stderr)
        sys.exit(1)

    answer, meta = route_ticket_meta(ticket_text, user_prompt=args.prompt, ticket_id=args.id)
    pretty_print_answer(answer, meta)

    low_conf = meta.get("reason") in ("low_conf", "no_candidates")

    print("Действия: [a] approve  [d] decline & teach  [m] modify with hint  [q] quit")
    if low_conf:
        print("(подсказка: low confidence — удобнее сразу выбрать d/m)")

    while True:
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            # если внезапно закрыли ввод — просто сохраним ответ
            log_to_jsonl("answers", log_id=args.id or gen_id(), ticket_text=ticket_text, answer=answer, meta=meta)
            print("\n(ответ записан в logs/answers.jsonl)")
            break

        if choice == "a":
            log_to_jsonl("answers", log_id=args.id or gen_id(), ticket_text=ticket_text, answer=answer, meta=meta)
            log_to_jsonl("approvals", log_id=args.id or gen_id(), ticket_text=ticket_text, answer=answer, meta=meta)
            print("✅ Approved. Записано в logs/answers.jsonl и logs/approvals.jsonl")
            break

        elif choice == "d":
            corrected = read_block("Введите эталонный корректный ответ.")
            if not corrected:
                print("Пустой эталонный ответ — отмена.")
                continue

            contains_line = input("Ключевые подстроки для матча (через запятую, можно пусто): ").strip()
            contains = [s.strip() for s in contains_line.split(",") if s.strip()] if contains_line else auto_contains(ticket_text, 3)

            register_feedback(
                user_text=ticket_text,
                correct_answer=corrected,
                contains=contains,
                regex=None,
                tags=[],
                platform=meta.get("platform"),
                language=meta.get("language"),
            )

            new_answer, new_meta = route_ticket_meta(ticket_text, user_prompt=args.prompt, ticket_id=args.id)
            print("\n=== Обновлённый ответ (после обучения) ===\n")
            print(new_answer)
            print(f"\n(confidence={new_meta.get('confidence', 0):.2f}, top_score={new_meta.get('top_score', 0):.2f})\n")
            log_to_jsonl("answers", log_id=args.id or gen_id(), ticket_text=ticket_text, answer=new_answer, meta=new_meta)
            print("🧠 Оверрайд сохранён, ответ записан в logs/answers.jsonl.")
            break

        elif choice == "m":
            hint = input("Краткий хинт для бота: ").strip()
            hinted_prompt = (args.prompt or "Compose a helpful support reply.") + f"\n\nHint: {hint}"
            new_answer, new_meta = route_ticket_meta(ticket_text, user_prompt=hinted_prompt, ticket_id=args.id)
            print("\n=== Обновлённый ответ (с учётом хинта) ===\n")
            print(new_answer)
            print(f"\n(confidence={new_meta.get('confidence', 0):.2f}, top_score={new_meta.get('top_score', 0):.2f})\n")
            log_to_jsonl("answers", log_id=args.id or gen_id(), ticket_text=ticket_text, answer=new_answer, meta=new_meta)
            print("✏️ Ответ записан в logs/answers.jsonl (без обучения).")
            # остаёмся в цикле: можно ещё a/d/m/q

        elif choice == "q":
            print("Выход без сохранения.")
            break
        else:
            print("Варианты: a / d / m / q")

if __name__ == "__main__":
    main()
