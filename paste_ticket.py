# paste_ticket.py — читает тикет из stdin и отдаёт в RAG-роутер
import sys
from rag_answer import route_ticket

def main():
    print("Вставь текст тикета и нажми Ctrl+D (на macOS/Linux).")
    raw = sys.stdin.read()
    if raw is None:
        raw = ""
    if not raw.strip():
        # минимальный инпут, чтобы отработал intent=empty
        raw = " "

    print("\n=== Ответ ===\n")
    print(route_ticket(raw))

if __name__ == "__main__":
    main()
