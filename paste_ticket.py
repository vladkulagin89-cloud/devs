# paste_ticket.py — вставляешь тикет, получаешь готовый ответ в стиле Vlad/Prisma
import sys
from rag_answer import route_ticket

def read_stdin() -> str:
    print("Вставь текст тикета и нажми Ctrl+D (на macOS/Linux).")
    return sys.stdin.read().strip()

if __name__ == "__main__":
    ticket = read_stdin()
    if not ticket:
        print("Пустой ввод.")
        sys.exit(0)
    # user_prompt можно не задавать — формат подтягивается из assistant_prefill
    ans = route_ticket(ticket)
    print("\n=== Ответ ===\n")
    print(ans)
