# z0_test_conn.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

SUB = os.environ["ZENDESK_SUBDOMAIN"]
EMAIL = os.environ["ZENDESK_EMAIL"]
TOKEN = os.environ["ZENDESK_API_TOKEN"]
AUTH = (f"{EMAIL}/token", TOKEN)
BASE = f"https://{SUB}.zendesk.com/api/v2"

r = requests.get(f"{BASE}/users/me.json", auth=AUTH, timeout=30)
r.raise_for_status()
me = r.json()["user"]
print("OK:", me["email"], "role:", me.get("role"))
