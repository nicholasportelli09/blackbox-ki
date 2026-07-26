"""
One-time setup: seeds logs/ with the precomputed catalog + a sentiment snapshot so
the app runs immediately, with no API keys and no waiting for a first scan.

    python bootstrap.py

Then:  python app_user.py   ->  http://localhost:8090
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
DATA = os.path.join(HERE, "data")

os.makedirs(LOGS, exist_ok=True)

for f in ("catalog.json", "sentiment.json"):
    src, dst = os.path.join(DATA, f), os.path.join(LOGS, f)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"  seeded logs/{f}")
    else:
        print(f"  logs/{f} already exists - kept")

for f, default in (("app_users.json", "{}"), ("app_accounts.json", "{}")):
    p = os.path.join(LOGS, f)
    if not os.path.exists(p):
        open(p, "w").write(default)
        print(f"  created empty logs/{f}")

missing = []
for mod in ("flask", "requests", "pandas", "numpy"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)

print()
if missing:
    print("MISSING PACKAGES:", ", ".join(missing))
    print("   run:  pip install -r requirements.txt")
    sys.exit(1)

has_env = os.path.exists(os.path.join(HERE, ".env"))
print("Setup complete.")
print("   Catalog + sentiment seeded - the web app works right now:")
print("       python app_user.py     ->  http://localhost:8090")
print()
if has_env:
    print("   .env found - live market marks enabled.")
else:
    print("   Live per-strategy holdings (the personal Blackbox panel) need a")
    print("   free OANDA practice key in .env. Everything else - strategies,")
    print("   backtests, projections, quiz, accounts, AI scanner - runs with")
    print("   no credentials at all.")
