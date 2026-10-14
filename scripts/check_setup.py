"""Non-network health check; prints only whether an API key exists, never the key."""

import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print("PromptForge AI - local setup check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {root}")
    missing = []
    for name in ("streamlit", "groq", "dotenv"):
        installed = importlib.util.find_spec(name) is not None
        print(f"{name}: {'OK' if installed else 'NOT INSTALLED'}")
        if not installed:
            missing.append(name)
    key = bool(os.getenv("GROQ_API_KEY", "").strip())
    if (root / ".env").is_file() and not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(root / ".env")
            key = bool(os.getenv("GROQ_API_KEY", "").strip())
        except ImportError:
            pass
    print(f"GROQ_API_KEY in environment or .env: {'SET' if key else 'NOT SET (prompt preview still works)'}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
