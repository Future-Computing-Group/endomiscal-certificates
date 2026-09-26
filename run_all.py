#!/usr/bin/env python3
"""Run every certificate. Exit status is non-zero if any assertion fails."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = sorted(p.name for p in HERE.glob("cert_*.py"))
assert SCRIPTS, "no certificate scripts found beside run_all.py"

def main():
    failed = []
    for s in SCRIPTS:
        print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")
        rc = subprocess.call([sys.executable, "-u", str(HERE / s)])
        if rc != 0:
            failed.append(s)
    print(f"\n{'=' * 72}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"ALL {len(SCRIPTS)} CERTIFICATES PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
