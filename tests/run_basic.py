"""Minimal runner (no pytest needed):  python tests/run_basic.py"""
import importlib, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ok = bad = 0
for mod in ["tests.test_preprocessing", "tests.test_evaluation"]:
    m = importlib.import_module(mod)
    for n in sorted(dir(m)):
        if n.startswith("test_"):
            try:
                getattr(m, n)(); ok += 1; print("PASS", mod, n)
            except Exception:
                bad += 1; print("FAIL", mod, n); traceback.print_exc()
print(f"{ok} passed, {bad} failed")
sys.exit(1 if bad else 0)
