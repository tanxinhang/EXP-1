import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
L = io.open("report/MVS-C_C21_report.md", encoding="utf-8").read().splitlines()
keys = ("U_FA≤α 边最优点", "真实差距", "dominance-safety", "区域恒等式",
        "Gate D1 判决", "Gate D2", "总耗时", "反例", "prune_probe_ok",
        "registered frozen")
for i, x in enumerate(L):
    if any(k in x for k in keys):
        print(i + 1, x[:320])