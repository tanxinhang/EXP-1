# -*- coding: utf-8 -*-
"""Extract performance comparison numbers from generated reports."""
import io
import sys
sys.stdout.reconfigure(encoding="utf-8")

def grab(path, needle, width=1800):
    t = io.open(path, encoding="utf-8").read()
    i = t.find(needle)
    if i < 0:
        return None
    return t[i:i + width].replace("\n", " ")

# C3b FULL: five-method comparison
t = io.open("report/MVS-C_C3b_report.md", encoding="utf-8").read()
print("=== C3b FULL: Test @ H=96 table ===")
i = t.find("## 2. Test @ H=96")
print(t[i:i + 1500].replace("\n", " "))
print()
print("=== C3b FULL: causal pairs ===")
i = t.find("## 4. 四层因果对照")
print(t[i:i + 1400].replace("\n", " "))
print()
print("=== C3b FULL: calibration ===")
i = t.find("## 1. Calibration")
print(t[i:i + 900].replace("\n", " "))