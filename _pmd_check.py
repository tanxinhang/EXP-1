# -*- coding: utf-8 -*-
"""Check P_MD settings and achieved values across 口径."""
import io
import sys
sys.stdout.reconfigure(encoding="utf-8")

# 1. C3b/C3c mechanism (G2) beta target
import run_mvsb07g2 as g2
print("G2/C3b/C3c mechanism QoS: alpha =", g2.ALPHA, " beta(P_MD目标) =", g2.BETA)

# 2. Paper matched beta (C2.1: BETA8 = 1 - P_D,max^det-thr,8b + eps)
import run_mvsc021 as c21
print("\nC2.1 matched: ALPHA =", c21.ALPHA, " EPS_D =", c21.EPS_D)
print("BETA8 (matched P_MD 目标) = 1 - 0.8509 + 0.01 =",
      1 - 0.8509 + c21.EPS_D)
print("BETA4 (matched P_MD 目标) = 1 - 0.8482 + 0.01 =",
      1 - 0.8482 + c21.EPS_D)

# 3. Actual achieved P_MD in C3b FULL (H=96)
t = io.open("report/MVS-C_C3b_report.md", encoding="utf-8").read()
i = t.find("## 2. Test @ H=96")
print("\nC3b FULL H=96 (五方法，mechanism 口径 α=0.12/β=0.40):")
print(t[i:i+900].replace("\n", " "))

# 4. C2.1 matched verdict (N=4) - gap to beta8
t2 = io.open("report/MVS-C_C21_report.md", encoding="utf-8").read()
j = t2.find("3.4 matched")
print("\nC2.1 matched §3.4 (N=4, α=0.05, β8=0.1591):")
print(t2[j:j+600].replace("\n", " "))