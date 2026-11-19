# -*- coding: utf-8 -*-
"""ストロボと受信機を動的に切り替える (6 台が順番に送信役になる TDMA) 構成の識別性を、固定ストロボと比べる。
使い方: python dynamic_switch.py E:/CSI/sim_out/jac_6nodes

公平にするため「60 Hz の 1 フレームあたりに送るパケット数 (予算)」をそろえる。
同じリンクを M 回測ると、外乱 (φ0, τ, a) はパケットごとに別なので、フィッシャー情報量はリンクごとの値の M 倍になる。
"""
import sys
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else r"E:/CSI/sim_out/jac_6nodes"]
exec(open(r"e:/CSI/sim/identifiability.py", encoding="utf-8").read().split('ht40 = np.ones')[0])   # 読み込みと fisher() を共用

M = len(meta["tx"]); assert M * len(meta["rx"]) == L
ht40 = np.ones(K, bool)
link = lambda t, r: t * M + r
SNR = 25

def evaluate(plan):
    """plan: [(送信機, 回数)]。各送信機の 1 回のパケットを、自分以外の全員が受ける。戻り値: 決まる自由度, CRB"""
    ndof, crb = [], []
    for n in range(N):
        F = np.zeros((NP, NP))
        for t, rep in plan:
            F += rep * fisher(n, [link(t, r) for r in range(M) if r != t], ht40, SNR)
        Fs = F * np.outer(UNIT, UNIT); ev = np.linalg.eigvalsh(Fs); ndof.append(int((ev >= 1).sum()))
        crb.append(np.sqrt(np.diag(np.linalg.inv(Fs + 1e-6 * np.eye(NP)))) * UNIT)
    return np.array(ndof), np.median(np.array(crb), 0)

def show(label, plan):
    d, c = evaluate(plan)
    budget = sum(r for _, r in plan); nl = sum(r * (M - 1) for _, r in plan)
    bad = [names[i] for i in range(3, NP) if names[i] != "head_yaw" and c[i] > 4 * UNIT[i]]
    print(f"  {label:36s} 予算 {budget:2d}  CSI {nl:3d} 個  決まる自由度 {int(np.median(d)):2d} [{d.min():2d},{d.max():2d}]"
          f"  関節角 CRB 中央 {np.rad2deg(np.median(c[3:])):7.1f}°  20°超: {len(bad)} 個")
    return d, c

print(f"機器 {M} 台 (全員が送信も受信もできる)。各パケットの SNR {SNR} dB、HT40、姿勢 {N} 個。head_yaw は原理的に決まらないので上限 18。\n")
print("予算 1 パケット / フレーム (60 Hz で 60 パケット/秒)")
show("固定ストロボ (機器 0) → 他の 5 台", [(0, 1)])
print("予算 6 パケット / フレーム (360 パケット/秒)")
show("固定ストロボ (機器 0) を 6 回", [(0, 6)])
best = max(range(M), key=lambda t: np.median(evaluate([(t, 6)])[0]))
show(f"固定ストロボ (一番良い機器 {best}) を 6 回", [(best, 6)])
show("3 台が交代 (0, 3, 4) で 2 回ずつ", [(0, 2), (3, 2), (4, 2)])
show("6 台全員が交代 (1 回ずつ)", [(t, 1) for t in range(M)])
print("予算 12 パケット / フレーム (720 パケット/秒)")
show("固定ストロボ (機器 0) を 12 回", [(0, 12)])
show("6 台全員が交代 (2 回ずつ)", [(t, 2) for t in range(M)])

print("\n参考: 往復 (a→b と b→a) の両方を使う意味。電波の経路は同じだが、外乱と雑音はパケットごとに別")
def evaluate_links(links, rep=1):
    out = []
    for n in range(N):
        F = rep * fisher(n, links, ht40, SNR); ev = np.linalg.eigvalsh(F * np.outer(UNIT, UNIT)); out.append(int((ev >= 1).sum()))
    return np.array(out)
one_way = [link(t, r) for t in range(M) for r in range(M) if t < r]
both = [link(t, r) for t in range(M) for r in range(M) if t != r]
print(f"  片方向だけ (15 リンク) × 2 回: 決まる自由度 {int(np.median(evaluate_links(one_way, 2)))}   往復 (30 リンク) × 1 回: {int(np.median(evaluate_links(both, 1)))}")

print("\nパラメータごとの CRB (6 台全員が交代、予算 6)")
_, c = evaluate([(t, 1) for t in range(M)])
for i in range(NP):
    v = c[i] * 100 if i < 3 else np.rad2deg(c[i]); u = "cm" if i < 3 else "°"
    flag = "○" if c[i] <= UNIT[i] else ("△" if c[i] <= 4 * UNIT[i] else "×")
    print(f"  {flag} {names[i]:12s} {v:9.1f} {u}")
