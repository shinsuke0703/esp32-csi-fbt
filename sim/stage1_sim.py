# -*- coding: utf-8 -*-
"""PDF (Gemini 作成) の「段階 1: 無同期の位相の乱れを消す」が、全身の識別性をどれだけ戻すかをフィッシャー情報量で検証する。

配置: 3D メッシュ 6 台、全員が交代で 1 回ずつ送信 (送信はアンテナ A)。受信はアンテナ A (と B)。
外乱 (ESP32 で未知): 共通位相 φ0、位相の傾き τ、利得 a。どの観測とどの観測が外乱を共有するかだけを変えて比べる。
外乱の扱い: 同じ外乱を共有する観測をまとめて縦に並べ、外乱の方向を射影で取り除く (外乱を未知のまま同時推定するのと同じ)。

使い方: python stage1_sim.py [SNR=25]
"""
import sys, json
import numpy as np

SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 25
UNIT = np.array([0.05] * 3 + [np.deg2rad(5)] * 16)
GROUPS = {"位置": [0, 1, 2], "向き・背骨": [3, 4, 5], "肩": [7, 8, 10, 11], "肘": [9, 12], "股関節": [13, 14, 16, 17], "膝": [15, 18]}

base = r"E:/CSI/sim_out/jac_mesh3d_2ant"; meta = json.load(open(base + ".json", encoding="utf-8"))
N, L, K, NP = meta["N"], meta["L"], meta["K"], meta["NP"]
a = np.fromfile(base + ".bin", dtype=np.float64); o = N * NP
H0 = a[o:o + N * L * K * 2].reshape(N, L, K, 2); o += N * L * K * 2
J = a[o:].reshape(N, NP, L, K, 2)
H0 = H0[..., 0] + 1j * H0[..., 1]; J = J[..., 0] + 1j * J[..., 1]
k = np.array(meta["tones"], float)
NRX = len(meta["rx"]); M = len(meta["tx"])           # 受信アンテナ 12 = A (0..5) + B (6..11)
lk = lambda t, r: t * NRX + r
R = lambda z: np.concatenate([z.real, z.imag], 0)

def group_fisher(n, obs, nuis, snr_gain_db=0.0):
    """obs: [リンク番号] (同じパケットで外乱を共有しうる観測の組)。
    nuis: 外乱の種類のリスト。'phi' / 'tau' = 組全体で共通、'phi_each' / 'tau_each' / 'a_each' = 観測ごとに別、'a_common' = 組全体で共通。"""
    hs = [H0[n, l] for l in obs]; Js = [J[n][:, l].T for l in obs]          # [K], [K, NP]
    Kk = len(k); G = len(obs)
    Jr = np.concatenate([R(Ji) for Ji in Js], 0)                              # [G*2K, NP]
    cols = []
    def block(vals):   # 観測ごとのベクトルを組の長さに並べる (実部・虚部の順を観測ごとに)
        return np.concatenate([R(v) for v in vals], 0)
    for kind in nuis:
        if kind == "phi": cols.append(block([1j * h for h in hs]))
        if kind == "tau": cols.append(block([1j * k * h for h in hs]))
        if kind == "a_common": cols.append(block(hs))
        for g in range(G):
            z = [np.zeros(Kk, complex) for _ in range(G)]
            if kind == "phi_each": z[g] = 1j * hs[g]; cols.append(block(z))
            if kind == "tau_each": z[g] = 1j * k * hs[g]; cols.append(block(z))
            if kind == "a_each": z[g] = hs[g]; cols.append(block(z))
    if cols:
        Q, _ = np.linalg.qr(np.stack(cols, 1)); Jr = Jr - Q @ (Q.T @ Jr)
    W = np.concatenate([np.full(2 * Kk, 1.0 / ((np.abs(h) ** 2).mean() / 10 ** ((SNR + snr_gain_db) / 10) / 2)) for h in hs])
    return (Jr * W[:, None]).T @ Jr

def scenario(n, kind):
    F = np.zeros((NP, NP))
    for t in range(M):
        others = [r for r in range(M) if r != t]
        if kind == "esp32":            # 今: アンテナ 1 本、リンクごとに外乱が独立
            for r in others: F += group_fisher(n, [lk(t, r)], ["phi_each", "tau_each", "a_each"])
        elif kind == "2ant_indep":     # アンテナ 2 本、時計は別々 (アンテナが増えた効果だけ)
            for r in others: F += group_fisher(n, [lk(t, r), lk(t, r + M)], ["phi_each", "tau_each", "a_each"])
        elif kind == "2ant_shared":    # アンテナ 2 本、時計を共有 (esp-crab 型): φ0・τ は共通、利得は別
            for r in others: F += group_fisher(n, [lk(t, r), lk(t, r + M)], ["phi", "tau", "a_each"])
        elif kind == "ideal":          # 外乱がすべて既知
            for r in others: F += group_fisher(n, [lk(t, r)], [])
    if kind.startswith("bidir"):       # 往復の幾何平均: 無向の辺ごとに 1 観測 (2 パケット分なので SNR +3 dB)
        rest = {"bidir_phi": ["tau_each", "a_each"], "bidir_phi_tau": ["a_each"]}[kind]
        F = np.zeros((NP, NP))
        for t in range(M):
            for r in range(t + 1, M): F += group_fisher(n, [lk(t, r)], rest, snr_gain_db=10 * np.log10(2))
    return F

LABEL = {"esp32": "今の ESP32 (アンテナ 1 本)", "2ant_indep": "アンテナ 2 本・時計は別々", "2ant_shared": "アンテナ 2 本・時計を共有 (esp-crab 型)",
         "bidir_phi": "往復の幾何平均で φ0 が消える (τ は残る)", "bidir_phi_tau": "往復で φ0 も τ も消える", "ideal": "理想 (外乱がすべて既知)"}
print(f"3D メッシュ 6 台、全員が交代で 1 回ずつ送信、HT40、SNR {SNR:.0f} dB、{N} 姿勢の中央値。CRB: 位置 cm / 角度 °")
print(f"  {'構成':42s} 決まる自由度   " + "  ".join(f"{g:>6s}" for g in GROUPS))
for kind in LABEL:
    nd, crb = [], []
    for n in range(N):
        Fs = scenario(n, kind) * np.outer(UNIT, UNIT); nd.append(int((np.linalg.eigvalsh(Fs) >= 1).sum()))
        crb.append(np.sqrt(np.diag(np.linalg.inv(Fs + 1e-6 * np.eye(NP)))) * UNIT)
    c = np.median(np.array(crb), 0); nd = np.array(nd)
    cells = [f"{c[idx].max() * 100:6.1f}" if g == "位置" else f"{np.rad2deg(c[idx].max()):6.1f}" for g, idx in GROUPS.items()]
    print(f"  {LABEL[kind]:42s} {int(np.median(nd)):2d} [{nd.min():2d},{nd.max():2d}]   " + "  ".join(cells), flush=True)
