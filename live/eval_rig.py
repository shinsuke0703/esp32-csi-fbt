# -*- coding: utf-8 -*-
"""5 リンク実機 (C3 ストロボ → DevKitC 5 台) の CSI で CSI-OPQ を評価する。
使い方: python eval_rig.py rig_motion.npz rig_static.npz

評価 (どれもラベル不要):
 1. 復元誤差: リンクごとに、前 60% で符号器を学習、後 40% で NMSE / 振幅 NMSE。
 2. 別リンクへの汎化: 4 リンクで学習した符号器を残り 1 リンクに当てる (同じリンクで学習した場合との差)。
 3. リンク間予測 (情報が残ったかの代わり): リンク A の振幅からリンク B の振幅の主成分 (動きの成分) をリッジ回帰で予測。
    回帰は元の CSI で学習し、評価時に A を「元」と「復元」で入れ替えて R² を比べる。保持率 = R²(復元) / R²(元)。
 4. 動きの検出: 1 秒窓の振幅の揺らぎで「静止」と「歩行」を見分ける AUC を、元と復元で比べる。
"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, r"e:/SPINEX/moetrace/csi_test")
import csi_opq as co

co.NSC = 56; co.D = 112
FFT_IDX = list(range(36, 64)) + list(range(1, 29))   # classic ESP32 の HT-LTF (FFT 順) → 周波数順 -28..-1, 1..28
SPLIT = 0.6
BPRS = (0.5, 1, 2)

def load(path):
    d = np.load(path); out = {}
    for p in d["ports"]:
        p = str(p); ss = d[f"{p}_sseq"]; ok = ss != 0xFFFFFFFF
        r = d[f"{p}_raw"][ok].astype(np.float64)[:, 128:256]
        c = r[:, 1::2] + 1j * r[:, 0::2]
        ss = ss[ok]; o = np.argsort(ss, kind="stable"); ss, c = ss[o], c[o]
        u = np.concatenate([[True], np.diff(ss) != 0]); out[p] = (ss[u], c[u][:, FFT_IDX])
    return out

mot = load(sys.argv[1]); sta = load(sys.argv[2]) if len(sys.argv) > 2 else None
ports = list(mot)
s_all = np.concatenate([mot[p][0] for p in ports]); s_cut = np.quantile(s_all, SPLIT)
print(f"リンク {ports}  フレーム数 " + ", ".join(f"{p}:{len(mot[p][0])}" for p in ports) + f"  学習/評価の境目 seq={s_cut:.0f}")
for p in ports:
    _, S = co.canon(mot[p][1]); d = np.angle(np.exp(1j * np.diff(S[:, 0])))
    A = np.abs(mot[p][1]); print(f"  {p}: 共通位相の隣接差 std {d.std():.2f} rad  振幅の変動 (std/mean) {(A.std(0) / A.mean(0)).mean():.3f}")

def split(p, D=mot):
    ss, H = D[p]; tr = ss < s_cut; return (ss[tr], H[tr]), (ss[~tr], H[~tr])

CFGS = {"K+E (PCA+E8)": dict(use_N=False, use_P=False), "N+K+E": dict(use_P=False), "N+K+P+E (提案)": dict()}
res = []

# ---------- 1. 復元誤差 (同じリンクで学習) と 2. 別リンクへの汎化 ----------
recon = {}   # (cfg, bpr, port) → 評価区間の復元 H
for name, kw in CFGS.items():
    for bpr in BPRS:
        t0 = time.time(); same, loo = [], []
        cds = []
        for p in ports:
            (_, Htr), _ = split(p); cd = co.Codec(Htr, **kw); cd.allocate(bpr * co.D); cds.append(cd)
        outs = co.run_many(cds, [split(p)[1][1] for p in ports]) if kw.get("use_P", True) else [cd.run(split(p)[1][1]) for cd, p in zip(cds, ports)]
        for p, Hh in zip(ports, outs):
            Hte = split(p)[1][1]; recon[(name, bpr, p)] = Hh; same.append((co.nmse(Hte, Hh), co.nmse_amp(Hte, Hh)))
        lcds = []
        for p in ports:   # 別リンクへの汎化: 他の 4 リンクの学習区間をつないで学習
            Htr = np.concatenate([split(q)[0][1] for q in ports if q != p]); cd = co.Codec(Htr, **kw); cd.allocate(bpr * co.D); lcds.append(cd)
        louts = co.run_many(lcds, [split(p)[1][1] for p in ports]) if kw.get("use_P", True) else [cd.run(split(p)[1][1]) for cd, p in zip(lcds, ports)]
        for p, Hh in zip(ports, louts):
            Hte = split(p)[1][1]; loo.append((co.nmse(Hte, Hh), co.nmse_amp(Hte, Hh)))
        same, loo = np.array(same), np.array(loo)
        r = {"手法": name, "bit/実数": bpr, "ビット/フレーム": int(cds[0].used),
             "NMSE 同リンク": round(float(same[:, 0].mean()), 2), "振幅NMSE 同リンク": round(float(same[:, 1].mean()), 2),
             "NMSE 別リンク": round(float(loo[:, 0].mean()), 2), "振幅NMSE 別リンク": round(float(loo[:, 1].mean()), 2),
             "リンク別 NMSE (同)": [round(float(v), 1) for v in same[:, 0]], "秒": round(time.time() - t0, 1)}
        print(json.dumps(r, ensure_ascii=False), flush=True); res.append(r)

# ---------- 3. リンク間予測 ----------
def feat(H):   # 振幅を対数にしてサブキャリアごとに使う
    return np.log1p(np.abs(H))
def smooth(X, w=10):   # 0.1 s の移動平均 (100 Hz)
    k = np.ones(w) / w; return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, X)
def r2(Y, P): return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()

common_tr = set.intersection(*[set(split(p)[0][0].tolist()) for p in ports])
common_te = set.intersection(*[set(split(p)[1][0].tolist()) for p in ports])
def pick(ss, X, keep):
    m = np.isin(ss, np.array(sorted(keep))); return X[m]
NPC = 8
pair_rows = []
for a in ports:
    (sa_tr, Ha_tr), (sa_te, Ha_te) = split(a)
    Xa_tr = pick(sa_tr, feat(Ha_tr), common_tr); Xa_te = pick(sa_te, feat(Ha_te), common_te)
    m, s = Xa_tr.mean(0), Xa_tr.std(0) + 1e-9
    for b in ports:
        if b == a: continue
        (sb_tr, Hb_tr), (sb_te, Hb_te) = split(b)
        Yb_tr = smooth(pick(sb_tr, feat(Hb_tr), common_tr)); Yb_te = smooth(pick(sb_te, feat(Hb_te), common_te))
        ym = Yb_tr.mean(0); _, _, Vt = np.linalg.svd(Yb_tr - ym, full_matrices=False); V = Vt[:NPC]
        Ttr, Tte = (Yb_tr - ym) @ V.T, (Yb_te - ym) @ V.T
        Ztr = smooth((Xa_tr - m) / s); W = np.linalg.solve(Ztr.T @ Ztr + 100 * np.eye(Ztr.shape[1]), Ztr.T @ Ttr)
        row = {"A": a, "B": b, "R2 元": r2(Tte, smooth((Xa_te - m) / s) @ W)}
        for name in CFGS:
            for bpr in BPRS:
                Xr = pick(sa_te, feat(recon[(name, bpr, a)]), common_te)
                row[(name, bpr)] = r2(Tte, smooth((Xr - m) / s) @ W)
        pair_rows.append(row)
r2raw = np.array([r["R2 元"] for r in pair_rows])
print(f"\nリンク間予測 (A の振幅 → B の振幅の主成分 {NPC} 個、{len(pair_rows)} 組の平均): R² 元の CSI {r2raw.mean():.3f} (最小 {r2raw.min():.3f}, 最大 {r2raw.max():.3f})")
for name in CFGS:
    for bpr in BPRS:
        v = np.array([r[(name, bpr)] for r in pair_rows])
        rr = {"手法": name, "bit/実数": bpr, "リンク間 R² (復元)": round(float(v.mean()), 3), "ΔR² (復元 - 元)": round(float(v.mean() - r2raw.mean()), 3),
              "保持率": round(float(v.mean() / r2raw.mean()), 3) if r2raw.mean() > 0.05 else None}
        print(json.dumps(rr, ensure_ascii=False), flush=True); res.append(rr)

# ---------- 4. 動きの検出 (静止 vs 歩行) ----------
if sta:
    def win_score(H, w=100):   # 1 秒窓ごとの、振幅 (対数) の時間方向の揺らぎ (サブキャリア平均)
        X = feat(H); n = len(X) // w; return np.array([X[i * w:(i + 1) * w].std(0).mean() for i in range(n)])
    def auc(neg, pos): return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    print("\n動きの検出 AUC (1 秒窓、リンク平均。静止の収録全体 vs 歩行の評価区間)")
    base = {}
    for p in ports:
        base[p] = win_score(sta[p][1])
    raw_auc = np.mean([auc(base[p], win_score(split(p)[1][1])) for p in ports])
    print(f"  元の CSI: {raw_auc:.3f}")
    for name, kw in CFGS.items():
        for bpr in BPRS:
            scds = []
            for p in ports:   # 静止の収録も同じ符号器 (歩行の学習区間で学習) で符号化する
                (_, Htr), _ = split(p); cd = co.Codec(Htr, **kw); cd.allocate(bpr * co.D); scds.append(cd)
            souts = co.run_many(scds, [sta[p][1] for p in ports]) if kw.get("use_P", True) else [cd.run(sta[p][1]) for cd, p in zip(scds, ports)]
            a_list = [auc(win_score(Hs_), win_score(recon[(name, bpr, p)])) for p, Hs_ in zip(ports, souts)]
            rr = {"手法": name, "bit/実数": bpr, "動き検出 AUC (復元)": round(float(np.mean(a_list)), 3), "元": round(float(raw_auc), 3)}
            print(json.dumps(rr, ensure_ascii=False), flush=True); res.append(rr)

json.dump(res, open(sys.argv[1].replace(".npz", "_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
