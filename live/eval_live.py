"""実機 (DevKitC ストロボ → C3) の CSI で CSI-OPQ を評価。使い方: python eval_live.py motion120.npz"""
import sys, json, time, math, numpy as np, torch
sys.path.insert(0, r"e:/SPINEX/moetrace/csi_test")
import csi_opq as co
# HT20 HT-LTF: バッファの後半 128 バイト = 64 本 (周波数順 -32..31、[虚部, 実部])。有効は -28..-1, 1..28 の 56 本
co.NSC = 56; co.D = 112
d = np.load(sys.argv[1]); r = d["raw"].astype(np.float64)[:, 128:256]
c = r[:, 1::2] + 1j * r[:, 0::2]; idx = list(range(4, 32)) + list(range(33, 61)); H = c[:, idx]
seq = d["seq"]; print(f"フレーム {len(H)}  seq 抜け {(np.diff(seq) != 1).sum()}  RSSI {d['rssi'].mean():.1f}±{d['rssi'].std():.1f} dBm  値の範囲 {r.min():.0f}..{r.max():.0f}")
A = np.abs(H); print(f"振幅の時間変動 (サブキャリア平均の 標準偏差/平均): {(A.std(0) / A.mean(0)).mean():.3f}")
T = len(H); cut = int(T * 0.6); Htr, Hte = H[:cut], H[cut:]
_, S = co.canon(Htr); dphi = np.angle(np.exp(1j * np.diff(S[:, 0])))
print(f"共通位相の隣接差 std {dphi.std():.2f} rad (一様なら 1.81)  τ: {S[:,1].mean():.4f}±{S[:,1].std():.4f}  log a std {S[:,2].std():.3f}")
def opq5_raw(H, bits):
    X = np.zeros((len(H), 128)); X[:, :co.D] = co.c2r(H)
    ote = co.ote; blob, sc, _ = ote.opq_encode_matrix_t(torch.tensor(X), bits, 42, scale_mode="mse", device="cpu", transform="rht")
    fm, um = co.KLEIN_PARAMS[bits]; tab = ote.build_g_table_t(bits, um, device="cpu")
    return co.r2c(ote.opq_decode_matrix_t(blob, len(X), 128, bits, tab, sc, device="cpu", transform="rht", seed=42).numpy()[:, :co.D])
res = []
def rec(name, bpr, used, Hh, Hx=Hte, extra=None):
    r = {"手法": name, "bit/実数": bpr, "ビット/フレーム": int(used), "NMSE": round(co.nmse(Hx, Hh), 2), "振幅NMSE": round(co.nmse_amp(Hx, Hh), 2)}
    if extra: r.update(extra)
    print(json.dumps(r, ensure_ascii=False), flush=True); res.append(r)
for b in (2, 3, 4):
    rec("OPQ5 そのまま", b, b * 128 + 32, opq5_raw(Hte, b))
cfgs = {"K+E (PCA+E8)": dict(use_N=False, use_P=False), "K+P+E": dict(use_N=False), "N+K+E": dict(use_P=False),
        "N+K+P+E (提案)": dict(), "N+K+P+一様": dict(quant="uni")}
for name, kw in cfgs.items():
    for bpr in (0.5, 1, 2, 3):
        cd = co.Codec(Htr, **kw); cd.allocate(bpr * co.D); t0 = time.time()
        rec(name, bpr, cd.used, cd.run(Hte), extra={"秒": round(time.time() - t0, 1)})
cd0 = co.Codec(Htr); print("提案の ρ (エネルギー上位 10):", cd0.rho[np.argsort(-cd0.sig ** 2)[:10]].round(2).tolist())
cdk = co.Codec(Htr, use_N=False); print("N なしの ρ (上位 10):", cdk.rho[np.argsort(-cdk.sig ** 2)[:10]].round(2).tolist())
perm = np.random.default_rng(1).permutation(len(Hte))
for bpr in (1, 2):
    cd = co.Codec(Htr); cd.allocate(bpr * co.D); rec("陰性対照: 提案 (順番を混ぜた評価)", bpr, cd.used, cd.run(Hte[perm]), Hx=Hte[perm])
json.dump(res, open(sys.argv[1].replace(".npz", "_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
