# -*- coding: utf-8 -*-
"""Kinect の軌跡を使う校正の共通部分: 同時収録 (rec_both.py) を読み、組ごとの往復の時系列と、同じ時刻の Kinect の腰の位置 (部屋の座標) をそろえる。
部屋の座標: Kinect の座標を、足の高さが一番そろう角度で x 軸まわりに回し、床を高さ 0 にしたもの。X 左右、Y 高さ、Z 奥行き [m]。
Kinect の時刻は CSI より約 0.22 秒遅れる (sync_check.py) ので、Kinect の時刻から引いてそろえる。"""
import os
import numpy as np
import pandas as pd

KINECT_LAG = 0.22


def rot(th, x, y, z):
    c, s = np.cos(th), np.sin(th); return x, c * y - s * z, s * y + c * z


def load_session(ses, lag=KINECT_LAG, smooth=10):
    """戻り値: dict(pairs, t[p], P[p] (トーンの和、複素), Pk[p] (トーンごと), rssi[p] (2 方向の和), body[p] (その時刻の腰 [n,3]、部屋の座標),
    joints (Kinect の全関節、部屋の座標 [T,20,3])、tk (Kinect の時刻、遅れを引いた後)、tilt (度)、kinect_pos (部屋の座標での Kinect の位置))"""
    d = dict(np.load(os.path.join(ses, "csi.npz"))); typ, rx, tgt = d["type"], d["rx"], d["target"]
    raw = d["csi"].astype(float); lltf_all = raw[:, 1::2] + 1j * raw[:, 0::2]
    ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
    ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
    pw = np.mean(np.abs(lltf_all[list(ping.values())[:3000]]) ** 2, 0); valid = pw > 0.02 * pw.max()
    by = {}
    for (s, q), a in ping.items():
        b = ack.get((s, q))
        if b is None: continue
        by.setdefault((s, int(tgt[a])), []).append((d["t_host"][b], a, b))
    # Kinect: 足の高さが一番そろう回転で部屋の座標に
    k = pd.read_csv(os.path.join(ses, "kinect.csv")); k = k[k["state"] == 2].drop_duplicates("frame")
    best = None
    for deg in np.arange(-40, 40, 0.5):
        th = np.radians(deg); ys = []
        for f in ("FootLeft", "FootRight"):
            ok = k[f + "_s"] == 2; ys.append(rot(th, k[f + "_x"][ok], k[f + "_y"][ok], k[f + "_z"][ok])[1])
        y = np.concatenate(ys)
        if best is None or np.std(y) < best[1]: best = (deg, np.std(y), np.median(y))
    th = np.radians(best[0]); floor = best[2]
    names = [c[:-2] for c in k.columns if c.endswith("_x")]
    J = np.stack([np.stack(rot(th, k[f"{n}_x"].values, k[f"{n}_y"].values, k[f"{n}_z"].values), -1) for n in names], 1)
    J[..., 1] -= floor
    tk = k["t_unix"].values - lag
    hip = J[:, names.index("HipCenter")]; spine = J[:, names.index("Spine")]
    torso = 0.5 * (hip + spine)                         # 胴の中心 (散乱の中心の目安)
    out = dict(pairs=[], t={}, P={}, Pk={}, rssi={}, body={}, joints=J, tk=tk, names=names, tilt=best[0],
               kinect_pos=np.array([0.0, -floor, 0.0]), valid=valid)
    for p, L in sorted(by.items()):
        L.sort(); t = np.array([x[0] for x in L]); ia = np.array([x[1] for x in L]); ib = np.array([x[2] for x in L])
        Pk = lltf_all[ia][:, valid] * lltf_all[ib][:, valid]
        m = (t >= tk[0]) & (t <= tk[-1])
        if m.sum() < 100: continue
        out["pairs"].append(p); out["t"][p] = t[m]; out["Pk"][p] = Pk[m]; out["P"][p] = Pk[m].sum(1)
        out["rssi"][p] = (d["rssi"][ia] + d["rssi"][ib])[m].astype(float)
        out["body"][p] = np.stack([np.interp(t[m], tk, torso[:, c]) for c in range(3)], 1)
    return out
