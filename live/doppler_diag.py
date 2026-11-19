# -*- coding: utf-8 -*-
"""ドップラーの情報がデータにあるかの確認 (配置を知らなくてよい): 体の成分 η の位相が回る速さ |ω| と、Kinect の胴の速さの相関 (リンクごと)。
使い方: python doppler_diag.py sessions/<名前>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from calib_common import load_session
S = load_session(sys.argv[1])
for (hp, MA, LAG) in ((True, 5, 2), (True, 5, 3), (True, 10, 3)):
    print(f"== 1 秒ごとの平均を引き、{MA} パケット平均、{LAG} パケット間隔")
    res = []
    for p in S["pairs"]:
        Pk = S["Pk"][p]; t = S["t"][p]
        bg = np.median(Pk.real, 0) + 1j * np.median(Pk.imag, 0); w = np.abs(bg) ** 2
        eta = np.sqrt((Pk * np.conj(bg)).sum(1) / w.sum()) - 1
        if hp:
            blk = ((t - t[0]) // 1.0).astype(int); m = np.bincount(blk, eta.real) / np.bincount(blk) + 1j * np.bincount(blk, eta.imag) / np.bincount(blk)
            eta = eta - m[blk]
        eta = np.convolve(eta, np.ones(MA) / MA, 'same')
        z = eta[LAG:] * np.conj(eta[:-LAG]); tt = t[LAG:]
        win = ((tt - tt[0]) // 0.25).astype(int)                       # 0.25 秒ごとに z を足して回転の速さ
        zs = np.bincount(win, z.real) + 1j * np.bincount(win, z.imag); omega = np.abs(np.angle(zs))   # 1 パケットあたりの回転 [rad]
        tw = np.bincount(win, tt) / np.bincount(win)
        x = S["body"][p][LAG:]
        xw = np.stack([np.bincount(win, x[:, c]) / np.bincount(win) for c in range(3)], 1)   # 区間ごとの平均の位置 (Kinect のぶれをならす)
        vw = np.r_[np.nan, np.linalg.norm(np.diff(xw, axis=0), axis=1) / 0.25]              # 区間の間の移動 / 0.25 秒
        ok = np.isfinite(omega) & np.isfinite(vw)
        c = np.corrcoef(omega[ok], vw[ok])[0, 1]; res.append(c)
        print(f"  {p[0]}-{p[1]}: 位相の回る速さと胴の速さの相関 {c:+.2f}")
    print(f"  中央値 {np.median(res):+.2f}")
