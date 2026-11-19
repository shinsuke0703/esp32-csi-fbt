# -*- coding: utf-8 -*-
"""体の成分 η のドップラーのスペクトル (1 秒ごと、60 Hz なので ±30 Hz) を、胴の速さで「止まっている (< 0.15 m/s)」「歩いている (> 0.5 m/s)」に分けて平均し、
どの帯域が歩くと強まるかを見る (全リンクの平均)。使い方: python doppler_spec.py sessions/<名前>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from calib_common import load_session
S = load_session(sys.argv[1]); N = 60
spec = {"止": [], "歩": []}
for p in S["pairs"]:
    Pk = S["Pk"][p]; t = S["t"][p]
    bg = np.median(Pk.real, 0) + 1j * np.median(Pk.imag, 0); w = np.abs(bg) ** 2
    eta = np.sqrt((Pk * np.conj(bg)).sum(1) / w.sum()) - 1
    x = S["body"][p]
    for s in range(0, len(t) - N, N // 2):
        if t[s + N - 1] - t[s] > 1.1: continue                       # 抜けのある区間は使わない
        seg = eta[s:s + N] - eta[s:s + N].mean()
        F = np.abs(np.fft.fftshift(np.fft.fft(seg * np.hanning(N)))) ** 2
        v = np.linalg.norm(x[s + N - 1] - x[s]) / (t[s + N - 1] - t[s])
        if v < 0.15: spec["止"].append(F)
        elif v > 0.5: spec["歩"].append(F)
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / 60))
A, B = np.mean(spec["止"], 0), np.mean(spec["歩"], 0)
print(f"区間の数: 止まっている {len(spec['止'])}、歩いている {len(spec['歩'])}")
print("周波数 [Hz] | 止まっているとき | 歩いているとき | 比 (歩 / 止)")
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30)]:
    m = (np.abs(f) >= lo) & (np.abs(f) < hi)
    print(f"{lo:2d}〜{hi:2d} | {A[m].mean():.2e} | {B[m].mean():.2e} | {B[m].mean() / A[m].mean():5.2f}")
