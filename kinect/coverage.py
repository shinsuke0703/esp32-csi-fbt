# -*- coding: utf-8 -*-
"""Kinect の記録 (KinectLog.exe の CSV) から、どこで骨格が取れるかを調べる。
使い方: python coverage.py kinect.csv
フレーム番号が飛んだ所 = 骨格が取れなかったフレーム。取れた所は腰 (HipCenter) の距離 z と左右の角度 atan(x/z) で分け、
「取れたフレームの割合」と「推定に頼った関節の数」(足が画面から切れると増える) を出す。取れなくなる直前の位置も数える。"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
fr_all = np.arange(df["frame"].min(), df["frame"].max() + 1)
tr = df[df["state"] == 2].drop_duplicates("frame").set_index("frame")
ok = np.isin(fr_all, tr.index)
print(f"全 {len(fr_all)} フレーム ({len(fr_all) / 30:.0f} 秒)、骨格あり {ok.mean() * 100:.0f} %")
z = tr["HipCenter_z"]; x = tr["HipCenter_x"]; ang = np.degrees(np.arctan2(x, z))
s_cols = [c for c in tr.columns if c.endswith("_s")]
inf = (tr[s_cols] != 2).sum(1)
foot = ((tr["FootLeft_s"] == 2) & (tr["FootRight_s"] == 2))
print("\n腰の距離 [m] | フレーム数 | 推定に頼った関節 (平均) | 両足とも見えている割合")
for a, b in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5)]:
    m = (z >= a) & (z < b)
    if m.sum(): print(f"{a:.1f}〜{b:.1f} | {m.sum():5d} | {inf[m].mean():4.1f} | {foot[m].mean() * 100:4.0f} %")
print("\n左右の角度 [度] | フレーム数 | 推定に頼った関節 (平均)")
for a in range(-35, 35, 10):
    m = (ang >= a) & (ang < a + 10)
    if m.sum(): print(f"{a:+3d}〜{a + 10:+3d} | {m.sum():5d} | {inf[m].mean():4.1f}")
# 取れなくなる直前 (骨格あり → なし に変わる所) の腰の位置
lost = [f for f in tr.index if (f + 1) in fr_all and (f + 1) not in tr.index]
if lost:
    L = tr.loc[lost]
    print(f"\n骨格が外れた回数 {len(lost)} 回。外れる直前の腰: 距離 {np.round(L['HipCenter_z'].values, 2).tolist()[:20]} m、"
          f"角度 {np.round(np.degrees(np.arctan2(L['HipCenter_x'], L['HipCenter_z'])).values, 0).tolist()[:20]} 度")
