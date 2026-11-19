# -*- coding: utf-8 -*-
"""C3 の置き場所を決めるための表示: 時間枠の方式で収録しながら、2 秒ごとに C3 → 1〜5 の RSSI と届いた回数を出す。
C3 を動かしながら、5 本とも RSSI が −75 dBm より強く、回数が 60 Hz 近くになる場所を探す。
使い方: python c3_place.py [秒=60]"""
import sys, os, subprocess, time
import numpy as np

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60
os.makedirs("tmp_ab", exist_ok=True); env = dict(os.environ, PYTHONIOENCODING="utf-8")
t0 = time.time(); i = 0
while time.time() - t0 < dur:
    fn = os.path.abspath(f"tmp_ab/place_{i % 2}.npz"); i += 1
    subprocess.run([sys.executable, "-P", "slot_run.py", "2", fn], capture_output=True, env=env)
    d = dict(np.load(fn)); typ, rx, tx, tgt = d["type"], d["rx"], d["tx"], d["target"]; T = max(d["t_host"].max() - d["t_host"].min(), 1e-3)
    cells = []
    for t in range(1, 6):
        m = (typ == 4) & (tx == 0) & (rx == t) & (tgt == t)
        r = int(np.median(d["rssi"][m])) if m.any() else None
        mark = "" if (r is not None and r >= -75) else " ←弱い"
        cells.append(f"→{t}: {r if r is not None else '--':>4} dBm {m.sum() / T:4.0f} Hz{mark}")
    print(f"{time.time() - t0:5.0f} 秒 | " + " | ".join(cells), flush=True)
