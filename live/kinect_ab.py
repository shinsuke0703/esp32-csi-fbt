# -*- coding: utf-8 -*-
"""Kinect が ESP32 (特に C3) の往復を邪魔しているかの切り分け: Kinect なし / あり を交互に 10 秒ずつ収録し、
C3 → 1〜5 と DevKitC 1 → 2〜5 の PING が届いた回数と RSSI を比べる。収録中は人が動かないこと。
使い方: python kinect_ab.py [繰り返し=2]"""
import sys, os, subprocess, time
import numpy as np

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
os.makedirs("tmp_ab", exist_ok=True); env = dict(os.environ, PYTHONIOENCODING="utf-8")
def summary(fn):
    d = dict(np.load(fn)); typ, rx, tx, tgt = d["type"], d["rx"], d["tx"], d["target"]; T = d["t_host"].max() - d["t_host"].min()
    hz = lambda s, t: round(((typ == 4) & (tx == s) & (rx == t) & (tgt == t)).sum() / T)
    rs = lambda s, t: int(np.median(d["rssi"][(typ == 4) & (tx == s) & (rx == t)])) if ((typ == 4) & (tx == s) & (rx == t)).any() else None
    return f"C3→1〜5 {[hz(0, t) for t in range(1, 6)]} Hz、RSSI {[rs(0, t) for t in range(1, 6)]} | DevKitC 1→2〜5 {[hz(1, t) for t in range(2, 6)]} Hz"
for r in range(reps):
    for kin in (False, True):
        k = None
        if kin:
            k = subprocess.Popen([r"E:/CSI/kinect/KinectLog.exe", "20", os.path.abspath("tmp_ab/k.csv")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
        fn = os.path.abspath(f"tmp_ab/ab_{r}_{int(kin)}.npz")
        subprocess.run([sys.executable, "-P", "slot_run.py", "10", fn], capture_output=True, env=env)
        print(f"{r + 1} 回目 Kinect {'あり' if kin else 'なし'}: {summary(fn)}", flush=True)
        if k: k.wait()
