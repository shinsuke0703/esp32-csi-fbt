# -*- coding: utf-8 -*-
"""CSI (ESP32 6 台、slot_run.py) と Kinect v1 の骨格 (KinectLog.exe) を同時に収録する。
どちらも PC の時計 (UNIX 秒) で時刻を付けるので、あとで時刻でそろえられる (Kinect の遅れは sync_check.py で測る)。
使い方: python rec_both.py 秒数 [名前=日時]
出力: E:/CSI/live/sessions/<名前>/csi.npz、kinect.csv (と最後の深度画像 kinect.pgm)
Kinect の首振りの角度は変えない (今の向きのまま)。Kinect は先に始めて、骨格を捕まえる立ち上がり (約 1 秒) を CSI の前に済ませる。
"""
import sys, os, subprocess, time, datetime

dur = float(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.path.join(r"E:/CSI/live/sessions", name); os.makedirs(out, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
# Kinect: CSI の準備 (ポートを開いて 2.5 秒ほど) と後始末の分だけ長めに記録する
kin = subprocess.Popen([r"E:/CSI/kinect/KinectLog.exe", f"{dur + 8:.0f}", os.path.join(out, "kinect.csv")],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
time.sleep(1.5)
print(f"収録 {dur:.0f} 秒 → {out}", flush=True)
csi = subprocess.run([sys.executable, "-P", r"E:/CSI/live/slot_run.py", f"{dur}", os.path.join(out, "csi.npz")],
                     capture_output=True, text=True, encoding="utf-8", env=env)
print(csi.stdout.strip(), csi.stderr.strip()[-500:], sep="\n")
kout, _ = kin.communicate(timeout=60)
lines = [l for l in kout.splitlines() if "秒:" in l]
tracked = [l for l in lines if "骨格あり" in l]
print("Kinect:", lines[0] if lines else "", "…", lines[-1] if lines else "")
print("Kinect の記録 →", os.path.join(out, "kinect.csv"))
