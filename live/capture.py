"""ストロボの CSI を収録して npz に保存。使い方: python capture.py 秒数 出力.npz"""
import sys, time
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
dur, out = float(sys.argv[1]), sys.argv[2]
s = open_port(); time.sleep(3); s.reset_input_buffer()
buf = b""; pk = []; t0 = time.time(); last = t0
while time.time() - t0 < dur:
    buf += s.read(16384); p, buf, _ = parse(buf); pk += p
    if time.time() - last > 10: last = time.time(); print(f"{time.time()-t0:5.0f}s  {len(pk)} pkt", flush=True)
s.close()
np.savez_compressed(out, raw=np.stack([p["raw"] for p in pk]), seq=np.array([p["seq"] for p in pk]), ts=np.array([p["ts"] for p in pk]),
                    rssi=np.array([p["rssi"] for p in pk]), len=np.array([p["len"] for p in pk]), sig=np.array([p["sig"] for p in pk]),
                    cwb=np.array([p["cwb"] for p in pk]), mac=np.array([p["mac"] for p in pk]))
print("saved", out, len(pk))
