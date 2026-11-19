import sys, time, collections
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); time.sleep(1.5); s.reset_input_buffer()
cmd(s, "R0")   # 間隔制限なしで数える
res = {}
for ch in range(1, 14):
    cmd(s, f"C{ch}"); s.reset_input_buffer(); buf = b""; pk = []; t0 = time.time()
    while time.time() - t0 < 3: buf += s.read(8192); p, buf, _ = parse(buf); pk += p
    c = collections.Counter((p["mac"], p["len"], p["cwb"], p["sig"]) for p in pk)
    res[ch] = (len(pk), c.most_common(3))
    print(f"ch{ch:2d}: {len(pk)/3:6.1f} pkt/s  上位: " + "; ".join(f"{m} len{l} cwb{w} sig{g} x{n}" for (m, l, w, g), n in c.most_common(3)), flush=True)
s.close()
