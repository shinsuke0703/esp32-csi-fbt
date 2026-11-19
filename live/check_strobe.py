import sys, time, collections
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); time.sleep(3); s.reset_input_buffer()
for c in ("RADAR", "B20", "C6", "R0", "M CC:DB:A7:32:13:80"): cmd(s, c)
buf = b""; pk = []; t0 = time.time()
while time.time() - t0 < 5: buf += s.read(16384); p, buf, txt = parse(buf); pk += p
print(f"{len(pk)/5:.1f} pkt/s")
if pk:
    c = collections.Counter((p["mac"], p["len"], p["cwb"], p["sig"], p["fwi"]) for p in pk); print(c.most_common(5))
    print("rssi", [p["rssi"] for p in pk[:10]], "nf", pk[0]["nf"]); print("raw[:32]", pk[0]["raw"][:32].tolist())
    print("ts 間隔 ms", np.diff([p["ts"] for p in pk[:30]]).tolist())
s.close()
