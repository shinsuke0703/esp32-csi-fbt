import sys, time, re
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); time.sleep(3); s.reset_input_buffer()
cmd(s, "MODE"); cmd(s, "STAT")
def counts():
    buf = b""; t0 = time.time()
    while time.time() - t0 < 1.2: buf += s.read(8192)
    m = re.findall(rb"\[STAT\] promisc (\d+) csi_cb (\d+)", buf); return tuple(map(int, m[-1])) if m else None
for bw in ("B20",):
    s.write((bw + "\n").encode()); time.sleep(0.3)
    for ch in (8, 9, 10):
        s.write(f"C{ch}\n".encode()); time.sleep(0.5); a = counts(); time.sleep(2.5); b = counts()
        print(f"{bw} ch{ch:2d}: promisc {(b[0]-a[0])/2.5:6.1f}/s  csi {(b[1]-a[1])/2.5:6.1f}/s", flush=True)
s.close()
