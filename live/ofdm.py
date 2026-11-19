import sys, time
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); time.sleep(3); s.reset_input_buffer(); cmd(s, "MODE"); cmd(s, "STAT"); cmd(s, "B20")
for ch in (1, 6, 8, 9, 10, 11):
    cmd(s, f"C{ch}"); buf = b""; t0 = time.time()
    while time.time() - t0 < 4: buf += s.read(8192)
    st = [l for l in buf.decode("utf-8", "replace").splitlines() if "STAT" in l]; print(ch, st[0], "→", st[-1])
