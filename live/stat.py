import sys, time
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); time.sleep(3); s.reset_input_buffer()
cmd(s, "MODE"); cmd(s, "STAT")   # テキスト表示 + 統計
for ch in (1, 6, 11):
    cmd(s, f"C{ch}"); buf = b""; t0 = time.time()
    while time.time() - t0 < 3: buf += s.read(8192)
    print(f"--- ch{ch}"); print("\n".join(l for l in buf.decode("utf-8", "replace").splitlines() if "STAT" in l or "CMD" in l or l.startswith("CSI"))[:800])
s.close()
