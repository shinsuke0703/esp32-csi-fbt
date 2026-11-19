import sys, time
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(); buf = b""; t0 = time.time()
reset = len(sys.argv) > 1
if reset:   # USB-Serial-JTAG のリセット (RTS)
    s.rts = True; time.sleep(0.1); s.rts = False
while time.time() - t0 < 5: buf += s.read(8192)
s.close(); print(len(buf), "bytes"); print(buf[:1500].decode("utf-8", "replace"))
