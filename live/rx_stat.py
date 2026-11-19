import sys, time, re
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
s = open_port(sys.argv[1], dtr=False); time.sleep(0.5); s.reset_input_buffer(); s.write(b"STAT\n")
buf = b""; t0 = time.time()
while time.time() - t0 < 3.5: buf += s.read(8192)
s.write(b"STAT\n"); time.sleep(0.2); s.close()
print("\n".join(m.decode() for m in re.findall(rb"\[STAT\][^\n]*", buf)))
