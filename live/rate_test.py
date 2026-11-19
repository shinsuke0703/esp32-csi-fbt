import sys, time, re, serial
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
st = serial.Serial(port=None, baudrate=115200, timeout=0.1); st.dtr = True; st.rts = False; st.port = "COM6"; st.open(); time.sleep(2)
rx = open_port("COM10", dtr=False); time.sleep(0.5); rx.write(b"STAT\n")
def stat():
    buf = b""; t0 = time.time()
    while time.time() - t0 < 1.3: buf += rx.read(8192)
    m = re.findall(rb"strobe_promisc (\d+) espnow_rx (\d+) csi_match (\d+)[^|]*\| sig (\d+) rate (\d+)", buf)
    return tuple(map(int, m[-1])) if m else None
for r in sys.argv[1:]:
    st.write((r + "\n").encode()); time.sleep(0.5); a = stat(); time.sleep(3); b = stat()
    print(r, st.read(4096).decode(errors="replace").strip().splitlines()[-1:], f"strobe_promisc {(b[0]-a[0])/3:.1f}/s espnow {(b[1]-a[1])/3:.1f}/s csi_match {(b[2]-a[2])/3:.1f}/s sig {b[3]} rate {b[4]}")
rx.write(b"STAT\n"); rx.close(); st.close()
