# -*- coding: utf-8 -*-
"""同時送信方式の上限の切り分け: 指定したノードだけに送らせて (FR)、ACK の記録の数から 1 組あたりのレートを測る。
使い方: python free_one.py <秒> <周期 µs> <送らせるノード, 例 "0" や "0,2">
ノードとポートの対応は 2026-11-19 時点 (0 COM6、1 COM11、2 COM3、3 COM8、4 COM9、5 COM12)。"""
import sys, time, struct, threading
import serial, serial.tools.list_ports

PORT = {0: "COM6", 1: "COM11", 2: "COM3", 3: "COM8", 4: "COM9", 5: "COM12"}
dur, period = float(sys.argv[1]), int(sys.argv[2]); senders = [int(x) for x in sys.argv[3].split(",")]
HDR = struct.Struct("<IBBBBIIIbbBBHBB"); SIZE = HDR.size + 128; MAGIC = struct.pack("<I", 0xB0B0C511)
S = {}
for i, dev in PORT.items():
    s = serial.Serial(port=None, baudrate=921600, timeout=0.02)
    s.dtr = "303A" in next(p.hwid for p in serial.tools.list_ports.comports() if p.device == dev); s.rts = False; s.port = dev; s.open(); S[i] = s
cnt = {i: {} for i in PORT}; stop = False
def reader(i):
    buf = b""
    while not stop:
        buf += S[i].read(8192)
        while True:
            k = buf.find(MAGIC)
            if k < 0: buf = buf[-3:]; break
            buf = buf[k:]
            if len(buf) < SIZE: break
            f = HDR.unpack_from(buf, 0); buf = buf[SIZE:]
            key = (f[3], f[1], f[2]); cnt[i][key] = cnt[i].get(key, 0) + 1   # (type, rx, tx)
th = [threading.Thread(target=reader, args=(i,), daemon=True) for i in PORT]
for t in th: t.start()
for s in S.values(): s.write(b"C1\nN 1\n")
time.sleep(0.3)
for i in senders: S[i].write(f"FR {period}\n".encode())
t0 = time.time(); time.sleep(dur); el = time.time() - t0
for s in S.values(): s.write(b"FR 0\n")
time.sleep(0.5); stop = True
for t in th: t.join()
for s in S.values(): s.write(b"C0\n"); s.close()
for i in senders:
    acks = sum(v for (ty, rx, tx), v in cnt[i].items() if ty == 5)
    print(f"ノード {i}: ACK の記録 {acks} 件 → 相手 {5 - i} 台、1 組あたり {acks / el / (5 - i):.1f} Hz")
pings = sum(v for i in PORT for (ty, rx, tx), v in cnt[i].items() if ty == 4 and tx in senders)
print(f"相手が PING を受けた記録: {pings} 件 ({pings / el:.0f} 件/秒)")
