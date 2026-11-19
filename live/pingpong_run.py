# -*- coding: utf-8 -*-
"""往復 (PING/PONG) 実験の収録。6 台 (E:/CSI/esp32_pingpong) に PC から順番に「j に PING を送れ」と指示し、全員の記録を集める。
使い方: python pingpong_run.py 秒数 出力.npz [1 組あたりの間隔 ms=8]
"""
import sys, time, struct, threading, re
import numpy as np
import serial, serial.tools.list_ports

REC = struct.Struct("<IBBBBIIIbbBBHBB256s"); MAGIC = struct.pack("<I", 0xB0B0C510)
dur, out = float(sys.argv[1]), sys.argv[2]; gap = (float(sys.argv[3]) if len(sys.argv) > 3 else 8) / 1000
CMD = sys.argv[4] if len(sys.argv) > 4 else "P"   # P = ブロードキャストの PING (相手が PONG を返す)、U = ユニキャストの PING (相手のハードウェアが ACK を返す)

ports = [p.device for p in serial.tools.list_ports.comports() if "10C4:EA60" in p.hwid or "303A:1001" in p.hwid]
def open_port(dev):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.02)
    s.dtr = "COM6" == dev; s.rts = False; s.port = dev; s.open(); return s
S = {d: open_port(d) for d in ports}; time.sleep(2.0)
recs = []; text = {d: b"" for d in ports}; stop = False; lock = threading.Lock()

def reader(dev):
    buf = b""; s = S[dev]
    while not stop:
        buf += s.read(16384)
        while True:
            i = buf.find(MAGIC)
            if i < 0: text[dev] += buf[:-3]; buf = buf[-3:]; break
            text[dev] += buf[:i]; buf = buf[i:]
            if len(buf) < REC.size: break
            f = REC.unpack_from(buf, 0); buf = buf[REC.size:]
            with lock: recs.append(f[1:15] + (np.frombuffer(f[15], np.int8).copy(),))
th = [threading.Thread(target=reader, args=(d,), daemon=True) for d in ports]
for t in th: t.start()

# どのポートが何番のノードか
for d in ports: S[d].write(b"S\n")
time.sleep(0.5)
idof = {}
for d in ports:
    m = re.findall(rb"\[PP\] id (-?\d+)", text[d])
    if m: idof[int(m[-1])] = d
print("ノードとポート:", {k: idof[k] for k in sorted(idof)})
ids = sorted(k for k in idof if k >= 0)
pairs = [(i, j) for i in ids for j in ids if i != j]

t0 = time.time(); rounds = 0
while time.time() - t0 < dur:
    for i, j in pairs:
        S[idof[i]].write(f"{CMD} {j}\n".encode()); time.sleep(gap)
    rounds += 1
time.sleep(0.5); stop = True
for t in th: t.join()
for d in ports:
    S[d].write(b"S\n")
time.sleep(0.3)
for d in ports: S[d].close()
cols = ["rx", "tx", "type", "target", "seq", "t_rx", "t_aux", "rssi", "noise", "sig", "cwb", "len", "fwi", "chip"]
arr = {c: np.array([r[k] for r in recs], dtype=np.int64) for k, c in enumerate(cols)}
arr["csi"] = np.stack([r[14] for r in recs]) if recs else np.zeros((0, 256), np.int8)
np.savez_compressed(out, **arr)
print(f"{rounds} 巡 ({len(pairs)} 組 × {gap*1000:.0f} ms)、記録 {len(recs)} 件 → {out}")
for k in (1, 2, 3, 4, 5):
    print(f"  type {k} ({["", "PING", "PONG", "PONG 送信の記録", "ユニキャスト PING", "ACK"][k]}): {(arr['type'] == k).sum()} 件")
