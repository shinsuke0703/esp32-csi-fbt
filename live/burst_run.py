# -*- coding: utf-8 -*-
"""ACK 往復のバースト収録。送信役 (既定: ノード 0, 3, 4) が順番に、自分以外の全員へユニキャスト PING を gap µs おきに送る。
相手のハードウェアが返す ACK の CSI (送信役が記録) と、PING の CSI (相手が記録) を集める。記録は短い形式 (LLTF のみ)。
使い方: python burst_run.py 秒数 出力.npz [gap_us=1200] [送信役=0,3,4]
"""
import sys, time, struct, threading, re
import numpy as np
import serial, serial.tools.list_ports

HDR = struct.Struct("<IBBBBIIIbbBBHBB"); SIZE = HDR.size + 128; MAGIC = struct.pack("<I", 0xB0B0C511)
dur, out = float(sys.argv[1]), sys.argv[2]
gap = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
senders = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,3,4").split(",")]
WAIT = float(sys.argv[5]) / 1000 if len(sys.argv) > 5 else 0.014   # 送信役ごとの待ち時間の上限 [ms → s]

ports = [p.device for p in serial.tools.list_ports.comports() if "10C4:EA60" in p.hwid or "303A:1001" in p.hwid]
def open_port(dev):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.02); s.dtr = dev == "COM6"; s.rts = False; s.port = dev; s.open(); return s
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
            if len(buf) < SIZE: break
            f = HDR.unpack_from(buf, 0); c = np.frombuffer(buf[HDR.size:SIZE], np.int8).copy(); buf = buf[SIZE:]
            with lock: recs.append(f[1:] + (time.time(), c))
th = [threading.Thread(target=reader, args=(d,), daemon=True) for d in ports]
for t in th: t.start()
for d in ports: S[d].write(b"S\n")
time.sleep(0.5)
idof = {}
for d in ports:
    m = re.findall(rb"\[PP\] id (-?\d+)", text[d])
    if m: idof[int(m[-1])] = d
for d in ports: S[d].write(b"C1\n")
time.sleep(0.2)
print("ノードとポート:", {k: idof[k] for k in sorted(idof)}, " 送信役:", senders, f" gap {gap} µs")
wait = 5 * gap / 1e6 + 0.0015
t0 = time.time(); cycles = 0
while time.time() - t0 < dur:
    for s_ in senders:   # 送信役のバーストが終わった合図 [BD] を待ってから次へ (重なると ACK を取り違える)
        dev = idof[s_]; n0 = text[dev].count(b"[BD]")
        S[dev].write(f"B {gap}\n".encode()); tw = time.time()
        # [BD] は記録のデータと混ざって読めないことがあるので、バーストの長さに合わせた固定の時間 (14 ms) も上限にする
        while text[dev].count(b"[BD]") == n0 and time.time() - tw < WAIT: time.sleep(0.0005)
    cycles += 1
time.sleep(0.3); stop = True
for t in th: t.join()
for d in ports: S[d].write(b"C0\n")
for d in ports: S[d].close()
cols = ["rx", "tx", "type", "target", "seq", "t_rx", "t_aux", "rssi", "noise", "sig", "cwb", "len", "fwi", "chip", "t_host"]
arr = {c: np.array([r[k] for r in recs], dtype=np.float64 if c == "t_host" else np.int64) for k, c in enumerate(cols)}
arr["csi"] = np.stack([r[15] for r in recs]) if recs else np.zeros((0, 128), np.int8)
np.savez_compressed(out, **arr)
el = time.time() - t0
print(f"{cycles} 巡 ({cycles / el:.1f} 巡/秒)、記録 {len(recs)} 件 → {out}")
print(f"  ユニキャスト PING を相手が受けた: {((arr['type'] == 4) & (arr['rx'] == arr['target'])).sum()} 件、ACK: {(arr['type'] == 5).sum()} 件 (要求 {cycles * len(senders) * 5} 回)")
