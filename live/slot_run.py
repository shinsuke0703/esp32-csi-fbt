# -*- coding: utf-8 -*-
"""時間枠の方式の収録 (研究環境用、制御はすべて USB)。DevKitC 5 台 (ノード 1〜5) の 10 組が主力、C3 (ノード 0) は補助。
PC は周期ごとに全員へ "G" を一斉に送るだけ。各ノードはあらかじめ "SL <遅れ> <相手>" で決めた自分の時間枠で 1 手番を行う。
電波には測定用の PING と ACK しか流れない。
使い方: python slot_run.py 秒数 出力.npz [周期 µs=16667] [C3EVERY=1 (C3 が送る巡の間隔、0 で送らない)]
時間枠 (往復 1 回 ≒ 0.8 ms として): ノード 1 → 2,3,4,5 は 0 ms、2 → 3,4,5 は 3.5 ms、3 → 4,5 は 6.3 ms、4 → 5 は 8.3 ms、C3 → 1〜5 は 9.6 ms
"""
import sys, time, struct, threading, re, ctypes
import numpy as np
import serial, serial.tools.list_ports
ctypes.windll.winmm.timeBeginPeriod(1)

HDR = struct.Struct("<IBBBBIIIbbBBHBB"); SIZE = HDR.size + 128; MAGIC = struct.pack("<I", 0xB0B0C511)
dur, out = float(sys.argv[1]), sys.argv[2]
period = next((int(a) for a in sys.argv[3:] if a.isdigit()), 16667)
C3EVERY = next((int(a[8:]) for a in sys.argv[3:] if a.startswith("C3EVERY=")), 1)   # 既定は毎巡 (C3 の組は位置の手がかりとして欠かせない。重みは解く側で下げる)
SLOTS = {1: (0, [2, 3, 4, 5]), 2: (3500, [3, 4, 5]), 3: (6300, [4, 5]), 4: (8300, [5]), 0: (9600, [1, 2, 3, 4, 5])}

ports = [p.device for p in serial.tools.list_ports.comports() if "10C4:EA60" in p.hwid or "303A:1001" in p.hwid]
def open_port(dev):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.002); s.dtr = "303A" in next(p.hwid for p in serial.tools.list_ports.comports() if p.device == dev); s.rts = False; s.port = dev; s.open(); return s
S = {d: open_port(d) for d in ports}; time.sleep(2.0)
recs = []; text = {d: b"" for d in ports}; stop = False; lock = threading.Lock()
def reader(dev):
    buf = b""; s = S[dev]
    while not stop:
        try: buf += s.read(max(1, s.in_waiting))
        except serial.SerialException: continue
        while True:
            i = buf.find(MAGIC)
            if i < 0: text[dev] += buf[:-3]; buf = buf[-3:]; break
            text[dev] += buf[:i]; buf = buf[i:]
            if len(buf) < SIZE: break
            f = HDR.unpack_from(buf, 0); c = np.frombuffer(buf[HDR.size:SIZE], np.int8).copy(); buf = buf[SIZE:]
            with lock: recs.append(f[1:] + (time.time(), c))
th = [threading.Thread(target=reader, args=(d,), daemon=True) for d in ports]
for t in th: t.start()
port_of = {}
for attempt in range(5):   # 全員が答えるまで状態の問い合わせを繰り返す (1 回だと、まれに 1 台が答えず、その台が送らないまま収録してしまう)
    for d in ports:
        if d not in port_of.values(): S[d].write(b"S\n")
    time.sleep(0.5)
    for d in ports:
        m = re.findall(rb"\[PP\] id (-?\d+)", text[d])
        if m: port_of[int(m[-1])] = d
    if len(port_of) >= len(ports): break
if len(port_of) < len(ports): print("注意: 状態の問い合わせに答えないポートがある:", [d for d in ports if d not in port_of.values()])
print("ノードとポート:", dict(sorted(port_of.items())), f" 周期 {period} µs ({1e6 / period:.1f} Hz)、C3 は {C3EVERY} 巡に 1 回" if C3EVERY else "、C3 は送らない")
for d in ports: S[d].write(b"C1\nPL 0\nPR 16\n")
for rep in range(3):   # 設定のコマンドを取りこぼさないよう 3 回送る (1 回だと、まれにノードが受け損ねて 1 台まるごと送らないことがあった)
    for i, (off, tg) in SLOTS.items():
        if i in port_of: S[port_of[i]].write(f"SL {off} {sum(1 << j for j in tg)}\n".encode())
    time.sleep(0.1)
time.sleep(0.3)
main = [port_of[i] for i in (1, 2, 3, 4) if i in port_of]; c3 = port_of.get(0)
t0 = time.perf_counter(); k = 0; late = 0
while time.perf_counter() - t0 < dur:
    tk = t0 + k * period * 1e-6
    while True:
        r = tk - time.perf_counter()
        if r <= 0: break
        if r > 0.002: time.sleep(0.001)
    if time.perf_counter() - tk > 0.002: late += 1
    for d in main: S[d].write(b"G\n")
    if c3 and C3EVERY and k % C3EVERY == 0: S[c3].write(b"G\n")
    k += 1
el = time.perf_counter() - t0
time.sleep(0.5); stop = True
for t in th: t.join()
for d in ports: S[d].write(b"SL 0 0\nC0\n"); S[d].close()
cols = ["rx", "tx", "type", "target", "seq", "t_rx", "t_aux", "rssi", "noise", "sig", "cwb", "len", "fwi", "chip", "t_host"]
arr = {c: np.array([r[k_] for r in recs], dtype=np.float64 if c == "t_host" else np.int64) for k_, c in enumerate(cols)}
arr["csi"] = np.stack([r[15] for r in recs]) if recs else np.zeros((0, 128), np.int8)
np.savez_compressed(out, **arr)
print(f"{el:.1f} 秒、巡回 {k} 回 ({k / el:.1f} 巡/秒)、遅れた開始 {late} 回、記録 {len(recs)} 件 → {out}")
print(f"  ユニキャスト PING を相手が受けた: {((arr['type'] == 4) & (arr['rx'] == arr['target'])).sum()} 件、ACK: {(arr['type'] == 5).sum()} 件")
