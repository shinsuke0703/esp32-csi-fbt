# -*- coding: utf-8 -*-
"""USB で送信を指示する収録 (研究環境用: 6 台すべて PC につないでいる前提)。電波は測定用の PING と ACK だけで、トークンは使わない。
PC が順番に「T <相手のビットマスク> <回数>」を送り、そのノードが 1 手番 (相手へユニキャスト PING → ACK の CSI) を終えて
"[TD]" を返したら次のノードへ進む。
使い方: python usb_sched.py 秒数 出力.npz [予定=fwd] [回数=1]
  予定: fwd  ノード i → 番号の大きい全員 (トークン方式と同じ 15 組)
        rev  ノード i → 番号の小さい全員 (送る側と返す側を入れ替える。C3 = ノード 0 は返すだけになる)
        both fwd と rev を交互に
"""
import sys, time, struct, threading, re
import numpy as np
import serial, serial.tools.list_ports
import ctypes
ctypes.windll.winmm.timeBeginPeriod(1)   # Windows の時間の刻みを 1 ms に (既定 15.6 ms だと短い待ちが伸びる)

HDR = struct.Struct("<IBBBBIIIbbBBHBB"); SIZE = HDR.size + 128; MAGIC = struct.pack("<I", 0xB0B0C511)
dur, out = float(sys.argv[1]), sys.argv[2]
plan = sys.argv[3] if len(sys.argv) > 3 else "fwd"; reps = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1

ports = [p.device for p in serial.tools.list_ports.comports() if "10C4:EA60" in p.hwid or "303A:1001" in p.hwid]
def open_port(dev):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.002); s.dtr = "303A" in next(p.hwid for p in serial.tools.list_ports.comports() if p.device == dev); s.rts = False; s.port = dev; s.open(); return s
S = {d: open_port(d) for d in ports}; time.sleep(2.0)
recs = []; text = {d: b"" for d in ports}; ntd = {d: 0 for d in ports}; stop = False; lock = threading.Lock()
ev_td = {d: threading.Event() for d in ports}   # "[TD]" (手番の終わり) が届いた合図
sys.setswitchinterval(0.0002)

def reader(dev):
    buf = b""; s = S[dev]
    while not stop:
        try: buf += s.read(max(1, s.in_waiting))
        except serial.SerialException: continue   # Windows の読み取りのタイムアウト (C3 の USB で起きる) は無視
        while True:
            i = buf.find(MAGIC)
            if i < 0: text[dev] += buf[:-3]; buf = buf[-3:]; break
            text[dev] += buf[:i]; buf = buf[i:]
            if len(buf) < SIZE: break
            f = HDR.unpack_from(buf, 0); c = np.frombuffer(buf[HDR.size:SIZE], np.int8).copy(); buf = buf[SIZE:]
            with lock: recs.append(f[1:] + (time.time(), c))
        n = text[dev].count(b"[TD]")
        if n != ntd[dev]: ntd[dev] = n; ev_td[dev].set()
th = [threading.Thread(target=reader, args=(d,), daemon=True) for d in ports]
for t in th: t.start()

# ノード番号とポートの対応 (状態表示 S の id から)
for d in ports: S[d].write(b"S\n")
time.sleep(0.5)
port_of = {}
for d in ports:
    m = re.findall(rb"\[PP\] id (-?\d+)", text[d])
    if m: port_of[int(m[-1])] = d
print("ノードとポート:", dict(sorted(port_of.items())), f" 予定 {plan}、回数 {reps}")
for d in ports: S[d].write(b"C1\n")
PLEN = next((int(a_[3:]) for a_ in sys.argv[3:] if a_.startswith("PL=")), 0)   # PING の中身の長さ (バイト、0 なら最小)
for d in ports: S[d].write(f"PL {PLEN}\n".encode())
PRATE = next((int(a_[3:], 0) for a_ in sys.argv[3:] if a_.startswith("PR=")), 0x10)   # PHY レート (既定 0x10 = MCS0、0x0B = 6M)
for d in ports: S[d].write(f"PR {PRATE}\n".encode())
time.sleep(0.3)

def steps(kind):
    if kind == "fwd": return [(i, sum(1 << j for j in range(i + 1, 6))) for i in range(5)]
    if kind == "rev": return [(i, sum(1 << j for j in range(0, i))) for i in range(1, 6)]
    # main: DevKitC 5 台 (ノード 1〜5) の 10 組が主力。C3 (ノード 0) は補助で、C3EVERY 巡に 1 回だけ 5 台へ送る (0 なら送らない)
    if kind == "main": return [(i, sum(1 << j for j in range(i + 1, 6))) for i in range(1, 5)]
    raise ValueError(kind)
sched = steps("fwd") + steps("rev") if plan == "both" else steps(plan)
C3EVERY = next((int(a_[8:]) for a_ in sys.argv[3:] if a_.startswith("C3EVERY=")), 0)
C3STEP = (0, sum(1 << j for j in range(1, 6)))

t0 = time.time(); ncyc = 0; ntimeout = 0; turn_t = []
while time.time() - t0 < dur:
    cur = sched + ([C3STEP] if plan == "main" and C3EVERY > 0 and ncyc % C3EVERY == 0 else [])
    for (i, mask) in cur:
        d = port_of[i]; ev_td[d].clear(); n0 = ntd[d]; ts = time.perf_counter()
        S[d].write(f"T {mask} {reps}\n".encode())
        ev_td[d].wait(0.05)
        if ntd[d] == n0: ntimeout += 1
        turn_t.append(time.perf_counter() - ts)
    ncyc += 1
el = time.time() - t0
time.sleep(0.4); stop = True
for t in th: t.join()
for d in ports: S[d].write(b"C0\n"); S[d].close()
cols = ["rx", "tx", "type", "target", "seq", "t_rx", "t_aux", "rssi", "noise", "sig", "cwb", "len", "fwi", "chip", "t_host"]
arr = {c: np.array([r[k] for r in recs], dtype=np.float64 if c == "t_host" else np.int64) for k, c in enumerate(cols)}
arr["csi"] = np.stack([r[15] for r in recs]) if recs else np.zeros((0, 128), np.int8)
np.savez_compressed(out, **arr)
print(f"{el:.1f} 秒、巡回 {ncyc} 回 ({ncyc / el:.1f} 巡/秒)、手番の時間 中央値 {np.median(turn_t) * 1000:.2f} ms、応答なし {ntimeout} 回、記録 {len(recs)} 件 → {out}")
print(f"  ユニキャスト PING を相手が受けた: {((arr['type'] == 4) & (arr['rx'] == arr['target'])).sum()} 件、ACK: {(arr['type'] == 5).sum()} 件")
