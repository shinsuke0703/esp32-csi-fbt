# -*- coding: utf-8 -*-
"""ACK 往復の自律モード (トークン) の収録。ノード 0〜4 が順番に、自分より番号の大きい全員へ 1 回ずつ往復する (15 組 / 巡)。
PC は開始と停止を指示して、記録を集めるだけ。
使い方: python token_run.py 秒数 出力.npz [周期 µs=16667]
"""
import sys, time, struct, threading, re
import numpy as np
import serial, serial.tools.list_ports

HDR = struct.Struct("<IBBBBIIIbbBBHBB"); SIZE = HDR.size + 128; MAGIC = struct.pack("<I", 0xB0B0C511)
dur, out = float(sys.argv[1]), sys.argv[2]; period = int(sys.argv[3]) if len(sys.argv) > 3 else 16667
HOPPING = len(sys.argv) > 4 and sys.argv[4] == "hop"   # 1 巡ごとに ch 1/6/11 を切り替える (実験 E2)

ports = [p.device for p in serial.tools.list_ports.comports() if "10C4:EA60" in p.hwid or "303A:1001" in p.hwid]
def open_port(dev):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.02); s.dtr = "303A" in next(p.hwid for p in serial.tools.list_ports.comports() if p.device == dev); s.rts = False; s.port = dev; s.open(); return s
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
def status():
    for d in ports: S[d].write(b"S\n")
    time.sleep(0.4); st = {}
    for d in ports:
        m = re.findall(rb"\[PP\] id (-?\d+)[^\n]*", text[d]); line = re.findall(rb"\[PP\][^\n]*", text[d])
        if m: st[int(m[-1])] = (d, line[-1].decode(errors="replace"))
    return st
st0 = status(); idof = {k: v[0] for k, v in st0.items()}
print("ノードとポート:", {k: idof[k] for k in sorted(idof)}, f" 周期 {period} µs ({1e6 / period:.0f} Hz)")
for d in ports: S[d].write(b"C1\n")
REPS = next((int(a[2:]) for a in sys.argv[3:] if a.startswith("N=")), 1)   # 1 回の手番で同じ相手に送る回数 (速い方式のファームのみ)
for d in ports: S[d].write(f"N {REPS}\n".encode())
PRATE = next((int(a[3:], 0) for a in sys.argv[3:] if a.startswith("PR=")), None)   # PHY レート (例 PR=0x13 は MCS3)
if PRATE is not None:
    for d in ports: S[d].write(f"PR {PRATE}\n".encode())
    time.sleep(0.3)
    for d in ports:
        m = re.findall(rb"\[PR\][^\n]*", text[d])
        if m: print("  ", m[-1].decode(errors="replace"))
time.sleep(0.2)
if HOPPING:   # 全員を ch 1 にそろえてから始める
    for d in ports: S[d].write(b"H 1\n")
    time.sleep(0.3)
FREE = "FR" in sys.argv[3:]   # 同時送信方式 (トークンなし、送信役がそれぞれ周期 period で送る)
for d in ports: S[d].write((f"FR {period}\n" if FREE else f"R {period}\n").encode())
t0 = time.time(); time.sleep(dur); el = time.time() - t0
for d in ports: S[d].write(b"FR 0\n" if FREE else b"R 0\n")
time.sleep(0.4)
if HOPPING:
    for d in ports: S[d].write(b"SH\n")
    time.sleep(0.4)
    for d in ports:
        m = re.findall(rb"\[HOP\][^\n]*", text[d])
        if m: print("  ", m[-1].decode(errors="replace"))
    for d in ports: S[d].write(b"H 0\n")
    time.sleep(0.2)
for d in ports: S[d].write(b"C0\n")
time.sleep(0.2)
for d in ports: S[d].write(b"ST\n")   # 手番の時間の内訳 (計測用のファームのみ)
time.sleep(0.4)
for d in ports:
    m = re.findall(rb"\[ST\][^\n]*", text[d])
    if m: print("  ", m[-1].decode(errors="replace"))
st1 = status(); stop = True
for t in th: t.join()
for d in ports: S[d].close()
cols = ["rx", "tx", "type", "target", "seq", "t_rx", "t_aux", "rssi", "noise", "sig", "cwb", "len", "fwi", "chip", "t_host"]
arr = {c: np.array([r[k] for r in recs], dtype=np.float64 if c == "t_host" else np.int64) for k, c in enumerate(cols)}
arr["csi"] = np.stack([r[15] for r in recs]) if recs else np.zeros((0, 128), np.int8)
np.savez_compressed(out, **arr)
def num(st, key):
    try: return int(re.findall(key + r" (\d+)", st[0][1])[0])
    except Exception: return None
c0, c1 = num(st0, "cycles"), num(st1, "cycles")
cyc = (c1 - c0) if (c0 is not None and c1 is not None) else int((arr["type"] == 4).sum() / 15)
print(f"{el:.1f} 秒、巡回 {cyc} 回 ({cyc / el:.1f} 巡/秒)、トークンを失った回数 (累計) {num(st1, 'tok_lost')}、記録 {len(recs)} 件 → {out}")
for k in sorted(st1): print(f"  ノード {k}: {st1[k][1]}")
print(f"  ユニキャスト PING を相手が受けた: {((arr['type'] == 4) & (arr['rx'] == arr['target'])).sum()} 件、ACK: {(arr['type'] == 5).sum()} 件 (理想 {15 * cyc} 回)")
