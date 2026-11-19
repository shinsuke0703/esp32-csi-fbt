"""複数の受信ノード (E:/CSI/esp32_rx) から同時に収録。ストロボの通し番号 (device_id, version 8) でそろえる。
使い方: python capture_multi.py 秒数 出力.npz COM3 COM7 ..."""
import sys, time, threading, numpy as np
sys.path.insert(0, r"e:/CSI/live")
from esp_csi import *
dur, out, ports = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
data = {p: [] for p in ports}; stop = False
def reader(p):
    s = open_port(p, dtr=False); s.reset_input_buffer(); buf = b""
    while not stop:
        buf += s.read(8192); pk, buf, _ = parse(buf)
        for x in pk: x["sseq"] = int.from_bytes(bytes(HDR.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b"\0" * 6)[4:8]), "little")  # placeholder
        data[p] += pk
    s.close()
# device_id (ストロボ通し番号) を parse が返さないので、生パケットから取り直す版の parse を使う
import esp_csi
_orig = esp_csi.parse
def parse2(buf):
    out, text = [], b""
    while True:
        i = buf.find(MAGIC)
        if i < 0: buf = buf[-3:]; break
        buf = buf[i:]
        if len(buf) < PK: break
        p = bytes(buf[:PK]); buf = buf[PK:]; f = HDR.unpack_from(p, 0)
        out.append(dict(sseq=f[1], seq=f[2], ts=f[3], rssi=f[4], ver=f[5], len=f[7], cwb=f[9], sig=f[10], raw=np.frombuffer(p[33:289], np.int8).copy()))
    return out, buf, text
def reader(p):
    s = open_port(p, dtr=False); s.reset_input_buffer(); buf = b""
    while not stop:
        buf += s.read(8192); pk, buf, _ = parse2(buf); data[p] += pk
    s.close()
th = [threading.Thread(target=reader, args=(p,), daemon=True) for p in ports]
for t in th: t.start()
t0 = time.time(); last = t0
while time.time() - t0 < dur:
    time.sleep(0.5)
    if time.time() - last > 10: last = time.time(); print(f"{time.time()-t0:5.0f}s  " + "  ".join(f"{p}:{len(data[p])}" for p in ports), flush=True)
stop = True
for t in th: t.join()
save = {}
for p in ports:
    pk = data[p]
    if not pk: print(p, "パケットなし"); continue
    for k in ("sseq", "seq", "ts", "rssi", "ver", "len", "cwb", "sig"): save[f"{p}_{k}"] = np.array([x[k] for x in pk], dtype=np.int64)
    save[f"{p}_raw"] = np.stack([x["raw"] for x in pk])
    ss = save[f"{p}_sseq"]; ok = ss != 0xFFFFFFFF
    print(f"{p}: {len(pk)} pkt ({len(pk)/dur:.1f}/s)  ver {set(save[f'{p}_ver'].tolist())}  通し番号なし {(~ok).sum()}  通し番号の飛び {(np.diff(ss[ok]) != 1).sum()}  len {set(save[f'{p}_len'].tolist())}  RSSI {save[f'{p}_rssi'].mean():.1f}")
np.savez_compressed(out, ports=np.array(ports), **save)
sets = [set(save[f"{p}_sseq"].tolist()) - {0xFFFFFFFF} for p in ports if f"{p}_sseq" in save]
if sets: print("全ポート共通の通し番号:", len(set.intersection(*sets)), " / 最大", max(len(s) for s in sets))
