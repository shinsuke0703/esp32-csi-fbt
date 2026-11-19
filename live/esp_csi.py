"""ESP32 スキャナ (E:/CSI/esp32s3_scanner) の 289 バイトのパケットを読む共通部分"""
import serial, struct, time, numpy as np
MAGIC = b'\x10\xc5\x10\xc5'; PK = 289
HDR = struct.Struct('<IIIIbBBHBBBBBb6s')
def open_port(port="COM6", dtr=True):
    s = serial.Serial(port=None, baudrate=921600, timeout=0.05); s.dtr = dtr; s.rts = False; s.port = port; s.open(); return s
def cmd(s, c): s.write((c + "\n").encode()); time.sleep(0.3)
def parse(buf):
    """buf から完成したパケットを取り出す。戻り値: (パケットのリスト, 残りのバッファ, テキスト行)"""
    out, text = [], b""
    while True:
        i = buf.find(MAGIC)
        if i < 0: text += buf[:-3] if len(buf) > 3 else b""; buf = buf[-3:]; break
        text += buf[:i]; buf = buf[i:]
        if len(buf) < PK: break
        p = bytes(buf[:PK]); buf = buf[PK:]
        f = HDR.unpack_from(p, 0)
        out.append(dict(seq=f[2], ts=f[3], rssi=f[4], ver=f[5], fwi=f[6], len=f[7], cwb=f[9], sig=f[10], sec=f[11],
                        nf=f[13], mac=":".join(f"{b:02X}" for b in f[14]), raw=np.frombuffer(p[33:33 + 256], np.int8).copy()))
    return out, buf, text
