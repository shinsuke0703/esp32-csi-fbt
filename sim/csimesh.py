# -*- coding: utf-8 -*-
"""CSI メッシュ: 受信機ごとにばらばらに届く CSI を、PC 側で「1 フレーム = 1 つのメッシュ」にまとめる。

メッシュ = ノード (ESP32 の位置) と辺 (送信 → 受信のリンク) のグラフ。各フレームで、辺ごとに
  h      外乱 (共通位相 φ0・位相の傾き τ・利得 a) を外した CSI の「形」 [K] (CSI-OPQ の N と同じ考え方)
  side   外した外乱 (φ0, τ, log a)
  valid  その辺がこのフレームで受信できたか
を持つ。往復 (a→b と b→a) は経路が同じなので、無向の辺にまとめられる (外乱はパケットごとに別なので、形の空間で平均)。

作り方:
  from_sim(...)       シミュレータの出力 + ESP32 の欠点 (simio.Esp32Impairment)
  from_capture(...)   実機の収録 (E:/CSI/live/capture_multi.py の npz、ストロボ 1 台 → 受信 N 台)
"""
from dataclasses import dataclass, field
import numpy as np


def canon(H, tones):
    """外乱を外した形。H [..., K] → h [..., K], side [..., 3] (φ0, τ, log a)。
    τ は隣り合うサブキャリア (番号の差が 1) の位相差から求める (HT40 の DC をまたぐ所は除く)。"""
    k = np.asarray(tones, float)
    dk = np.diff(k); step = dk[dk > 0].min()        # サブキャリアを間引いた場合も、一番細かい間隔の組で傾きを求める
    adj = (dk == step).astype(float)
    tau = np.angle((np.conj(H[..., :-1]) * H[..., 1:] * adj).sum(-1)) / step
    H1 = H * np.exp(-1j * tau[..., None] * k)
    phi = np.angle(H1.sum(-1))
    a = np.sqrt((np.abs(H) ** 2).mean(-1)) + 1e-30
    return H1 * np.exp(-1j * phi)[..., None] / a[..., None], np.stack([phi, tau, np.log(a)], -1)


@dataclass
class CsiMesh:
    nodes: np.ndarray            # [M, 3] ノードの位置 (分からなければ NaN)
    edges: list                  # [(送信, 受信)] 有向の辺。undirected() の後は (小さい番号, 大きい番号)
    tones: np.ndarray            # [K] サブキャリア番号
    t: np.ndarray                # [T] フレームの時刻 [s]
    h: np.ndarray                # [T, E, K] complex 形
    side: np.ndarray             # [T, E, 3] (φ0, τ, log a)
    valid: np.ndarray            # [T, E] bool
    raw: np.ndarray = None       # [T, E, K] complex 外乱を外す前 (必要なら)
    info: dict = field(default_factory=dict)

    @property
    def T(self): return self.h.shape[0]
    @property
    def E(self): return len(self.edges)

    def undirected(self):
        """往復をまとめる。両方受信できたフレームは、形の空間で位相をそろえて平均する (形は共通位相を外してあるので、そのまま足せる)。"""
        pairs = sorted({tuple(sorted(e)) for e in self.edges})
        idx = {e: i for i, e in enumerate(self.edges)}
        T, K = self.T, len(self.tones)
        h = np.zeros((T, len(pairs), K), complex); side = np.zeros((T, len(pairs), 3)); valid = np.zeros((T, len(pairs)), bool)
        for j, (a, b) in enumerate(pairs):
            acc = np.zeros((T, K), complex); n = np.zeros(T)
            for e in ((a, b), (b, a)):
                if e not in idx: continue
                i = idx[e]; v = self.valid[:, i]
                acc[v] += self.h[v, i]; n += v
                side[v, j] = self.side[v, i]
            ok = n > 0; h[ok, j] = acc[ok] / n[ok, None]; valid[:, j] = ok
        return CsiMesh(self.nodes, pairs, self.tones, self.t, h, side, valid, None, dict(self.info, undirected=True))

    def at_rate(self, fps, hold=2):
        """出力のフレームレートに合わせる (例: 60 Hz)。各出力時刻に、直前に受信できた値を使う。
        hold フレーム (入力側) より古い値は使わない (valid = False)。"""
        tout = np.arange(self.t[0], self.t[-1], 1.0 / fps)
        dt_in = np.median(np.diff(self.t))
        T2, E, K = len(tout), self.E, len(self.tones)
        h = np.zeros((T2, E, K), complex); side = np.zeros((T2, E, 3)); valid = np.zeros((T2, E), bool)
        for e in range(E):
            ti = self.t[self.valid[:, e]]; src = np.where(self.valid[:, e])[0]
            if len(ti) == 0: continue
            j = np.searchsorted(ti, tout, side="right") - 1
            ok = (j >= 0) & (tout - ti[np.clip(j, 0, None)] <= hold * dt_in + 1e-9)
            h[ok, e] = self.h[src[j[ok]], e]; side[ok, e] = self.side[src[j[ok]], e]; valid[ok, e] = True
        return CsiMesh(self.nodes, self.edges, self.tones, tout, h, side, valid, None, dict(self.info, fps=fps))

    def summary(self):
        return (f"CSI メッシュ: ノード {len(self.nodes)}、辺 {self.E}、サブキャリア {len(self.tones)}、フレーム {self.T} "
                f"({self.T / max(self.t[-1] - self.t[0], 1e-9):.0f} Hz)、受信率 {self.valid.mean() * 100:.1f}%")


def from_sim(sim, imp=None, links=None):
    """sim = simio.load_sim の結果。imp = simio.Esp32Impairment (None なら欠点なし)。
    links = 使う辺 [(送信, 受信)] (None なら自分自身以外の全部)。リンクの並びは 送信番号 * 受信機の数 + 受信番号。"""
    meta = sim["meta"]; tones = np.array(meta["tones"])
    tx = np.array(meta["tx"], float); rx = np.array(meta["rx"], float)
    same_nodes = tx.shape == rx.shape and np.allclose(tx, rx)
    nodes = tx if same_nodes else np.vstack([tx, rx])
    nrx = len(rx); off = 0 if same_nodes else len(tx)
    all_links = [(a, b) for a in range(len(tx)) for b in range(nrx) if not (same_nodes and a == b)]
    links = links or all_links
    li = [a * nrx + b for a, b in links]
    H = sim["H"][:, li]
    if imp is not None: X, ok, _ = imp.apply(H, tones)
    else: X, ok = H, np.ones(H.shape[:2], bool)
    h, side = canon(X.astype(np.complex128), tones)
    edges = [(a, b + off) for a, b in links]
    t = np.arange(H.shape[0]) / meta["fps"]
    return CsiMesh(nodes, edges, tones, t, h, side, ok, X, dict(source="sim"))


# classic ESP32 の HT20 HT-LTF (FFT 順) → 周波数順 -28..-1, 1..28
_CLASSIC_HT20 = list(range(36, 64)) + list(range(1, 29))

def from_capture(npz_path, strobe_pos=None, rx_pos=None):
    """実機の収録 (capture_multi.py) → メッシュ。ストロボ (ノード 0) → 受信機 (ノード 1..N)。
    フレームはストロボの通し番号でそろえる (時刻 = 通し番号 / 100 Hz)。位置が分からなければ NaN。"""
    d = np.load(npz_path); ports = [str(p) for p in d["ports"]]
    tones = np.array(list(range(-28, 0)) + list(range(1, 29)))
    seqs = {}
    for p in ports:
        ss = d[f"{p}_sseq"]; ok = ss != 0xFFFFFFFF
        r = d[f"{p}_raw"][ok].astype(np.float64)[:, 128:256]
        c = (r[:, 1::2] + 1j * r[:, 0::2])[:, _CLASSIC_HT20]
        seqs[p] = dict(zip(ss[ok].tolist(), c))
    allseq = np.array(sorted(set().union(*[set(s) for s in seqs.values()])))
    T, E, K = len(allseq), len(ports), len(tones)
    X = np.zeros((T, E, K), complex); valid = np.zeros((T, E), bool)
    pos = {s: i for i, s in enumerate(allseq.tolist())}
    for e, p in enumerate(ports):
        for s, c in seqs[p].items(): X[pos[s], e] = c; valid[pos[s], e] = True
    h, side = canon(X, tones)
    nodes = np.full((E + 1, 3), np.nan)
    if strobe_pos is not None: nodes[0] = strobe_pos
    if rx_pos is not None: nodes[1:] = rx_pos
    return CsiMesh(nodes, [(0, e + 1) for e in range(E)], tones, (allseq - allseq[0]) / 100.0, h, side, valid, X,
                   dict(source="capture", ports=ports))
