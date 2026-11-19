# -*- coding: utf-8 -*-
"""CSI シミュレータ (E:/CSI/CsiSim, E:/CSI/CsiSimCli) の出力を読み、ESP32 の欠点をわざと加える。

シミュレータは理想的な H (物理モデルそのもの) を出す。検証では、ここで実機に近い欠点を加えてから数理モデルに渡す。
既定値は 2026-11-18 に実機 (ESP32, ESP-NOW ストロボ) で測った値:
  共通位相 φ0: パケットごとに一様分布 (隣接フレームの差の std 1.80 rad ≒ 一様)
  位相の傾き τ: 平均 -0.0033, std 0.0043 rad/サブキャリア
  受信率: 74〜99% (リンクによる)
"""
import json
import numpy as np


def load_sim(base):
    """base = 拡張子なしのパス。戻り値: dict(H [T,L,K] complex64, joints [T,NJ,3], params [T,NP], meta)"""
    meta = json.load(open(base + ".json", encoding="utf-8"))
    T, L, K, NJ, NP = meta["T"], meta["L"], meta["K"], meta["NJ"], meta["NP"]
    a = np.fromfile(base + ".bin", dtype=np.float32)
    n1, n2 = T * L * K * 2, T * NJ * 3
    assert a.size == n1 + n2 + T * NP, f"サイズが合わない: {a.size}"
    H = a[:n1].reshape(T, L, K, 2); H = (H[..., 0] + 1j * H[..., 1]).astype(np.complex64)
    return dict(H=H, joints=a[n1:n1 + n2].reshape(T, NJ, 3), params=a[n1 + n2:].reshape(T, NP), meta=meta)


class Esp32Impairment:
    """ESP32 の欠点のモデル。各項目は 0 / False にすれば外せる (どの欠点が効くかを一つずつ調べるため)。

    snr_db       : 熱雑音。リンクごとの平均電力に対する SNR [dB] (None で雑音なし)
    random_phase : パケットごとの共通位相 φ0 (一様分布)
    tau_mean/std : パケットごとの位相の傾き [rad/サブキャリア] (STO)
    agc_db_std   : パケットごとの利得のばらつき [dB]。その上で振幅を target_rms にそろえる (AGC)
    target_rms   : AGC 後の振幅の RMS (int8 の値の大きさ。実機では 20〜40 程度)
    quantize     : int8 に丸めて ±127 で切る
    loss         : パケットの抜ける確率 (リンクごとの配列か、全リンク共通の値)
    """

    def __init__(self, snr_db=25.0, random_phase=True, tau_mean=-0.0033, tau_std=0.0043, agc_db_std=0.5,
                 target_rms=30.0, quantize=True, loss=0.05, seed=0):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.rng = np.random.default_rng(seed)

    def apply(self, H, tones):
        """H [T,L,K] → (欠点を加えた H [T,L,K] complex64, 受信できたか [T,L] bool, 加えた値の dict)"""
        rng = self.rng; T, L, K = H.shape; k = np.asarray(tones, float)
        X = H.astype(np.complex128).copy()
        if self.snr_db is not None:   # 雑音はリンクごとの平均電力を基準に
            p = (np.abs(H) ** 2).mean(axis=(0, 2), keepdims=True)
            s = np.sqrt(p / 10 ** (self.snr_db / 10) / 2)
            X += s * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
        phi = rng.uniform(-np.pi, np.pi, (T, L)) if self.random_phase else np.zeros((T, L))
        tau = rng.normal(self.tau_mean, self.tau_std, (T, L)) if self.tau_std or self.tau_mean else np.zeros((T, L))
        X *= np.exp(1j * (phi[..., None] + tau[..., None] * k))
        g_db = rng.normal(0, self.agc_db_std, (T, L)) if self.agc_db_std else np.zeros((T, L))
        rms = np.sqrt((np.abs(X) ** 2).mean(axis=2))                               # パケットごとに AGC でそろえる
        X *= (self.target_rms / (rms + 1e-30) * 10 ** (g_db / 20))[..., None]
        if self.quantize:
            X = np.clip(np.round(X.real), -127, 127) + 1j * np.clip(np.round(X.imag), -127, 127)
        loss = np.broadcast_to(np.asarray(self.loss, float), (L,))
        ok = rng.random((T, L)) >= loss[None, :]
        return X.astype(np.complex64), ok, dict(phi=phi, tau=tau, agc_db=g_db)
