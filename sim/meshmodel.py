# -*- coding: utf-8 -*-
"""CSI メッシュに当てる数理モデル (AI なし) と、それを使った姿勢の推定器。

観測モデル (辺 e = ノード a → b):
    H_e(p) = B_e + Σ_{体の散乱点 i} s_i(p)·exp(-j2πf(d_ai + d_ib)/c)  −  (1 − b_e(p))·D_e
      B_e  地図 (人がいないときに測った背景)。床や物体の反射も含めて、測った値をそのまま使う
      D_e  直接波 (ノードの位置から計算)
      b_e  体による直接波の遮蔽 (0..1)
      s_i  点散乱体の 2 局レーダー式の振幅
    観測できるのは外乱 (φ0, τ, a) を外した「形」だけ: canon(H_e(p)) ≈ h_e (CSI メッシュの値)

推定 (フレームごと):  min_p  Σ_e ||h_e − canon(H_e(p))||² / σ_h²  +  ||(p − p_pred) / σ_motion||²
    p_pred は等速の予測。レーベンバーグ・マーカート法、ヤコビアンは数値微分。

シミュレータ (C#) より、わざと粗いモデルにしている (同じモデルで解くと結果が良く出すぎるため):
  散乱点の間隔 0.25 m (シミュレータは 0.15 m)、散乱の経路の遮蔽なし、床・物体での散乱の変化なし。
"""
import numpy as np
from csimesh import canon

C = 299792458.0
NP = 19
PARAM_NAMES = ["root_x", "root_z", "pelvis_h", "yaw", "spine_pitch", "spine_roll", "head_yaw",
               "l_sh_pitch", "l_sh_roll", "l_elbow", "r_sh_pitch", "r_sh_roll", "r_elbow",
               "l_hip_pitch", "l_hip_roll", "l_knee", "r_hip_pitch", "r_hip_roll", "r_knee"]
D = np.deg2rad
LO = np.array([-9, -9, 0.4, -1e9, D(-20), D(-30), D(-90), D(-60), D(-10), 0, D(-60), D(-10), 0, D(-40), D(-20), 0, D(-40), D(-20), 0])
HI = np.array([9, 9, 1.1, 1e9, D(70), D(30), D(90), D(180), D(180), D(150), D(180), D(180), D(150), D(120), D(60), D(140), D(120), D(60), D(140)])

# ---------------- 体のモデル (C# の Humanoid と同じ骨格) ----------------
BODY = dict(torso=0.45, neck=0.10, head=0.20, shoulderW=0.18, shoulderH=0.48, upperArm=0.30, forearm=0.27, hand=0.08,
            hipW=0.10, thigh=0.45, shin=0.43, foot=0.15, rTorso=0.14, rChest=0.15, rHead=0.10, rUpperArm=0.05, rForearm=0.04,
            rHand=0.035, rThigh=0.07, rShin=0.05, rFoot=0.04, rho=0.5)

def _rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# System.Numerics の Quaternion.CreateFromAxisAngle と Vector3.Transform は右手の回転行列と同じ向きに回す

def joints(p, b=BODY):
    """順運動学 (C# の Humanoid.Joints と同じ)。p [19] → 関節 [20, 3]"""
    J = np.zeros((20, 3)); P = np.array([p[0], p[2], p[1]])
    root = _ry(p[3]); spine = root @ _rx(p[4]) @ _rz(p[5])
    J[0] = P; J[1] = P + spine @ [0, b["torso"] * 0.5, 0]
    J[2] = P + spine @ [0, b["shoulderH"] + b["neck"] * 0.5, 0]
    J[3] = P + spine @ [0, b["shoulderH"] + b["neck"] + b["head"] * 0.5, 0.02]
    for side, o, j in ((0, 7, 4), (1, 10, 8)):
        sx = -1 if side == 0 else 1
        sh = P + spine @ [sx * b["shoulderW"], b["shoulderH"], 0]
        arm = spine @ _rx(-p[o]) @ _rz(sx * p[o + 1]); fore = arm @ _rx(-p[o + 2])
        J[j] = sh; J[j + 1] = sh + arm @ [0, -b["upperArm"], 0]
        J[j + 2] = J[j + 1] + fore @ [0, -b["forearm"], 0]; J[j + 3] = J[j + 2] + fore @ [0, -b["hand"], 0]
    for side, o, j in ((0, 13, 12), (1, 16, 16)):
        sx = -1 if side == 0 else 1
        hip = P + root @ [sx * b["hipW"], -0.05, 0]
        thigh = root @ _rx(-p[o]) @ _rz(sx * p[o + 1]); shin = thigh @ _rx(p[o + 2])
        J[j] = hip; J[j + 1] = hip + thigh @ [0, -b["thigh"], 0]
        J[j + 2] = J[j + 1] + shin @ [0, -b["shin"], 0]; J[j + 3] = J[j + 2] + root @ [0, -0.03, b["foot"]]
    return J

def capsules(p, J, b=BODY):
    """関節 → カプセル 15 個 (A [15,3], B [15,3], R [15])。C# の Humanoid.Capsules と同じ並び。"""
    spine = _ry(p[3]) @ _rx(p[4]) @ _rz(p[5])
    nb = J[0] + spine @ [0, b["shoulderH"] + b["neck"], 0]
    segs = [(J[0], J[1], b["rTorso"]), (J[1], J[2], b["rChest"]), (nb, nb + spine @ [0, b["head"] - 2 * b["rHead"] + 0.01, 0.02], b["rHead"])]
    for j in (4, 8): segs += [(J[j], J[j + 1], b["rUpperArm"]), (J[j + 1], J[j + 2], b["rForearm"]), (J[j + 2], J[j + 3], b["rHand"])]
    for j in (12, 16): segs += [(J[j], J[j + 1], b["rThigh"]), (J[j + 1], J[j + 2], b["rShin"]), (J[j + 2], J[j + 3], b["rFoot"])]
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs]); R = np.array([s[2] for s in segs])
    return A, B, R

def _seg_seg_dist(P, Q, A, B):
    """線分 P-Q [E,3] と線分 A-B [S,3] の最短距離 [E,S] (C# の SegSegDist と同じ式、ベクトル化)。"""
    d1 = (Q - P)[:, None]; d2 = (B - A)[None]; r = P[:, None] - A[None]
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1); c = (d1 * r).sum(-1); bb = (d1 * d2).sum(-1)
    den = a * e - bb * bb
    s = np.clip(np.where(den > 1e-12, (bb * f - c * e) / np.where(den > 1e-12, den, 1), 0), 0, 1)
    t = (bb * s + f) / np.maximum(e, 1e-12)
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((bb - c) / a, 0, 1), s)); t = np.clip(t, 0, 1)
    return np.linalg.norm((P[:, None] + d1 * s[..., None]) - (A[None] + d2 * t[..., None]), axis=-1)


class MeshModel:
    """CSI メッシュ全体の観測モデル。edges = [(a, b)]、nodes [M, 3]、background [E, K] (地図)。"""

    def __init__(self, nodes, edges, tones, background, fc=2.447e9, df=312.5e3, block_att=0.2, spacing=0.25, tone_step=1,
                 floor_gamma=-0.5, leg_block=True):
        self.tsel = np.arange(0, len(tones), tone_step)
        self.tones = np.asarray(tones)[self.tsel]; self.f = fc + self.tones * df; self.fc = fc
        self.Pa = nodes[[a for a, _ in edges]]; self.Pb = nodes[[b for _, b in edges]]
        self.B = background[:, self.tsel]; self.block_att = block_att; self.spacing = spacing; self.leg_block = leg_block
        d = np.linalg.norm(self.Pb - self.Pa, axis=1)
        self.Dlos = C / (4 * np.pi * self.f[None] * d[:, None]) * np.exp(-2j * np.pi * self.f[None] * d[:, None] / C)   # [E, K]
        # 床での反射 (鏡像法)。体がこの経路をさえぎる分を、地図から差し引くために使う
        self.floor = floor_gamma != 0
        if self.floor:
            img = self.Pa * [1, -1, 1]; L = np.linalg.norm(self.Pb - img, axis=1)
            tt = self.Pa[:, 1] / (self.Pa[:, 1] + self.Pb[:, 1]); rp = self.Pa + (self.Pb * [1, 0, 1] + self.Pa * [0, 1, 0] - self.Pa) * tt[:, None]; rp[:, 1] = 0
            self.Prp = rp
            self.Dfloor = floor_gamma * C / (4 * np.pi * self.f[None] * L[:, None]) * np.exp(-2j * np.pi * self.f[None] * L[:, None] / C)

    def _block(self, P, Q, A, Bc, R, own=None):
        """線分 P-Q [E or E*N, 3] がカプセルに遮られる割合。own [行] = その行で除くカプセル番号。"""
        dist = _seg_seg_dist(P, Q, A, Bc)
        depth = np.clip(1 - dist / R[None], 0, 1); w = depth * depth * (3 - 2 * depth)
        if own is not None: w[np.arange(len(own)), own] = 0
        return np.prod(1 - (1 - self.block_att) * w, axis=1)

    def H(self, p):
        J = joints(p); A, Bc, R = capsules(p, J)
        # 体の散乱点
        pts, sig, own = [], [], []
        for i in range(len(A)):
            L = np.linalg.norm(Bc[i] - A[i]); n = max(1, int(np.ceil(L / self.spacing)))
            area = (2 * R[i] * L + np.pi * R[i] ** 2) / n
            for j in range(n): pts.append(A[i] + (Bc[i] - A[i]) * (j + 0.5) / n); sig.append(BODY["rho"] * area); own.append(i)
        pts = np.array(pts); sig = np.array(sig); own = np.array(own); E, N = len(self.Pa), len(pts)
        d1 = np.linalg.norm(pts[None] - self.Pa[:, None], axis=-1); d2 = np.linalg.norm(self.Pb[:, None] - pts[None], axis=-1)   # [E, N]
        amp = np.sqrt(sig)[None] * C / ((4 * np.pi) ** 1.5 * self.fc * d1 * d2)
        if self.leg_block:   # 散乱の経路 (送信 → 散乱点、散乱点 → 受信) が体の他の部位に遮られる分
            Pa = np.repeat(self.Pa, N, 0); Pb = np.repeat(self.Pb, N, 0); Pt = np.tile(pts, (E, 1)); ow = np.tile(own, E)
            amp = amp * (self._block(Pa, Pt, A, Bc, R, ow) * self._block(Pt, Pb, A, Bc, R, ow)).reshape(E, N)
        L = d1 + d2
        Hs = -(amp[..., None] * (self.fc / self.f)[None, None] * np.exp(-2j * np.pi * self.f[None, None] * L[..., None] / C)).sum(1)
        H = self.B + Hs - (1 - self._block(self.Pa, self.Pb, A, Bc, R))[:, None] * self.Dlos   # 直接波の遮蔽
        if self.floor:   # 床で反射する経路の遮蔽
            bf = self._block(self.Pa, self.Prp, A, Bc, R) * self._block(self.Prp, self.Pb, A, Bc, R)
            H = H - (1 - bf)[:, None] * self.Dfloor
        return H

    def shape(self, p):
        h, _ = canon(self.H(p), self.tones); return h

    def residual(self, p, hobs, valid):
        # 観測側も、モデルと同じ間引いたサブキャリアで外乱を外し直す (両方を同じ操作にそろえる)
        ho, _ = canon(hobs[:, self.tsel], self.tones)
        r = (ho - self.shape(p))[valid]
        return np.concatenate([r.real.ravel(), r.imag.ravel()])


class Tracker:
    """フレームごとに LM 法で姿勢を推定する。sigma_h: 形の 1 要素あたりの雑音 (実部・虚部それぞれ)。"""

    def __init__(self, model, p0, sigma_h=0.05, sig_pos=0.03, sig_ang=np.deg2rad(6), iters=3, eps=1e-3, fixed=(6,)):
        self.m = model; self.p = np.array(p0, float); self.v = np.zeros(NP); self.sh = sigma_h
        self.sig = np.array([sig_pos] * 3 + [sig_ang] * (NP - 3)); self.iters = iters; self.eps = eps
        self.free = np.array([i for i in range(NP) if i not in fixed])

    def step(self, hobs, valid):
        pred = self.p + self.v; p = pred.copy(); free = self.free
        if valid.sum() == 0: self.p, self.v = pred, self.v * 0.9; return pred
        lam = 1e-2
        for _ in range(self.iters):
            r0 = self.m.residual(p, hobs, valid) / self.sh
            Jm = np.zeros((len(r0), len(free)))
            for c, i in enumerate(free):
                q = p.copy(); q[i] += self.eps; Jm[:, c] = (self.m.residual(q, hobs, valid) / self.sh - r0) / self.eps
            Wp = 1 / self.sig[free] ** 2
            Hm = Jm.T @ Jm + np.diag(Wp); g = Jm.T @ r0 + Wp * (p[free] - pred[free])
            dp = -np.linalg.solve(Hm + lam * np.diag(np.diag(Hm)), g)
            q = p.copy(); q[free] += dp; q = np.clip(q, LO, HI)
            c_old = r0 @ r0 + (Wp * (p[free] - pred[free]) ** 2).sum()
            r1 = self.m.residual(q, hobs, valid) / self.sh; c_new = r1 @ r1 + (Wp * (q[free] - pred[free]) ** 2).sum()
            if c_new < c_old: p, lam = q, lam * 0.3
            else: lam *= 10
        self.v = 0.7 * (p - self.p); self.p = p
        return p
