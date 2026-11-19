"""人を囲うが、高さと距離がばらばらな配置を作る。囲う条件: 隣り合うノードの方位角の差が 150° 以下。"""
import json, math, sys
import numpy as np
n_layouts = int(sys.argv[1]) if len(sys.argv) > 1 else 20; M = int(sys.argv[2]) if len(sys.argv) > 2 else 6
base = json.load(open(r"E:/CSI/CsiSim/Assets/CsiSim/Configs/six_nodes.json", encoding="utf-8"))
rng = np.random.default_rng(2026); made = 0
while made < n_layouts:
    ang = np.sort(rng.uniform(0, 2 * np.pi, M))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    if gaps.max() > np.deg2rad(150): continue
    r = rng.uniform(1.8, 3.8, M); y = rng.uniform(0.3, 2.5, M)
    nodes = [{"x": round(float(r[i] * math.sin(ang[i])), 3), "y": round(float(y[i]), 3), "z": round(float(r[i] * math.cos(ang[i])), 3)} for i in range(M)]
    c = dict(base); c["tx"] = nodes; c["rx"] = nodes; c["out"] = f"E:/CSI/sim_out/rand{made}"
    json.dump(c, open(rf"E:/CSI/CsiSim/Assets/CsiSim/Configs/layout_rand{made}.json", "w", encoding="utf-8"), indent=1)
    made += 1
print("作成:", made)
