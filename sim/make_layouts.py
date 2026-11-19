"""配置 (2D リング / 3D メッシュ) の設定 JSON を作る。全員が送受信を兼ねる (tx = rx = 全ノード)。"""
import json, math
base = json.load(open(r"E:/CSI/CsiSim/Assets/CsiSim/Configs/six_nodes.json", encoding="utf-8"))
R = 3.0
def ring(n, heights, r=R, phase=0):
    return [{"x": round(r * math.sin(2 * math.pi * i / n + phase), 3), "y": heights[i % len(heights)], "z": round(r * math.cos(2 * math.pi * i / n + phase), 3)} for i in range(n)]
layouts = {
    "ring2d":   ring(6, [1.2]),                                   # 全員が腰の高さ (平面)
    "current":  base["tx"],                                       # 今の配置 (高さ 1.0〜2.4、周囲)
    "mesh3d":   ring(6, [0.3, 2.4]),                              # 低い・高いを交互 (三角の反角柱)
    "mesh3d_ceil": ring(5, [0.3, 2.4, 1.2, 0.3, 2.4]) + [{"x": 0, "y": 2.7, "z": 0}],   # 周囲 5 台 + 天井の中央
    "cube8":    [{"x": sx * 2.6, "y": y, "z": sz * 2.6} for y in (0.3, 2.4) for sx in (-1, 1) for sz in (-1, 1)],  # 参考: 8 台で立方体の角
}
for name, nodes in layouts.items():
    c = dict(base); c["tx"] = nodes; c["rx"] = nodes; c["out"] = f"E:/CSI/sim_out/{name}"
    json.dump(c, open(rf"E:/CSI/CsiSim/Assets/CsiSim/Configs/layout_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(name, len(nodes), [(n["x"], n["y"], n["z"]) for n in nodes])
