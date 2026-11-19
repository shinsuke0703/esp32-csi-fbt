// CSI シミュレータの電波モデル (Unity に依存しない。Unity と .NET のコマンドラインで共用)
//
// 1 リンク (送信 tx → 受信 rx) の周波数応答 H(f) を、経路の和で作る:
//   直接波        a = c / (4π f d)                                   × 遮蔽
//   床での反射    同上を鏡像で計算し、反射係数 Γ を掛ける               × 遮蔽
//   散乱 (人体・物体) 点散乱体の 2 局レーダー式: a = sqrt(σ) c / ((4π)^1.5 f d1 d2)、反射で位相 π
//   各経路の位相は exp(-j 2π f L / c) (L = 経路の長さ)
// 遮蔽は、経路の線分がカプセルの中を通る深さに応じて、なめらかに振幅を下げる (ヤコビアンを取りやすくするため)。
// ここでは ESP32 の欠点 (位相の乱れ・AGC・量子化・雑音) は入れない。それは検証側 (Python) でわざと加える。
using System;
using System.Collections.Generic;
using System.Numerics;

namespace CsiSim.Core
{
    /// <summary>散乱体と遮蔽体を兼ねるカプセル (線分 A-B と半径 R)。</summary>
    public struct Capsule
    {
        public Vector3 A, B;
        public float R;
        public float Rho;      // 単位投影面積あたりのレーダー断面積 (反射の強さ)
        public int Id;         // 自分自身による遮蔽を除くための番号
        public bool Blocks;    // 遮蔽体として扱うか
        public Capsule(Vector3 a, Vector3 b, float r, float rho, int id, bool blocks = true)
        { A = a; B = b; R = r; Rho = rho; Id = id; Blocks = blocks; }
    }

    [Serializable]
    public class RfSettings
    {
        public double fc = 2.447e9;          // HT40 (ch6 + 上側の副チャネル) の中心周波数
        public double df = 312.5e3;          // サブキャリア間隔
        public bool floor = true;            // 床での反射 (直接波のみ)
        public float floorY = 0f;
        public double floorGamma = -0.5;     // 床の反射係数
        public float blockAtt = 0.2f;        // カプセルの中心を通ったときの振幅の倍率
        public float pointSpacing = 0.15f;   // カプセルを点散乱体に分ける間隔 [m]
    }

    public static class RfModel
    {
        public const double C = 299792458.0;

        /// <summary>HT40 の 114 本 (-58..-2, 2..58)。</summary>
        public static int[] Ht40Tones()
        {
            var t = new List<int>();
            for (int k = -58; k <= 58; k++) if (Math.Abs(k) >= 2) t.Add(k);
            return t.ToArray();
        }

        /// <summary>線分 p-q と線分 a-b の最短距離。</summary>
        public static float SegSegDist(Vector3 p, Vector3 q, Vector3 a, Vector3 b)
        {
            Vector3 d1 = q - p, d2 = b - a, r = p - a;
            float A = Vector3.Dot(d1, d1), E = Vector3.Dot(d2, d2), F = Vector3.Dot(d2, r);
            float s, t;
            if (A <= 1e-12f && E <= 1e-12f) return r.Length();
            if (A <= 1e-12f) { s = 0; t = Clamp01(F / E); }
            else
            {
                float Cc = Vector3.Dot(d1, r);
                if (E <= 1e-12f) { t = 0; s = Clamp01(-Cc / A); }
                else
                {
                    float B = Vector3.Dot(d1, d2), den = A * E - B * B;
                    s = den > 1e-12f ? Clamp01((B * F - Cc * E) / den) : 0f;
                    t = (B * s + F) / E;
                    if (t < 0) { t = 0; s = Clamp01(-Cc / A); }
                    else if (t > 1) { t = 1; s = Clamp01((B - Cc) / A); }
                }
            }
            return ((p + d1 * s) - (a + d2 * t)).Length();
        }

        static float Clamp01(float v) => v < 0 ? 0 : (v > 1 ? 1 : v);

        /// <summary>線分 p-q がカプセル群に遮られる割合 (1 = 遮られない)。exclude の番号のカプセルは除く。</summary>
        public static double Block(Vector3 p, Vector3 q, IReadOnlyList<Capsule> caps, int exclude, float att)
        {
            double f = 1.0;
            for (int i = 0; i < caps.Count; i++)
            {
                var c = caps[i];
                if (!c.Blocks || c.Id == exclude) continue;
                float dist = SegSegDist(p, q, c.A, c.B);
                if (dist >= c.R) continue;
                double depth = 1.0 - dist / c.R;                          // 0 (表面) .. 1 (中心)
                double w = depth * depth * (3 - 2 * depth);                // なめらかに
                f *= 1.0 - (1.0 - att) * w;
            }
            return f;
        }

        /// <summary>経路 1 本の情報 (表示や解析用)。</summary>
        public struct PathInfo
        {
            public Vector3 P0, P1, P2;   // 送信 → (散乱点 or 反射点) → 受信。直接波は P1 = P2
            public double Amp;           // 中心周波数での振幅 (遮蔽込み)
            public double Length;        // 経路の長さ [m]
            public int Kind;             // 0 直接, 1 床反射, 2 散乱
            public int SourceId;         // 散乱体のカプセル番号
        }

        /// <summary>
        /// 1 リンクの周波数応答を計算する。re/im は tones と同じ長さ。paths を渡すと経路を記録する。
        /// </summary>
        public static void Synthesize(Vector3 tx, Vector3 rx, IReadOnlyList<Capsule> caps, RfSettings s, int[] tones,
                                      double[] re, double[] im, List<PathInfo> paths = null)
        {
            Array.Clear(re, 0, re.Length); Array.Clear(im, 0, im.Length);
            paths?.Clear();
            int K = tones.Length;
            var f = new double[K];
            for (int k = 0; k < K; k++) f[k] = s.fc + tones[k] * s.df;

            // 経路を足す: 振幅 a0 (中心周波数での値。周波数の違いは 1/f で補正) と長さ L、反射の符号
            void Add(double a0, double L, double sign)
            {
                for (int k = 0; k < K; k++)
                {
                    double a = a0 * s.fc / f[k] * sign;
                    double ph = -2 * Math.PI * f[k] * L / C;
                    re[k] += a * Math.Cos(ph); im[k] += a * Math.Sin(ph);
                }
            }

            // 送信と受信が同じ機器 (送受信を交代する構成で、自分自身へのリンク) は測れないので 0 のまま返す
            if ((rx - tx).Length() < 1e-4f) return;

            // 直接波
            double d = (rx - tx).Length();
            double blk = Block(tx, rx, caps, -1, s.blockAtt);
            double a0 = C / (4 * Math.PI * s.fc * d) * blk;
            Add(a0, d, 1);
            paths?.Add(new PathInfo { P0 = tx, P1 = rx, P2 = rx, Amp = a0, Length = d, Kind = 0, SourceId = -1 });

            // 床での反射 (鏡像法)
            if (s.floor)
            {
                var img = new Vector3(tx.X, 2 * s.floorY - tx.Y, tx.Z);
                double L = (rx - img).Length();
                float tt = (tx.Y - s.floorY) / ((tx.Y - s.floorY) + (rx.Y - s.floorY) + 1e-9f);
                var rp = tx + (new Vector3(rx.X, tx.Y, rx.Z) - tx) * tt; rp.Y = s.floorY;
                double b = Block(tx, rp, caps, -1, s.blockAtt) * Block(rp, rx, caps, -1, s.blockAtt);
                double a1 = C / (4 * Math.PI * s.fc * L) * Math.Abs(s.floorGamma) * b;
                Add(a1, L, Math.Sign(s.floorGamma));
                paths?.Add(new PathInfo { P0 = tx, P1 = rp, P2 = rx, Amp = a1, Length = L, Kind = 1, SourceId = -1 });
            }

            // 散乱 (カプセルを点散乱体に分ける)
            for (int i = 0; i < caps.Count; i++)
            {
                var c = caps[i];
                if (c.Rho <= 0) continue;
                float len = (c.B - c.A).Length();
                int n = Math.Max(1, (int)Math.Ceiling(len / s.pointSpacing));
                double area = (2 * c.R * len + Math.PI * c.R * c.R) / n;   // 投影面積を点に分ける
                double sigma = c.Rho * area;
                for (int j = 0; j < n; j++)
                {
                    var pt = c.A + (c.B - c.A) * ((j + 0.5f) / n);
                    double d1 = (pt - tx).Length(), d2 = (rx - pt).Length();
                    double bb = Block(tx, pt, caps, c.Id, s.blockAtt) * Block(pt, rx, caps, c.Id, s.blockAtt);
                    double a2 = Math.Sqrt(sigma) * C / (Math.Pow(4 * Math.PI, 1.5) * s.fc * d1 * d2) * bb;
                    Add(a2, d1 + d2, -1);
                    paths?.Add(new PathInfo { P0 = tx, P1 = pt, P2 = rx, Amp = a2, Length = d1 + d2, Kind = 2, SourceId = c.Id });
                }
            }
        }
    }
}
