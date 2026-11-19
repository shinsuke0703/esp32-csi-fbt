// 動きの生成: 時刻 t → 姿勢パラメータ。正解 (関節角と関節位置) がそのまま手に入る。Unity に依存しない。
using System;
using System.Collections.Generic;

namespace CsiSim.Core
{
    [Serializable]
    public class MotionSegment
    {
        public string type = "idle";   // tpose / idle / walk / arms / squat / reach / random / still
        public float dur = 10f;        // 秒
        public float x = 0f, z = 0f;   // 立つ位置 (walk は円の中心)
        public float radius = 1.2f;    // walk の円の半径
        public float speed = 0.8f;     // walk の速さ [m/s]
        public float yaw = 0f;         // 向き [deg]
    }

    public class MotionPlayer
    {
        readonly List<MotionSegment> segs;
        readonly double[] start;
        readonly double[][] rndAmp, rndFreq, rndPhase;
        public readonly double Total;
        const double Blend = 0.5;   // 区間の切り替えでなめらかにつなぐ時間 [s]

        public MotionPlayer(IEnumerable<MotionSegment> segments, int seed)
        {
            segs = new List<MotionSegment>(segments);
            start = new double[segs.Count + 1];
            for (int i = 0; i < segs.Count; i++) start[i + 1] = start[i] + segs[i].dur;
            Total = start[segs.Count];
            var rng = new Random(seed);
            rndAmp = new double[Humanoid.NP][]; rndFreq = new double[Humanoid.NP][]; rndPhase = new double[Humanoid.NP][];
            for (int k = 0; k < Humanoid.NP; k++)
            {
                rndAmp[k] = new double[3]; rndFreq[k] = new double[3]; rndPhase[k] = new double[3];
                for (int m = 0; m < 3; m++) { rndAmp[k][m] = rng.NextDouble(); rndFreq[k][m] = 0.1 + 0.6 * rng.NextDouble(); rndPhase[k][m] = 2 * Math.PI * rng.NextDouble(); }
            }
        }

        public double[] Pose(double t)
        {
            int i = 0;
            while (i < segs.Count - 1 && t >= start[i + 1]) i++;
            var p = SegPose(segs[i], t - start[i], t);
            double into = t - start[i];
            if (i > 0 && into < Blend)   // 前の区間からなめらかに移る
            {
                var q = SegPose(segs[i - 1], t - start[i - 1], t);
                double w = into / Blend; w = w * w * (3 - 2 * w);
                for (int k = 0; k < p.Length; k++) p[k] = q[k] + (p[k] - q[k]) * w;
            }
            return p;
        }

        static double D(double deg) => deg * Math.PI / 180;

        double[] SegPose(MotionSegment s, double tl, double tg)
        {
            var p = new double[Humanoid.NP];
            p[0] = s.x; p[1] = s.z; p[2] = 0.98; p[3] = D(s.yaw);
            switch (s.type)
            {
                case "tpose":
                    p[8] = p[11] = D(90); break;
                case "still":
                    break;
                case "idle":   // 呼吸程度の小さな揺れ
                    p[4] = D(2) * Math.Sin(2 * Math.PI * 0.25 * tg); p[8] = p[11] = D(8);
                    p[0] += 0.01 * Math.Sin(2 * Math.PI * 0.2 * tg); break;
                case "walk":
                {
                    double w = s.speed / s.radius, th = w * tl;   // 円の上を歩く
                    // 中心 (x, z) の円の上を歩く。進行方向 (cosθ, sinθ) を向く: Ry(yaw)·(0,0,1) = (sin yaw, 0, cos yaw) なので yaw = π/2 - θ
                    p[0] = s.x + s.radius * Math.Sin(th); p[1] = s.z - s.radius * Math.Cos(th);
                    p[3] = Math.PI / 2 - th;
                    double g = 2 * Math.PI * (s.speed / 1.2) * tl;   // 歩調
                    p[13] = D(25) * Math.Sin(g); p[16] = -D(25) * Math.Sin(g);
                    p[15] = D(30) * Math.Max(0, Math.Sin(g - 1.2)); p[18] = D(30) * Math.Max(0, Math.Sin(-g - 1.2));
                    p[7] = -D(20) * Math.Sin(g); p[10] = D(20) * Math.Sin(g); p[9] = p[12] = D(15);
                    p[8] = p[11] = D(8);
                    p[2] = 0.98 - 0.02 * Math.Abs(Math.Sin(g));
                    break;
                }
                case "arms":   // 両腕を交互に横・前に上げ下げ
                {
                    double u = 0.5 - 0.5 * Math.Cos(2 * Math.PI * 0.3 * tl), v = 0.5 - 0.5 * Math.Cos(2 * Math.PI * 0.3 * tl + Math.PI);
                    p[8] = D(10) + D(150) * u; p[10] = D(10) + D(150) * v; p[9] = D(30) * v; p[12] = D(30) * u; break;
                }
                case "squat":
                {
                    double u = 0.5 - 0.5 * Math.Cos(2 * Math.PI * 0.25 * tl);
                    p[13] = p[16] = D(90) * u; p[15] = p[18] = D(110) * u; p[4] = D(25) * u; p[2] = 0.98 - 0.40 * u;
                    p[7] = p[10] = D(70) * u; break;
                }
                case "reach":   // 前かがみで右手を伸ばす
                {
                    double u = 0.5 - 0.5 * Math.Cos(2 * Math.PI * 0.2 * tl);
                    p[4] = D(45) * u; p[10] = D(100) * u; p[12] = D(10); p[8] = D(10); break;
                }
                case "random":   // 関節ごとに、ゆっくり変わる不規則な動き (可動域の中)
                {
                    double[] lo = { 0, 0, 0, -180, -10, -15, -60, -30, 0, 0, -30, 0, 0, -20, 0, 0, -20, 0, 0 };
                    double[] hi = { 0, 0, 0, 180, 40, 15, 60, 160, 160, 130, 160, 160, 130, 90, 40, 120, 90, 40, 120 };
                    for (int k = 3; k < Humanoid.NP; k++)
                    {
                        double v = 0, sum = 0;
                        for (int m = 0; m < 3; m++) { v += rndAmp[k][m] * Math.Sin(2 * Math.PI * rndFreq[k][m] * tg + rndPhase[k][m]); sum += rndAmp[k][m]; }
                        double u = 0.5 + 0.5 * v / sum;
                        p[k] = D(lo[k] + (hi[k] - lo[k]) * u);
                    }
                    // 両膝を曲げた分だけ骨盤を下げる (足が床に刺さらないように、簡易的に)
                    double bend = Math.Max(1 - Math.Cos(p[15]) , 1 - Math.Cos(p[18])) * 0.43 + Math.Max(1 - Math.Cos(p[13]), 1 - Math.Cos(p[16])) * 0.2;
                    p[2] = 0.98 - Math.Min(0.35, bend);
                    break;
                }
            }
            return p;
        }
    }
}
