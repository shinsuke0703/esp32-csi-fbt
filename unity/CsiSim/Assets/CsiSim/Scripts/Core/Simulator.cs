// シミュレータ本体: 設定 → 時間を進めて全リンクの CSI と正解を作り、ファイルに書く。Unity に依存しない。
//
// 出力 (out で指定した名前に拡張子を付ける):
//   .json  メタ情報 (周波数、サブキャリア番号、送受信の位置、物体、関節名、パラメータ名、各配列の形)
//   .bin   float32 をこの順に連続で: H [T, L, K, 2] (実部, 虚部) / joints [T, 20, 3] / params [T, 19]
// ESP32 の欠点 (位相の乱れ・AGC・量子化・雑音・抜け) は入れない。検証側でわざと加える。
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;

namespace CsiSim.Core
{
    [Serializable] public class Vec3 { public float x, y, z; public Vector3 V => new Vector3(x, y, z); }

    [Serializable]
    public class ClutterSettings
    {
        public int count = 6;                           // 置く物体の数
        public float xmin = -3, xmax = 3, zmin = -3, zmax = 3;
        public float hmin = 0.3f, hmax = 1.8f;           // 物体の高さ
        public float rmin = 0.1f, rmax = 0.4f;           // 物体の太さ (半径)
        public float rho = 1.0f;                        // 反射の強さ (金属っぽいものは大きく)
        public int seed = 7;
    }

    [Serializable]
    public class SimConfig
    {
        public float fps = 100f;
        public int seed = 1;
        public Vec3[] tx = { new Vec3 { x = 0, y = 2.2f, z = 3.0f } };
        public Vec3[] rx = {
            new Vec3 { x = -2.8f, y = 1.2f, z = 2.5f }, new Vec3 { x = 2.8f, y = 1.2f, z = 2.5f },
            new Vec3 { x = -2.8f, y = 1.0f, z = -2.5f }, new Vec3 { x = 2.8f, y = 1.0f, z = -2.5f },
            new Vec3 { x = 0f, y = 2.4f, z = -3.0f } };
        public RfSettings rf = new RfSettings();
        public Humanoid.Body body = new Humanoid.Body();
        public ClutterSettings clutter = new ClutterSettings();
        public MotionSegment[] motion = {
            new MotionSegment { type = "tpose", dur = 3 }, new MotionSegment { type = "idle", dur = 5 },
            new MotionSegment { type = "walk", dur = 20 }, new MotionSegment { type = "arms", dur = 10 },
            new MotionSegment { type = "squat", dur = 8 }, new MotionSegment { type = "reach", dur = 6 },
            new MotionSegment { type = "random", dur = 30 } };
        public bool emptyRoom = false;   // true なら人を置かない (背景だけ)
        public string @out = "sim_out/default";
    }

    public class Simulator
    {
        public readonly SimConfig Cfg;
        public readonly int[] Tones;
        public readonly List<Capsule> Clutter;
        public readonly MotionPlayer Player;
        public int Links => Cfg.tx.Length * Cfg.rx.Length;

        public Simulator(SimConfig cfg)
        {
            Cfg = cfg; Tones = RfModel.Ht40Tones();
            Player = new MotionPlayer(cfg.motion, cfg.seed);
            Clutter = MakeClutter(cfg.clutter);
        }

        /// <summary>物体: 床に立つ円柱 (カプセル)。人と区別するため番号は 100 から。</summary>
        public static List<Capsule> MakeClutter(ClutterSettings c)
        {
            var rng = new Random(c.seed); var list = new List<Capsule>();
            for (int i = 0; i < c.count; i++)
            {
                float x = (float)(c.xmin + (c.xmax - c.xmin) * rng.NextDouble()), z = (float)(c.zmin + (c.zmax - c.zmin) * rng.NextDouble());
                float h = (float)(c.hmin + (c.hmax - c.hmin) * rng.NextDouble()), r = (float)(c.rmin + (c.rmax - c.rmin) * rng.NextDouble());
                list.Add(new Capsule(new Vector3(x, r, z), new Vector3(x, Math.Max(r, h - r), z), r, c.rho, 100 + i));
            }
            return list;
        }

        /// <summary>時刻 t の状態: 姿勢パラメータ、関節、全カプセル (人 + 物体)。</summary>
        public (double[] p, Vector3[] joints, List<Capsule> caps) State(double t)
        {
            var p = Player.Pose(t);
            var J = Humanoid.Joints(p, Cfg.body);
            var caps = new List<Capsule>(Clutter);
            if (!Cfg.emptyRoom) caps.AddRange(Humanoid.Capsules(J, Cfg.body, p));
            return (p, J, caps);
        }

        /// <summary>任意の姿勢パラメータ p でのカプセル (人 + 物体)。</summary>
        public List<Capsule> CapsulesAt(double[] p)
        {
            var caps = new List<Capsule>(Clutter);
            if (!Cfg.emptyRoom) caps.AddRange(Humanoid.Capsules(Humanoid.Joints(p, Cfg.body), Cfg.body, p));
            return caps;
        }

        /// <summary>
        /// 識別性の解析用: 姿勢 p での H と、各姿勢パラメータについての偏微分 dH/dp (中心差分)。
        /// 長さのパラメータ (root_x, root_z, pelvis_h) は eps_m [m]、角度は eps_rad [rad] だけ動かす。
        /// 出力: h0[l][k] (複素を実部・虚部で), jac[i][l][k]
        /// </summary>
        public void Jacobian(double[] p, double epsM, double epsRad, double[][] h0re, double[][] h0im, double[][][] jre, double[][][] jim)
        {
            Csi(CapsulesAt(p), h0re, h0im);
            int L = Links, K = Tones.Length;
            var pre = new double[L][]; var pim = new double[L][]; var mre = new double[L][]; var mim = new double[L][];
            for (int l = 0; l < L; l++) { pre[l] = new double[K]; pim[l] = new double[K]; mre[l] = new double[K]; mim[l] = new double[K]; }
            for (int i = 0; i < Humanoid.NP; i++)
            {
                double e = i < 3 ? epsM : epsRad;
                var pp = (double[])p.Clone(); pp[i] += e; Csi(CapsulesAt(pp), pre, pim);
                var pm = (double[])p.Clone(); pm[i] -= e; Csi(CapsulesAt(pm), mre, mim);
                for (int l = 0; l < L; l++)
                    for (int k = 0; k < K; k++) { jre[i][l][k] = (pre[l][k] - mre[l][k]) / (2 * e); jim[i][l][k] = (pim[l][k] - mim[l][k]) / (2 * e); }
            }
        }

        /// <summary>時刻 t の全リンクの CSI。H[l][k] の実部と虚部。</summary>
        public void Csi(List<Capsule> caps, double[][] re, double[][] im, List<RfModel.PathInfo>[] paths = null)
        {
            int l = 0;
            foreach (var t in Cfg.tx)
                foreach (var r in Cfg.rx)
                {
                    RfModel.Synthesize(t.V, r.V, caps, Cfg.rf, Tones, re[l], im[l], paths?[l]);
                    l++;
                }
        }

        /// <summary>全区間を生成してファイルに書く。progress は 0..1。</summary>
        public void Run(string baseDir, Action<double> progress = null)
        {
            string outBase = Path.IsPathRooted(Cfg.@out) ? Cfg.@out : Path.Combine(baseDir, Cfg.@out);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outBase)));
            int T = (int)Math.Round(Player.Total * Cfg.fps), L = Links, K = Tones.Length;
            var re = new double[L][]; var im = new double[L][];
            for (int l = 0; l < L; l++) { re[l] = new double[K]; im[l] = new double[K]; }
            var H = new float[T * L * K * 2]; var JJ = new float[T * Humanoid.NJ * 3]; var PP = new float[T * Humanoid.NP];
            for (int i = 0; i < T; i++)
            {
                var (p, J, caps) = State(i / Cfg.fps);
                Csi(caps, re, im);
                for (int l = 0; l < L; l++)
                    for (int k = 0; k < K; k++) { int o = ((i * L + l) * K + k) * 2; H[o] = (float)re[l][k]; H[o + 1] = (float)im[l][k]; }
                for (int j = 0; j < Humanoid.NJ; j++) { JJ[(i * Humanoid.NJ + j) * 3] = J[j].X; JJ[(i * Humanoid.NJ + j) * 3 + 1] = J[j].Y; JJ[(i * Humanoid.NJ + j) * 3 + 2] = J[j].Z; }
                for (int k = 0; k < Humanoid.NP; k++) PP[i * Humanoid.NP + k] = (float)p[k];
                if (progress != null && i % 200 == 0) progress((double)i / T);
            }
            using (var fs = new FileStream(outBase + ".bin", FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                foreach (var v in H) bw.Write(v); foreach (var v in JJ) bw.Write(v); foreach (var v in PP) bw.Write(v);
            }
            File.WriteAllText(outBase + ".json", Meta(T), new UTF8Encoding(false));
            progress?.Invoke(1.0);
        }

        string Meta(int T)
        {
            var ci = CultureInfo.InvariantCulture; var sb = new StringBuilder();
            string F(double v) => v.ToString("R", ci);
            string V(Vec3 v) => $"[{F(v.x)},{F(v.y)},{F(v.z)}]";
            string S(string[] a) => "[" + string.Join(",", Array.ConvertAll(a, x => "\"" + x + "\"")) + "]";
            sb.Append("{\n");
            sb.Append($" \"T\": {T}, \"L\": {Links}, \"K\": {Tones.Length}, \"NJ\": {Humanoid.NJ}, \"NP\": {Humanoid.NP}, \"fps\": {F(Cfg.fps)},\n");
            sb.Append($" \"fc\": {F(Cfg.rf.fc)}, \"df\": {F(Cfg.rf.df)}, \"tones\": [{string.Join(",", Tones)}],\n");
            sb.Append($" \"tx\": [{string.Join(",", Array.ConvertAll(Cfg.tx, V))}], \"rx\": [{string.Join(",", Array.ConvertAll(Cfg.rx, V))}],\n");
            sb.Append(" \"link_order\": \"tx-major (l = itx * n_rx + irx)\",\n");
            sb.Append($" \"clutter\": [{string.Join(",", Clutter.ConvertAll(c => $"{{\"a\":[{F(c.A.X)},{F(c.A.Y)},{F(c.A.Z)}],\"b\":[{F(c.B.X)},{F(c.B.Y)},{F(c.B.Z)}],\"r\":{F(c.R)},\"rho\":{F(c.Rho)}}}"))}],\n");
            sb.Append($" \"joint_names\": {S(Humanoid.JointNames)},\n \"param_names\": {S(Humanoid.ParamNames)},\n \"segment_names\": {S(Humanoid.SegmentNames)},\n");
            sb.Append($" \"empty_room\": {(Cfg.emptyRoom ? "true" : "false")}, \"floor\": {(Cfg.rf.floor ? "true" : "false")}, \"block_att\": {F(Cfg.rf.blockAtt)}, \"body_rho\": {F(Cfg.body.rho)},\n");
            sb.Append(" \"layout\": \"float32: H[T,L,K,2], joints[T,NJ,3], params[T,NP]\"\n}\n");
            return sb.ToString();
        }
    }
}
