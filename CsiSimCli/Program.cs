// 使い方:
//   データ生成:   dotnet run -c Release -- 設定.json [出力の基準フォルダ]
//   ヤコビアン:   dotnet run -c Release -- jacobian 設定.json 出力名 姿勢の数 [eps_m=0.005] [eps_rad=0.01]
//                 動きの全区間から等間隔に姿勢を取り、各姿勢で H と dH/dp (19 パラメータ) を出す。
//                 出力名.bin (float64): params[N,NP], H0[N,L,K,2], J[N,NP,L,K,2] / 出力名.json: 形と送受信の位置
using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using CsiSim.Core;

class Program
{
    static SimConfig Load(string path) => JsonSerializer.Deserialize<SimConfig>(File.ReadAllText(path),
        new JsonSerializerOptions { IncludeFields = true, PropertyNameCaseInsensitive = true, ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });

    static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "jacobian") return Jac(args);
        var cfg = args.Length > 0 ? Load(args[0]) : new SimConfig();
        string baseDir = args.Length > 1 ? args[1] : Directory.GetCurrentDirectory();
        var sim = new Simulator(cfg);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine($"リンク {sim.Links}、サブキャリア {sim.Tones.Length}、長さ {sim.Player.Total:F1} 秒 ({cfg.fps} Hz)、物体 {sim.Clutter.Count} 個、人 {(cfg.emptyRoom ? "なし" : "あり")}");
        sim.Run(baseDir, p => Console.Write($"\r{p * 100:F0}%"));
        Console.WriteLine($"\n完了 {sw.Elapsed.TotalSeconds:F1} 秒 → {Path.GetFullPath(Path.Combine(baseDir, cfg.@out))}.bin / .json");
        return 0;
    }

    static int Jac(string[] a)
    {
        var cfg = Load(a[1]); string outBase = a[2]; int N = int.Parse(a[3]);
        double epsM = a.Length > 4 ? double.Parse(a[4], CultureInfo.InvariantCulture) : 0.005;
        double epsR = a.Length > 5 ? double.Parse(a[5], CultureInfo.InvariantCulture) : 0.01;
        var sim = new Simulator(cfg); int L = sim.Links, K = sim.Tones.Length, NP = Humanoid.NP;
        var h0re = Arr(L, K); var h0im = Arr(L, K);
        var jre = new double[NP][][]; var jim = new double[NP][][];
        for (int i = 0; i < NP; i++) { jre[i] = Arr(L, K); jim[i] = Arr(L, K); }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outBase)));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using (var bw = new BinaryWriter(new FileStream(outBase + ".bin", FileMode.Create)))
        {
            var P = new double[N][];
            for (int n = 0; n < N; n++) P[n] = sim.Player.Pose((n + 0.5) * sim.Player.Total / N);
            foreach (var p in P) foreach (var v in p) bw.Write(v);
            var Hbuf = new double[N * L * K * 2]; var Jbuf = new double[N * NP * L * K * 2];
            for (int n = 0; n < N; n++)
            {
                sim.Jacobian(P[n], epsM, epsR, h0re, h0im, jre, jim);
                for (int l = 0; l < L; l++) for (int k = 0; k < K; k++) { int o = ((n * L + l) * K + k) * 2; Hbuf[o] = h0re[l][k]; Hbuf[o + 1] = h0im[l][k]; }
                for (int i = 0; i < NP; i++) for (int l = 0; l < L; l++) for (int k = 0; k < K; k++) { int o = (((n * NP + i) * L + l) * K + k) * 2; Jbuf[o] = jre[i][l][k]; Jbuf[o + 1] = jim[i][l][k]; }
                Console.Write($"\r{n + 1}/{N}");
            }
            foreach (var v in Hbuf) bw.Write(v); foreach (var v in Jbuf) bw.Write(v);
        }
        var ci = CultureInfo.InvariantCulture; string F(double v) => v.ToString("R", ci);
        string V(Vec3 v) => $"[{F(v.x)},{F(v.y)},{F(v.z)}]";
        File.WriteAllText(outBase + ".json",
            $"{{\"N\":{N},\"L\":{L},\"K\":{K},\"NP\":{NP},\"eps_m\":{F(epsM)},\"eps_rad\":{F(epsR)},\"fc\":{F(cfg.rf.fc)},\"df\":{F(cfg.rf.df)},\"tones\":[{string.Join(",", sim.Tones)}]," +
            $"\"tx\":[{string.Join(",", Array.ConvertAll(cfg.tx, V))}],\"rx\":[{string.Join(",", Array.ConvertAll(cfg.rx, V))}]," +
            $"\"param_names\":[{string.Join(",", Array.ConvertAll(Humanoid.ParamNames, s => "\"" + s + "\""))}],\"layout\":\"float64: params[N,NP], H0[N,L,K,2], J[N,NP,L,K,2]\"}}",
            new UTF8Encoding(false));
        Console.WriteLine($"\n完了 {sw.Elapsed.TotalSeconds:F1} 秒 → {outBase}.bin");
        return 0;
    }

    static double[][] Arr(int a, int b) { var r = new double[a][]; for (int i = 0; i < a; i++) r[i] = new double[b]; return r; }
}
