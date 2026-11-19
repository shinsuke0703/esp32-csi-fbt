// Kinect v1 (Xbox 360 版、SDK 1.8) の骨格を CSV に記録する。CSI の収録と時刻 (UNIX 秒、PC の時計) でそろえるための正解データ用。
// 使い方: KinectLog.exe 秒数 出力.csv [首振り角度 (度、-27〜27、省略で変えない)]
// 出力: 1 行 = 1 フレーム × 追跡中の 1 人。列: unix 秒, フレーム番号, 人の ID, 追跡の状態, 20 関節 × (x, y, z, 状態 0 なし / 1 推定 / 2 追跡)
// 座標は Kinect の座標系 (m): x 左右、y 上下、z 奥行き (Kinect からの距離)。
// 画面には 1 秒ごとに「骨格が取れたフレームの割合」「腰の距離」「推定に頼った関節の数」を出す。
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Kinect;

class KinectLog
{
    static void Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        double dur = args.Length > 0 ? double.Parse(args[0]) : 30;
        string outPath = args.Length > 1 ? args[1] : "kinect.csv";
        var sensor = KinectSensor.KinectSensors.FirstOrDefault(s => s.Status == KinectStatus.Connected);
        if (sensor == null) { Console.WriteLine("Kinect が見つからない"); return; }
        sensor.SkeletonStream.Enable(new TransformSmoothParameters { Smoothing = 0.0f, Correction = 0.0f, Prediction = 0.0f, JitterRadius = 0.0f, MaxDeviationRadius = 0.0f });   // 平滑化なし (生の値)
        sensor.SkeletonStream.TrackingMode = SkeletonTrackingMode.Default;
        sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
        sensor.Start();
        if (args.Length > 2) { try { sensor.ElevationAngle = int.Parse(args[2]); } catch (Exception e) { Console.WriteLine("首振りに失敗: " + e.Message); } }
        Console.WriteLine("Kinect 開始。首振り角度 " + sensor.ElevationAngle + " 度、記録 " + dur + " 秒 → " + outPath);
        var w = new StreamWriter(outPath, false, new UTF8Encoding(false));
        var hdr = new StringBuilder("t_unix,frame,id,state");
        foreach (JointType j in Enum.GetValues(typeof(JointType))) hdr.Append("," + j + "_x," + j + "_y," + j + "_z," + j + "_s");
        w.WriteLine(hdr.ToString());
        var sw = Stopwatch.StartNew(); var t0 = DateTime.UtcNow; var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        int nFrames = 0, nTracked = 0, nInfer = 0; double lastPrint = 0, zSum = 0; int zN = 0;
        Skeleton[] sk = new Skeleton[6];
        object lk = new object();
        short[] lastDepth = null; int dW = 640, dH = 480;
        sensor.DepthFrameReady += (s2, e2) => { using (var df = e2.OpenDepthImageFrame()) { if (df == null) return; var px = new DepthImagePixel[df.PixelDataLength]; df.CopyDepthImagePixelDataTo(px);
            var d = new short[px.Length]; for (int i = 0; i < px.Length; i++) d[i] = px[i].Depth; lastDepth = d; } };
        sensor.SkeletonFrameReady += (s, e) =>
        {
            using (var f = e.OpenSkeletonFrame())
            {
                if (f == null) return;
                double tu = (t0 - epoch).TotalSeconds + sw.Elapsed.TotalSeconds;
                f.CopySkeletonDataTo(sk);
                lock (lk)
                {
                    nFrames++; bool any = false;
                    foreach (var b in sk)
                    {
                        if (b == null || b.TrackingState == SkeletonTrackingState.NotTracked) continue;
                        var sb = new StringBuilder();
                        sb.Append(tu.ToString("F4") + "," + f.FrameNumber + "," + b.TrackingId + "," + (int)b.TrackingState);
                        foreach (Joint j in b.Joints) sb.Append("," + j.Position.X.ToString("F4") + "," + j.Position.Y.ToString("F4") + "," + j.Position.Z.ToString("F4") + "," + (int)j.TrackingState);
                        w.WriteLine(sb.ToString());
                        if (b.TrackingState == SkeletonTrackingState.Tracked)
                        {
                            any = true; zSum += b.Joints[JointType.HipCenter].Position.Z; zN++;
                            nInfer += b.Joints.Count(j => j.TrackingState != JointTrackingState.Tracked);
                        }
                    }
                    if (any) nTracked++;
                }
            }
        };
        while (sw.Elapsed.TotalSeconds < dur)
        {
            System.Threading.Thread.Sleep(50);
            if (sw.Elapsed.TotalSeconds - lastPrint >= 1.0)
            {
                lock (lk)
                {
                    Console.WriteLine(string.Format("{0,5:F0} 秒: 骨格あり {1}/{2} フレーム、腰の距離 {3}、推定に頼った関節 平均 {4:F1}/20",
                        sw.Elapsed.TotalSeconds, nTracked, nFrames, zN > 0 ? (zSum / zN).ToString("F2") + " m" : "-", nTracked > 0 ? (double)nInfer / nTracked : 0));
                    nFrames = nTracked = nInfer = 0; zSum = 0; zN = 0;
                }
                lastPrint = sw.Elapsed.TotalSeconds;
            }
        }
        sensor.Stop(); w.Close();
        if (lastDepth != null) {   // 最後の深度画像 (mm、0 は測れない所) を 16 ビットの PGM で保存
            var pgm = Path.ChangeExtension(outPath, ".pgm");
            using (var fs = new FileStream(pgm, FileMode.Create)) { var h = Encoding.ASCII.GetBytes("P5\n" + dW + " " + dH + "\n65535\n"); fs.Write(h, 0, h.Length);
                foreach (var v in lastDepth) { fs.WriteByte((byte)(v >> 8)); fs.WriteByte((byte)(v & 0xFF)); } }
            Console.WriteLine("深度画像 → " + pgm);
        }
        Console.WriteLine("終了");
    }
}
