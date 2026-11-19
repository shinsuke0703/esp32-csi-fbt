// プレイモードでの表示: アバター (カプセル 15 個)、ストロボ (送信)、受信機、物体、電波の経路、選んだリンクの振幅。
// 計算はすべて CsiSim.Core (Unity に依存しないコア) で行い、ここは表示だけ。
using System.Collections.Generic;
using UnityEngine;
using CsiSim.Core;
using NV = System.Numerics.Vector3;

namespace CsiSim.UnityView
{
    public class CsiSimRuntime : MonoBehaviour
    {
        [Tooltip("Assets/CsiSim/Configs の設定 (JSON)")] public TextAsset config;
        [Tooltip("再生の速さ (1 = 実時間)")] public float timeScale = 1f;
        [Tooltip("振幅を表示するリンク")] public int showLink = 0;
        [Tooltip("経路を表示する (散乱は振幅の大きい順に上位 N 本)")] public bool showPaths = true;
        public int topScatterPaths = 20;

        Simulator sim; double t;
        readonly List<Transform> bodyGo = new List<Transform>();
        double[][] re, im; List<RfModel.PathInfo>[] paths;
        Texture2D plot; readonly List<float[]> history = new List<float[]>();
        Material lineMat;

        static Vector3 U(NV v) => new Vector3(v.X, v.Y, v.Z);

        void Start()
        {
            var cfg = config != null ? JsonUtility.FromJson<SimConfig>(config.text) : new SimConfig();
            sim = new Simulator(cfg);
            int L = sim.Links, K = sim.Tones.Length;
            re = new double[L][]; im = new double[L][]; paths = new List<RfModel.PathInfo>[L];
            for (int l = 0; l < L; l++) { re[l] = new double[K]; im[l] = new double[K]; paths[l] = new List<RfModel.PathInfo>(); }

            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane); floor.name = "Floor";
            floor.transform.localScale = new Vector3(1, 1, 1); Strip(floor, new Color(0.25f, 0.25f, 0.28f));
            foreach (var p in cfg.tx) Marker("Strobe (TX)", U(p.V), 0.12f, Color.yellow);
            for (int i = 0; i < cfg.rx.Length; i++) Marker($"RX {i}", U(cfg.rx[i].V), 0.10f, Color.cyan);
            foreach (var c in sim.Clutter) MakeCapsule("Object", c, new Color(0.6f, 0.45f, 0.3f));
            if (!cfg.emptyRoom)
                for (int s = 0; s < Humanoid.NS; s++)
                {
                    var go = GameObject.CreatePrimitive(PrimitiveType.Capsule); go.name = Humanoid.SegmentNames[s];
                    Strip(go, new Color(0.85f, 0.85f, 0.9f)); bodyGo.Add(go.transform);
                }
            plot = new Texture2D(K, 120) { filterMode = FilterMode.Point };
            lineMat = new Material(Shader.Find("Hidden/Internal-Colored"));
            lineMat.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
        }

        void Update()
        {
            if (sim == null) return;
            t = (t + Time.deltaTime * timeScale) % sim.Player.Total;
            var (p, J, caps) = sim.State(t);
            sim.Csi(caps, re, im, paths);
            int nh = 0;
            foreach (var c in caps)
            {
                if (c.Id >= 100 || nh >= bodyGo.Count) continue;   // 物体は動かない
                Place(bodyGo[nh++], c);
            }
            // 振幅の履歴 (選んだリンク)
            int l = Mathf.Clamp(showLink, 0, sim.Links - 1), K = sim.Tones.Length;
            var a = new float[K];
            for (int k = 0; k < K; k++) a[k] = (float)System.Math.Sqrt(re[l][k] * re[l][k] + im[l][k] * im[l][k]);
            history.Add(a); if (history.Count > plot.height) history.RemoveAt(0);
        }

        void OnRenderObject()   // 経路を線で描く (直接波 = 黄、床 = 緑、散乱 = 赤。濃さは振幅)
        {
            if (!showPaths || paths == null || lineMat == null) return;
            lineMat.SetPass(0); GL.Begin(GL.LINES);
            for (int l = 0; l < paths.Length; l++)
            {
                if (paths[l].Count == 0) continue;
                double aMax = paths[l][0].Amp;
                var sc = paths[l].FindAll(x => x.Kind == 2); sc.Sort((x, y) => y.Amp.CompareTo(x.Amp));
                foreach (var pi in paths[l])
                {
                    if (pi.Kind == 2 && sc.IndexOf(pi) >= topScatterPaths) continue;
                    float w = Mathf.Clamp01((float)(pi.Amp / aMax) * (pi.Kind == 2 ? 20f : 1f));
                    var col = pi.Kind == 0 ? new Color(1, 1, 0, 0.5f) : pi.Kind == 1 ? new Color(0, 1, 0, 0.4f) : new Color(1, 0.2f, 0.2f, w);
                    if (l != showLink && pi.Kind != 0) continue;   // 散乱と反射は選んだリンクだけ
                    GL.Color(col);
                    GL.Vertex(U(pi.P0)); GL.Vertex(U(pi.P1));
                    if (pi.Kind != 0) { GL.Vertex(U(pi.P1)); GL.Vertex(U(pi.P2)); }
                }
            }
            GL.End();
        }

        void OnGUI()   // 選んだリンクの振幅 (横 = サブキャリア、縦 = 時間、明るさ = 振幅)
        {
            if (history.Count == 0) return;
            float lo = float.MaxValue, hi = float.MinValue;
            foreach (var a in history) foreach (var v in a) { lo = Mathf.Min(lo, v); hi = Mathf.Max(hi, v); }
            for (int y = 0; y < plot.height; y++)
            {
                var a = y < history.Count ? history[history.Count - 1 - y] : null;
                for (int x = 0; x < plot.width; x++) { float v = a == null ? 0 : (a[x] - lo) / (hi - lo + 1e-12f); plot.SetPixel(x, plot.height - 1 - y, Color.HSVToRGB(0.66f * (1 - v), 1, a == null ? 0 : 1)); }
            }
            plot.Apply();
            GUI.Box(new Rect(10, 10, 360, 190), $"リンク {showLink} の振幅  t = {t:F1} s");
            GUI.DrawTexture(new Rect(20, 35, 340, 155), plot);
        }

        void Marker(string name, Vector3 pos, float r, Color col)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere); go.name = name; go.transform.position = pos; go.transform.localScale = Vector3.one * r * 2; Strip(go, col);
        }

        void MakeCapsule(string name, Capsule c, Color col)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Capsule); go.name = name; Strip(go, col); Place(go.transform, c);
        }

        static void Place(Transform tr, Capsule c)   // Unity のカプセル (高さ 2、半径 0.5) を線分 A-B と半径 R に合わせる
        {
            Vector3 a = U(c.A), b = U(c.B); float len = (b - a).magnitude;
            tr.position = (a + b) * 0.5f;
            tr.rotation = len > 1e-5f ? Quaternion.FromToRotation(Vector3.up, (b - a) / len) : Quaternion.identity;
            tr.localScale = new Vector3(2 * c.R, (len + 2 * c.R) * 0.5f, 2 * c.R);
        }

        static void Strip(GameObject go, Color col)   // 物理は使わないのでコライダーを外し、色だけ付ける
        {
            var col_ = go.GetComponent<Collider>(); if (col_ != null) Destroy(col_);
            var r = go.GetComponent<Renderer>(); r.material.color = col;
        }
    }
}
