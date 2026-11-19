// 実機で収録した CSI (ACK 往復) の再生。replay_export.py が書き出した JSON を読み、
// ノードを球、15 本のリンクを線で表示し、各リンクの「体の成分の強さ」|H_dyn|²/|H_bg|² を色と太さで時間とともに見せる。
// 色: 青 (-20 dB 以下、体の影響なし) → 赤 (0 dB 以上、体の影響が背景と同じくらい)。
using System;
using UnityEngine;

namespace CsiSim.UnityView
{
    public class CsiReplay : MonoBehaviour
    {
        [Serializable] public class V3 { public float x, y, z; }
        [Serializable] public class Link { public int a, b; }
        [Serializable] public class Frame { public float t; public float[] v; }
        [Serializable] public class Seg { public string name; public float t0, t1; }
        [Serializable] public class Doc { public string source, layout; public float fps; public V3[] nodes; public Link[] links; public Seg[] segments; public Frame[] frames; }

        [Tooltip("Assets/CsiSim/Replays の JSON")] public TextAsset replay;
        [Tooltip("再生の速さ (1 = 実時間)")] public float speed = 1f;
        [Tooltip("開始時刻 [秒]")] public float startTime = 0f;
        public float dbMin = -20f, dbMax = 0f;

        Doc doc; float t; LineRenderer[] lines; Material lineMat;

        void Start()
        {
            doc = JsonUtility.FromJson<Doc>(replay.text);
            t = startTime;
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane); floor.name = "Floor"; Destroy(floor.GetComponent<Collider>());
            floor.GetComponent<Renderer>().material.color = new Color(0.22f, 0.22f, 0.25f);
            for (int i = 0; i < doc.nodes.Length; i++)
            {
                var n = GameObject.CreatePrimitive(PrimitiveType.Sphere); n.name = $"Node {i}"; Destroy(n.GetComponent<Collider>());
                n.transform.position = P(doc.nodes[i]); n.transform.localScale = Vector3.one * 0.18f;
                n.GetComponent<Renderer>().material.color = i == 0 ? new Color(1f, 0.85f, 0.2f) : new Color(0.3f, 0.9f, 1f);
            }
            lineMat = new Material(Shader.Find("Sprites/Default"));
            lines = new LineRenderer[doc.links.Length];
            for (int l = 0; l < doc.links.Length; l++)
            {
                var go = new GameObject($"Link {doc.links[l].a}-{doc.links[l].b}");
                var lr = go.AddComponent<LineRenderer>(); lr.material = lineMat; lr.positionCount = 2;
                lr.SetPosition(0, P(doc.nodes[doc.links[l].a])); lr.SetPosition(1, P(doc.nodes[doc.links[l].b]));
                lines[l] = lr;
            }
        }

        static Vector3 P(V3 v) => new Vector3(v.x, v.y, v.z);

        void Update()
        {
            if (doc == null || doc.frames.Length == 0) return;
            t += Time.deltaTime * speed;
            float T = doc.frames[doc.frames.Length - 1].t; if (t > T) t = 0;
            int i = Mathf.Clamp(Mathf.RoundToInt(t * doc.fps), 0, doc.frames.Length - 1);
            var f = doc.frames[i];
            for (int l = 0; l < lines.Length; l++)
            {
                float u = Mathf.InverseLerp(dbMin, dbMax, f.v[l]);
                var c = Color.Lerp(new Color(0.2f, 0.35f, 1f, 0.35f), new Color(1f, 0.15f, 0.1f, 1f), u);
                lines[l].startColor = lines[l].endColor = c;
                lines[l].startWidth = lines[l].endWidth = Mathf.Lerp(0.01f, 0.09f, u);
            }
        }

        void OnGUI()
        {
            if (doc == null) return;
            string seg = "";
            foreach (var s in doc.segments) if (t >= s.t0 && t < s.t1) seg = s.name;
            var st = new GUIStyle(GUI.skin.box) { fontSize = 16, alignment = TextAnchor.UpperLeft };
            GUI.Box(new Rect(10, 10, 430, 86), $"実機 CSI の再生  t = {t:F1} 秒\n区間: {seg}\n線の色と太さ = 体の成分の強さ (青 {dbMin} dB → 赤 {dbMax} dB)\nノードの位置: {doc.layout}", st);
        }
    }
}
