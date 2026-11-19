// エディタのメニュー: データの生成と、表示用シーンの作成。
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using CsiSim.Core;
using CsiSim.UnityView;

namespace CsiSim.EditorTools
{
    public static class CsiSimMenu
    {
        const string ConfigDir = "Assets/CsiSim/Configs";

        [MenuItem("CsiSim/データを生成 (選択中の設定 JSON)")]
        public static void GenerateSelected()
        {
            var ta = Selection.activeObject as TextAsset;
            string path = ta != null ? AssetDatabase.GetAssetPath(ta) : $"{ConfigDir}/default.json";
            Generate(path);
        }

        /// <summary>設定ファイルからデータを作る (MCP やバッチから呼べるように public)。</summary>
        public static string Generate(string configAssetPath)
        {
            var cfg = JsonUtility.FromJson<SimConfig>(File.ReadAllText(configAssetPath));
            var sim = new Simulator(cfg);
            string baseDir = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            try
            {
                sim.Run(baseDir, p => EditorUtility.DisplayProgressBar("CSI シミュレータ", $"{configAssetPath}  {p * 100:F0}%", (float)p));
            }
            finally { EditorUtility.ClearProgressBar(); }
            string outPath = Path.IsPathRooted(cfg.@out) ? cfg.@out : Path.Combine(baseDir, cfg.@out);
            Debug.Log($"[CsiSim] 生成しました: {outPath}.bin / .json  (リンク {sim.Links}, {sim.Player.Total:F1} 秒, {cfg.fps} Hz)");
            return outPath;
        }

        [MenuItem("CsiSim/表示用シーンを作る")]
        public static void CreateViewerScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            var cam = Camera.main;
            cam.transform.position = new Vector3(0, 6.5f, -7.5f); cam.transform.rotation = Quaternion.Euler(40, 0, 0);
            cam.backgroundColor = new Color(0.08f, 0.08f, 0.1f); cam.clearFlags = CameraClearFlags.SolidColor;
            var go = new GameObject("CsiSimulator");
            var rt = go.AddComponent<CsiSimRuntime>();
            rt.config = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ConfigDir}/default.json");
            Directory.CreateDirectory("Assets/CsiSim/Scenes");
            EditorSceneManager.SaveScene(scene, "Assets/CsiSim/Scenes/CsiSimViewer.unity");
            Debug.Log("[CsiSim] Assets/CsiSim/Scenes/CsiSimViewer.unity を作りました。再生すると動きます。");
        }
    }
}
