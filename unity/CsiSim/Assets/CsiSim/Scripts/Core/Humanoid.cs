// 既定のアバター: 基本形状 (カプセル 15 個) で作る人体と、その順運動学。Unity に依存しない。
// 座標: y が上、体は yaw = 0 のとき +z を向く。Unity と同じ左手系なので、右手側が +x。
// 状態 (姿勢パラメータ) は 19 個。関節の位置は Kinect と同じ並びの 20 個を出す。
using System;
using System.Collections.Generic;
using System.Numerics;

namespace CsiSim.Core
{
    public static class Humanoid
    {
        public static readonly string[] ParamNames = {
            "root_x", "root_z", "pelvis_h", "yaw", "spine_pitch", "spine_roll", "head_yaw",
            "l_sh_pitch", "l_sh_roll", "l_elbow", "r_sh_pitch", "r_sh_roll", "r_elbow",
            "l_hip_pitch", "l_hip_roll", "l_knee", "r_hip_pitch", "r_hip_roll", "r_knee" };
        public const int NP = 19;

        public static readonly string[] JointNames = {
            "hip_center", "spine", "shoulder_center", "head",
            "shoulder_l", "elbow_l", "wrist_l", "hand_l",
            "shoulder_r", "elbow_r", "wrist_r", "hand_r",
            "hip_l", "knee_l", "ankle_l", "foot_l",
            "hip_r", "knee_r", "ankle_r", "foot_r" };
        public const int NJ = 20;

        public static readonly string[] SegmentNames = {
            "torso_lower", "torso_upper", "head",
            "upper_arm_l", "forearm_l", "hand_l", "upper_arm_r", "forearm_r", "hand_r",
            "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r" };
        public const int NS = 15;

        /// <summary>体の寸法 [m]。キャリブレーション (SLAM の地図側) で推定する対象。</summary>
        [Serializable]
        public class Body
        {
            public float torso = 0.45f, neck = 0.10f, head = 0.20f, shoulderW = 0.18f, shoulderH = 0.48f;
            public float upperArm = 0.30f, forearm = 0.27f, hand = 0.08f;
            public float hipW = 0.10f, thigh = 0.45f, shin = 0.43f, foot = 0.15f;
            public float rTorso = 0.14f, rChest = 0.15f, rHead = 0.10f, rUpperArm = 0.05f, rForearm = 0.04f, rHand = 0.035f;
            public float rThigh = 0.07f, rShin = 0.05f, rFoot = 0.04f;
            public float rho = 0.5f;   // 体の反射の強さ
        }

        /// <summary>T ポーズ (腕を横に広げて立つ) の姿勢パラメータ。</summary>
        public static double[] TPose(float x = 0, float z = 0)
        {
            var p = new double[NP]; p[0] = x; p[1] = z; p[2] = 0.98; p[8] = Math.PI / 2; p[11] = Math.PI / 2; return p;
        }

        static Quaternion Rx(double a) => Quaternion.CreateFromAxisAngle(Vector3.UnitX, (float)a);
        static Quaternion Ry(double a) => Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)a);
        static Quaternion Rz(double a) => Quaternion.CreateFromAxisAngle(Vector3.UnitZ, (float)a);
        static Vector3 Rot(Quaternion q, float x, float y, float z) => Vector3.Transform(new Vector3(x, y, z), q);

        /// <summary>順運動学: 姿勢パラメータ → 関節 20 個の位置。</summary>
        public static Vector3[] Joints(double[] p, Body b)
        {
            var J = new Vector3[NJ];
            var P = new Vector3((float)p[0], (float)p[2], (float)p[1]);
            var root = Ry(p[3]);
            var spine = root * Rx(p[4]) * Rz(p[5]);   // 前屈 (正) と横屈
            J[0] = P;
            J[1] = P + Rot(spine, 0, b.torso * 0.5f, 0);
            J[2] = P + Rot(spine, 0, b.shoulderH + b.neck * 0.5f, 0);
            J[3] = P + Rot(spine, 0, b.shoulderH + b.neck + b.head * 0.5f, 0.02f);
            for (int side = 0; side < 2; side++)   // 0 = 左 (-x), 1 = 右 (+x)
            {
                float sx = side == 0 ? -1 : 1; int o = side == 0 ? 7 : 10; int j = side == 0 ? 4 : 8;
                var sh = P + Rot(spine, sx * b.shoulderW, b.shoulderH, 0);
                var arm = spine * Rx(-p[o]) * Rz(sx * p[o + 1]);      // 前に上げる (pitch 正)、横に上げる (roll 正)
                var fore = arm * Rx(-p[o + 2]);                      // 肘を曲げる (正)
                J[j] = sh;
                J[j + 1] = sh + Rot(arm, 0, -b.upperArm, 0);
                J[j + 2] = J[j + 1] + Rot(fore, 0, -b.forearm, 0);
                J[j + 3] = J[j + 2] + Rot(fore, 0, -b.hand, 0);
            }
            for (int side = 0; side < 2; side++)
            {
                float sx = side == 0 ? -1 : 1; int o = side == 0 ? 13 : 16; int j = side == 0 ? 12 : 16;
                var hip = P + Rot(root, sx * b.hipW, -0.05f, 0);
                var thigh = root * Rx(-p[o]) * Rz(sx * p[o + 1]);     // 前に上げる (pitch 正)、外に開く (roll 正)
                var shin = thigh * Rx(p[o + 2]);                     // 膝を曲げる (正、後ろへ)
                J[j] = hip;
                J[j + 1] = hip + Rot(thigh, 0, -b.thigh, 0);
                J[j + 2] = J[j + 1] + Rot(shin, 0, -b.shin, 0);
                J[j + 3] = J[j + 2] + Rot(root, 0, -0.03f, b.foot);
            }
            return J;
        }

        /// <summary>関節の位置 → カプセル 15 個 (散乱体 + 遮蔽体)。番号は 0..14。</summary>
        public static List<Capsule> Capsules(Vector3[] J, Body b, double[] p)
        {
            var spine = Ry(p[3]) * Rx(p[4]) * Rz(p[5]);
            var neckBase = J[0] + Rot(spine, 0, b.shoulderH + b.neck, 0);
            var c = new List<Capsule>(NS);
            void Add(Vector3 a, Vector3 bb, float r) => c.Add(new Capsule(a, bb, r, b.rho, c.Count));
            Add(J[0], J[1], b.rTorso);
            Add(J[1], J[2], b.rChest);
            Add(neckBase, neckBase + Rot(spine, 0, b.head - 2 * b.rHead + 0.01f, 0.02f), b.rHead);
            for (int s = 0; s < 2; s++)
            {
                int j = s == 0 ? 4 : 8;
                Add(J[j], J[j + 1], b.rUpperArm); Add(J[j + 1], J[j + 2], b.rForearm); Add(J[j + 2], J[j + 3], b.rHand);
            }
            for (int s = 0; s < 2; s++)
            {
                int j = s == 0 ? 12 : 16;
                Add(J[j], J[j + 1], b.rThigh); Add(J[j + 1], J[j + 2], b.rShin); Add(J[j + 2], J[j + 3], b.rFoot);
            }
            return c;
        }
    }
}
