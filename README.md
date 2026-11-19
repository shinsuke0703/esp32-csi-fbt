# ESP32 CSI 全身トラッキング（研究中）

Camera-free, device-free full-body tracking research using the 2.4 GHz Wi-Fi CSI of six off-the-shelf ESP32 boards. Work in progress; notes are in Japanese.

市販の ESP32 6 台の 2.4 GHz Wi-Fi CSI で、体に何も付けない全身トラッキング（60 Hz）を目指す研究のコードです。送信も受信も部屋に固定する Lighthouse 型の構成です。

## 現状（2026-11-19）

できていること:

- **ACK 往復で位相をそろえる**：ESP-NOW のユニキャストに対してハードウェアが返す ACK の CSI を拾います。往復の積 `P_ij = Y_ij · Y_ji` で、2 台の発振器の位相のずれが消えます。時計の共有はしていません。15 組すべてで R 0.87〜0.98 です。静止時の経路長は ±1 cm 以内で、ドリフトはありません。
- **全 15 リンクを 60 Hz で取得**：PC が 16.7 ms ごとに USB で全台へ `G` を送ります。各台は自分の時間枠で送信します（`live/slot_run.py`）。
- **位相のぶれの特定**：パケットごとの白色のぶれです。DevKitC の組は 0.10〜0.25 rad、C3 を含む組は約 0.3 rad です。AGC、送受信の向き、周波数のずれ × タイミング、パケット長は原因ではありませんでした。
- **Kinect v1 の骨格を正解データとして同時収録**：時刻の遅れは約 0.22 s です。CSI の動きの強さと Kinect の関節の速さの相関は約 0.8 です。
- **Unity 製の CSI シミュレータ**（CLI 版あり）：体の散乱の強さを実機に合わせて補正してあります。

できていないこと:

- **ノード位置の自動校正**：体の反射だけからは位置が決まりませんでした。RSSI の弱まりの当てはめでは、解が一つに決まりません。ドップラーの一致度は 0.522 で、ランダムな配置の 0.519 と変わりません。体の反射は直接波より約 10 dB 弱く、パケットごとの位相のぶれに埋もれるためです。次は、ノードを手に持って歩き、Kinect で手を追って直接波で校正する予定です。
- **全身の推定**：まだです。シミュレーションでは、雑音がなければ歩行の追跡誤差は 6.5 cm です。実機の雑音を乗せると 33〜50 cm です。

詳しい経過と数値は [live/pingpong_results.md](live/pingpong_results.md) と [sim/identifiability_results.md](sim/identifiability_results.md) にあります。

## 構成

| フォルダ | 中身 |
| --- | --- |
| `firmware/esp32_pingpong` | 本命のファームウェア（PlatformIO、Arduino）。ACK 往復、トークン、USB の時間枠の方式。`flash_all.sh` でつながっている全台に書き込みます |
| `firmware/esp32_rx`、`esp32_strobe`、`esp32s3_scanner`、`c3_hello` | 初期の受信機、ストロボ（送信機）、動作確認用 |
| `live` | PC 側の収録と解析（Python）。`slot_run.py`（60 Hz 収録）、`rec_both.py`（CSI と Kinect の同時収録）、`sync_check.py`（時刻合わせ）、`calib_*.py`（校正の試み）、ぶれの解析など |
| `sim` | シミュレーションでの識別性の検証、粒子フィルタ、校正の試み |
| `kinect` | Kinect v1 の骨格の記録ツール（`KinectLog.cs`、Kinect SDK 1.8）とカバー範囲の確認 |
| `unity/CsiSim` | Unity 2022.3 の CSI シミュレータ |
| `CsiSimCli` | シミュレータの CLI 版（.NET） |

## 機材

- ESP32 DevKitC × 5（ノード 1〜5、主力）と ESP32-C3 × 1（ノード 0、補助）。全台を USB で PC につなぎます。
- C3 の送信出力は 11 dBm 以下にします。C3 はすべてのリンクで RSSI が −75 dBm 以上になる場所に置きます。弱いと送り直しで時間枠があふれ、連鎖的に抜けます。
- Kinect v1（Xbox 360、Kinect for Windows SDK 1.8）。
- Windows（スクリプトは Windows のタイマー精度の設定を前提にしています）。

## 使い方（概略）

```powershell
cd firmware/esp32_pingpong; sh flash_all.sh            # 全台に書き込む
python live/slot_run.py 60 out.npz                      # 60 秒、60 Hz で収録
python live/rec_both.py 120 calib1                      # CSI と Kinect を同時に収録
python live/sync_check.py live/sessions/calib1          # Kinect の時刻の遅れを測る
```

ノードの MAC アドレスは `firmware/esp32_pingpong/src/main.cpp` の `NODE_MAC` に直書きしてあります。自分のボードに合わせて書き換えてください。

## 含まれていないもの

収録データ（CSI、Kinect の骨格、深度画像）とシミュレーションの出力は含めていません。スクリプト内のパスには、作者の環境（`E:/CSI/...`）が残っています。

## ライセンス

[MIT](LICENSE)
