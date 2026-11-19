// 往復 (PING/PONG) 実験用ファームウェア (ESP32 classic / ESP32-C3 共通)
//
// 目的: ESP32 の CSI の「パケットごとにランダムな共通位相」が、往復 (i→j と j→i) の積で打ち消せるかを確かめる。
// 動き:
//   - PC からシリアルで "P <j>\n" を受けたら、相手 j あての PING を ESP-NOW ブロードキャストで送る。
//   - 自分あての PING を受けたら、すぐに PONG を送り返す (受信コールバックからキュー経由で、優先度の高いタスクが送る)。
//   - 6 台のどれかから届いたすべてのフレーム (PING / PONG) の CSI を、ペイロードの情報と受信時刻を付けて PC に送る。
// 無線: ch6、HT20、ESP-NOW を MCS0 に固定 (OFDM でないと CSI が出ないため)。C3 は送信出力 8.5 dBm (高いと電波が出ない)。
//
// PC へのレコード (リトルエンディアン、合計 284 バイト):
//   u32 magic = 0xB0B0C510 / u8 rx_id / u8 tx_id / u8 type (1 PING, 2 PONG) / u8 target / u32 seq (送信元が振る番号)
//   u32 t_rx_us (受信側の時計) / u32 t_aux_us (PONG なら、PING を受けてから PONG を送るまでの時間、PING なら 0)
//   i8 rssi / i8 noise / u8 sig_mode / u8 cwb / u16 csi_len / u8 first_word_invalid / u8 chip (0 classic, 1 C3)
//   i8 csi[256] (受信した CSI の先頭 256 バイト。classic は FFT 順、C3 は周波数順)
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#define CHANNEL 6
static const uint8_t NODE_MAC[6][6] = {
    {0x44, 0x7B, 0x30, 0x4C, 0xAF, 0x98},   // 0: ESP32-C3 (COM6)
    {0x24, 0x6F, 0x28, 0x84, 0x90, 0x48},   // 1: DevKitC (COM3)
    {0xCC, 0xDB, 0xA7, 0x32, 0x13, 0x80},   // 2: DevKitC (COM7)
    {0xFC, 0xE8, 0xC0, 0x7B, 0xB9, 0xA8},   // 3: DevKitC (COM8)
    {0xFC, 0xE8, 0xC0, 0x7C, 0x8A, 0xD0},   // 4: DevKitC (COM9)
    {0xCC, 0xDB, 0xA7, 0x32, 0x2B, 0x64}};  // 5: DevKitC (COM10)
static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

#pragma pack(push, 1)
struct Payload { uint8_t m0, m1, type, sender, target; uint32_t seq; };
struct Record {
    uint32_t magic; uint8_t rx_id, tx_id, type, target; uint32_t seq, t_rx_us, t_aux_us;
    int8_t rssi, noise; uint8_t sig_mode, cwb; uint16_t csi_len; uint8_t fwi, chip; int8_t csi[256];
};
#pragma pack(pop)

static int g_id = -1;
static uint32_t g_seq = 0;
static Record g_pending; static volatile bool g_pending_valid = false;
static QueueHandle_t g_out_q, g_pong_q;
struct PongJob { uint32_t seq; uint8_t to; uint32_t t_rx; };
static volatile uint32_t g_n_csi = 0, g_n_rec = 0, g_n_pong = 0, g_drop = 0;

static int id_of(const uint8_t *mac) {
    for (int i = 0; i < 6; i++) if (memcmp(mac, NODE_MAC[i], 6) == 0) return i;
    return -1;
}

// ACK 実験: ユニキャストの PING を送ってから少しの間に届いた「送信元が 6 台のどれでもない」CSI を、相手が返した ACK とみなす
// (ACK には送信元アドレスがない)。type 5 で記録する。
static volatile uint32_t g_ack_wait_until = 0; static volatile uint8_t g_ack_from = 0; static volatile uint32_t g_ack_seq = 0, g_ack_sent_us = 0;
static volatile uint32_t g_n_ack = 0, g_n_unknown = 0;
static volatile uint32_t g_ack_window_us = 5000;

static void fill_common(Record &r, wifi_csi_info_t *info) {
    r.magic = 0xB0B0C510; r.rx_id = (uint8_t)g_id; r.t_rx_us = (uint32_t)esp_timer_get_time();
    r.rssi = info->rx_ctrl.rssi; r.noise = info->rx_ctrl.noise_floor; r.sig_mode = info->rx_ctrl.sig_mode; r.cwb = info->rx_ctrl.channel;
    r.csi_len = info->len; r.fwi = info->first_word_invalid;
#if CONFIG_IDF_TARGET_ESP32C3
    r.chip = 1;
#else
    r.chip = 0;
#endif
    memcpy(r.csi, info->buf, info->len < 256 ? info->len : 256);
}

// ACK の取り違えをなくす: プロミスキャスのコールバックで「自分あての ACK」(宛先 = 自分の MAC) を見つけ、
// その受信タイムスタンプ (rx_ctrl.timestamp) と同じタイムスタンプの CSI だけを ACK として記録する。
// 2 つのコールバックの順番は決まっていないので、どちらが先でも突き合わせられるよう、両方を小さなリングに置く。
static uint8_t g_my_mac[6];
#define RING 32
static volatile uint32_t g_ackts[RING]; static volatile uint8_t g_ackts_i = 0;                 // 自分あての ACK のタイムスタンプ
static Record g_cand[RING]; static uint32_t g_cand_ts[RING]; static volatile uint8_t g_cand_i = 0; // 送信元のない CSI (ACK 候補)
static volatile uint32_t g_n_ack_me = 0, g_n_match = 0;

// パイプライン (自律モード): 相手 5 台に続けて送り、どの ACK がどの相手のものかは後で「送信完了のコールバック」の順番と時刻で対応づける
static volatile bool g_pipe = false;
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
struct SendEv { uint8_t t; uint32_t seq, tau; bool ok; };
struct AckC { uint32_t ts; Record r; };
#define PMAX 32
static SendEv g_ev[PMAX]; static volatile int g_n_ev = 0;
static AckC g_acks[PMAX]; static volatile int g_n_acks = 0;
static uint32_t g_tseq[6];
// 速い方式 (ホッピングなし): PING に続けてすぐトークンを送る。送信完了のコールバックはまとめて遅れて届くので待たない。
// コールバックは送った順に届くので、i 番目のコールバック = i 番目に送ったもの (最後の 1 つがトークン)。
static volatile bool g_fast = false; static volatile int g_cb_n = 0, g_nsend = 0; static uint8_t g_st[PMAX]; static uint32_t g_sseq[PMAX];
static volatile uint32_t g_n_nomem = 0, g_n_senderr = 0;
static void send_fc(const uint8_t *mac, const void *d, int len) {   // 断られたら最大 5 ms まで待って送り直す
    uint32_t t0 = (uint32_t)esp_timer_get_time();
    while (true) {
        esp_err_t e = esp_now_send(mac, (const uint8_t *)d, len);
        if (e == ESP_OK) return;
        if (e != ESP_ERR_ESPNOW_NO_MEM || (uint32_t)esp_timer_get_time() - t0 > 5000) { g_n_senderr++; return; }
        g_n_nomem++; delayMicroseconds(50);
    }
}
static volatile int g_reps = 1;   // 1 回の手番で同じ相手に何回送るか ("N <回数>" で変更)
static volatile uint32_t g_n_pipe_match = 0, g_n_pipe_unmatched = 0;

static void emit_ack(Record &r, uint32_t now, uint32_t ts) {
    if (g_pipe) {   // 後で突き合わせる
        portENTER_CRITICAL(&g_mux);
        if (g_n_acks < PMAX) { g_acks[g_n_acks].ts = ts; g_acks[g_n_acks].r = r; g_n_acks++; }
        portEXIT_CRITICAL(&g_mux);
        g_n_ack++; return;
    }
    r.tx_id = g_ack_from; r.type = 5; r.target = g_ack_from; r.seq = g_ack_seq; r.t_aux_us = now - g_ack_sent_us;
    if (xQueueSend(g_out_q, &r, 0) != pdTRUE) g_drop++;
    g_n_ack++; g_n_match++; g_ack_wait_until = 0;
}

static volatile bool g_ev_on = false; static volatile int32_t g_off = 0; static volatile bool g_off_valid = false;
static volatile int g_tok_to = -1; static volatile bool g_tok_sent = false; static volatile uint32_t g_n_tok_fail = 0;
static void on_sent(const uint8_t *mac, esp_now_send_status_t st) {   // ユニキャストは ACK を受けた (または諦めた) 直後に呼ばれる
    if (g_fast) {
        int i = g_cb_n++;
        if (i < g_nsend) { portENTER_CRITICAL(&g_mux); if (g_n_ev < PMAX) { g_ev[g_n_ev] = {g_st[i], g_sseq[i], (uint32_t)esp_timer_get_time(), st == ESP_NOW_SEND_SUCCESS}; g_n_ev++; } portEXIT_CRITICAL(&g_mux); }
        else if (i == g_nsend) { g_tok_sent = true; if (st != ESP_NOW_SEND_SUCCESS) g_n_tok_fail++; }
        return;
    }
    // トークンの送信完了 (チャネルを切り替える前に、送り直しが終わるのを待つため)
    if (g_tok_to >= 0 && id_of(mac) == g_tok_to && !g_ev_on) { g_tok_sent = true; if (st != ESP_NOW_SEND_SUCCESS) g_n_tok_fail++; return; }
    if (!g_pipe || !g_ev_on) return;
    int t = id_of(mac); if (t < 0) return;
    portENTER_CRITICAL(&g_mux);
    if (g_n_ev < PMAX) { g_ev[g_n_ev] = {(uint8_t)t, g_tseq[t], (uint32_t)esp_timer_get_time(), st == ESP_NOW_SEND_SUCCESS}; g_n_ev++; }
    portEXIT_CRITICAL(&g_mux);
}

static void promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
    if (type != WIFI_PKT_CTRL) return;
    const wifi_promiscuous_pkt_t *p = (const wifi_promiscuous_pkt_t *)buf;
    const uint8_t *f = p->payload;
    if ((f[0] & 0xFC) != 0xD4 || memcmp(f + 4, g_my_mac, 6) != 0) return;   // ACK (フレーム制御 0xD4) で、宛先が自分
    g_n_ack_me++;
    uint32_t ts = p->rx_ctrl.timestamp;
    if (!g_ack_wait_until) return;
    for (int i = 0; i < RING; i++) if (g_cand_ts[i] == ts && g_cand_ts[i]) { g_cand_ts[i] = 0; emit_ack(g_cand[i], (uint32_t)esp_timer_get_time(), ts); return; }
    g_ackts[g_ackts_i] = ts; g_ackts_i = (g_ackts_i + 1) % RING;   // CSI がまだなら覚えておく
}

static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len < 64) return;
    int tx = id_of(info->mac);
    if (tx < 0) {
        g_n_unknown++;
        if (!g_ack_wait_until) return;
        // ACK はレガシー OFDM で、CSI は LLTF だけ (長さ 128)。それ以外は候補にしない (周りのフレームで候補のリングがあふれないように)
        if (info->rx_ctrl.sig_mode != 0 || info->len != 128) return;
        uint32_t now = (uint32_t)esp_timer_get_time(), ts = info->rx_ctrl.timestamp;
        Record r; memset(&r, 0, sizeof(r)); fill_common(r, info);
        for (int i = 0; i < RING; i++) if (g_ackts[i] == ts && g_ackts[i]) { g_ackts[i] = 0; emit_ack(r, now, ts); return; }
        g_cand[g_cand_i] = r; g_cand_ts[g_cand_i] = ts; g_cand_i = (g_cand_i + 1) % RING;   // ACK の判定がまだなら覚えておく
        return;
    }
    if (info->len < 128) return;
    g_n_csi++;
    Record &r = g_pending; memset(&r, 0, sizeof(r));
    r.magic = 0xB0B0C510; r.rx_id = (uint8_t)g_id; r.tx_id = (uint8_t)tx;
    r.t_rx_us = (uint32_t)esp_timer_get_time();
    r.rssi = info->rx_ctrl.rssi; r.noise = info->rx_ctrl.noise_floor; r.sig_mode = info->rx_ctrl.sig_mode; r.cwb = info->rx_ctrl.channel;
    r.csi_len = info->len; r.fwi = info->first_word_invalid;
#if CONFIG_IDF_TARGET_ESP32C3
    r.chip = 1;
#else
    r.chip = 0;
#endif
    memcpy(r.csi, info->buf, info->len < 256 ? info->len : 256);
    g_pending_valid = true;
}

// 自律モード (トークン): 送信役 0..4 が順番に、自分より番号の大きい全員へユニキャスト PING を 1 回ずつ送る (15 組を 1 回ずつ)。
// 終わったら次の送信役へトークン (type 6) を渡す (4 の次は 0)。ノード 0 は周期 g_period_us ごとに 1 巡を始め、
// トークンが戻らなければ 60 ms で次の巡回を始める。ノード 5 は送信役にならない (ACK はハードウェアが返す)。
static volatile bool g_run = false; static volatile uint32_t g_period_us = 16667;
static QueueHandle_t g_tok_q; static volatile bool g_cycle_done = true; static volatile uint32_t g_cycle = 0, g_n_cycles = 0, g_n_tok_lost = 0;
#define LAST_SENDER 4
static volatile uint32_t g_cycle_start = 0, g_cycle_us_sum = 0, g_n_stale = 0;
// 手番の時間の内訳 (計測用): トークンを受けてからタスクが動くまで / PING の送信完了がそろうまで / トークンの送信完了まで
static volatile uint32_t g_tok_rx_us = 0; static uint64_t g_m_wake = 0, g_m_pings = 0, g_m_tok = 0, g_m_first = 0, g_m_gap = 0; static uint32_t g_m_n = 0, g_m_ngap = 0;

// チャネルの切り替え (実験 E2): 1 巡ごとに ch 1 → 6 → 11 → 1 …。各ノードは「その巡回で自分が送ることも受けることもなくなった時点」で次へ移る。
//   ノード 0〜3: 自分の番が終わり、トークンを渡して突き合わせを終えた後 (それ以降は自分あての PING が来ない)
//   ノード 4: 自分の番の後、先に次のチャネルへ移ってからトークンをノード 0 へ渡す (ノード 0 はすでに次のチャネルにいる)
//   ノード 5: ノード 4 からの PING (その巡回の最後) を受けた後
// 何も受けない状態が 300 ms 続いたら ch 1 に戻る (ノード 0 はトークンが戻らなかったとき ch 1 からやり直す)。
static volatile bool g_hop = false; static const uint8_t HOP[3] = {1, 6, 11}; static volatile int g_hop_i = 0;
static volatile uint32_t HOP_SETTLE_US = 2000;   // チャネルを切り替えてから送るまで待つ時間 ("HS <µs>" で変更)
static volatile uint32_t g_last_rx_us = 0, g_n_hop = 0, g_n_hop_reset = 0;
static volatile uint32_t g_n_ch_err = 0, g_ch_us_sum = 0, g_ch_n = 0; static volatile int g_ch_last_err = 0;
static void set_ch(uint8_t ch) {
    uint32_t t0 = (uint32_t)esp_timer_get_time();
    esp_err_t e = esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    g_ch_us_sum += (uint32_t)esp_timer_get_time() - t0; g_ch_n++;
    if (e != ESP_OK) { g_n_ch_err++; g_ch_last_err = e; }
}
static void hop_next() { g_hop_i = (g_hop_i + 1) % 3; set_ch(HOP[g_hop_i]); g_n_hop++; }

static void recv_cb(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < (int)sizeof(Payload)) return;
    const Payload *p = (const Payload *)data;
    if (p->m0 != 'P' || p->m1 != 'G') return;
    uint32_t now = (uint32_t)esp_timer_get_time();
    g_last_rx_us = now;
    if (p->type == 6) {   // トークン (次の送信役の指名)。記録はしない
        g_pending_valid = false;
        if (p->target == g_id && g_run) { uint32_t cyc = p->seq; g_tok_rx_us = now; xQueueSend(g_tok_q, &cyc, 0); }
        return;
    }
    if (p->type == 1 && p->target == g_id) {   // 自分あての PING → すぐ PONG を返す
        PongJob j = {p->seq, p->sender, now}; xQueueSend(g_pong_q, &j, 0);
    }
    if (g_pending_valid && g_pending.tx_id == p->sender) {
        g_pending.type = p->type; g_pending.target = p->target; g_pending.seq = p->seq;
        if (xQueueSend(g_out_q, &g_pending, 0) != pdTRUE) g_drop++;
        g_pending_valid = false;
    }
    // ノード 5 は、その巡回の最後の PING (ノード 4 から) を受けたら次のチャネルへ (コールバックの中では切り替えず、タスクに頼む)
    if (g_hop && g_run && g_id == 5 && p->type == 4 && p->sender == LAST_SENDER && p->target == g_id) { uint32_t m = 0xFFFFFFFE; xQueueSend(g_tok_q, &m, 0); }
}

static void send_frame(uint8_t type, uint8_t target, uint32_t seq) {
    Payload p = {'P', 'G', type, (uint8_t)g_id, target, seq};
    esp_now_send(BCAST, (const uint8_t *)&p, sizeof(p));
}

// ユニキャストの PING (type 4)。相手のハードウェアが ACK を返すので、その CSI を待ち受ける。相手は PONG を返さない。
static void send_unicast_ping(uint8_t target, uint32_t seq) {
    Payload p = {'P', 'G', 4, (uint8_t)g_id, target, seq};
    g_ack_from = target; g_ack_seq = seq; g_ack_sent_us = (uint32_t)esp_timer_get_time();
    g_ack_wait_until = g_ack_sent_us + g_ack_window_us;   // この時間内に届いた「送信元のない CSI」を ACK とみなす
    esp_now_send(NODE_MAC[target], (const uint8_t *)&p, sizeof(p));
}

static void pong_task(void *) {   // PONG を最短で返す
    PongJob j;
    while (true) if (xQueueReceive(g_pong_q, &j, portMAX_DELAY) == pdTRUE) {
        send_frame(2, j.to, j.seq); g_n_pong++;
        // PING を受けてから送信を頼むまでの時間を、自分の PONG の記録用に残す (PC 側で seq と送信元で突き合わせる)
        Record r; memset(&r, 0, sizeof(r)); r.magic = 0xB0B0C510; r.rx_id = (uint8_t)g_id; r.tx_id = (uint8_t)g_id; r.type = 3;
        r.target = j.to; r.seq = j.seq; r.t_rx_us = j.t_rx; r.t_aux_us = (uint32_t)esp_timer_get_time() - j.t_rx;
        xQueueSend(g_out_q, &r, 0);
    }
}

// 短い記録 (magic 0xB0B0C511): ヘッダー 28 バイト + CSI の先頭 128 バイト (LLTF) = 156 バイト。往復のレートを上げるときに使う
static volatile bool g_short = false;
static void exchange(uint8_t t) {   // 1 往復: ユニキャスト PING を送り、ACK の CSI がそろう (最大 2 ms) まで待つ
    g_ack_window_us = 2000;
    uint32_t t0 = (uint32_t)esp_timer_get_time();
    send_unicast_ping(t, ++g_seq);
    while (g_ack_wait_until && (uint32_t)esp_timer_get_time() - t0 < 2000) delayMicroseconds(10);
    g_ack_wait_until = 0;
    delayMicroseconds(80);   // 遅れて届く CSI をやり過ごす
}

// 速い方式の 1 手番: 自分より番号の大きい全員へ reps 回ずつユニキャスト PING を続けて送り、next ≥ 0 ならその後にトークンを送る。
// 送信完了のコールバックは送った順に (まとめて遅れて) 届くので、i 番目のコールバック = i 番目に送ったもの。
// ACK (宛先が自分のものだけ) を受信時刻の順に並べ、個数がちょうど合うときだけ送った順に対応づける (合わなければ取り違えを避けて捨てる)。
static volatile int g_plen = 0;
static volatile uint32_t g_slot_off = 0; static volatile uint8_t g_slot_mask = 0; static uint32_t g_turn_end = 0, g_n_gskip = 0;   // 時間枠の方式: 最後の手番の終わりの時刻、捨てた古い G の数   // 時間枠の方式の自分の枠   // PING の中身の長さ (0 なら最小の 9 バイト)
static volatile uint8_t g_tmask = 0;   // USB 指示 ("T <マスク> <回数>") の相手。0 なら自分より番号の大きい全員
static void fast_turn(int nt, int8_t next, uint32_t tokseq) {
    uint8_t tm = g_tmask; if (tm == 0) for (int t = g_id + 1; t < 6; t++) tm |= (1 << t);
    tm &= ~(1 << g_id); nt = 0; for (int t = 0; t < 6; t++) nt += (tm >> t) & 1;
    if (nt == 0) return;
    portENTER_CRITICAL(&g_mux); g_n_ev = 0; g_n_acks = 0; portEXIT_CRITICAL(&g_mux);
    g_pipe = true; g_ev_on = true; g_ack_wait_until = (uint32_t)esp_timer_get_time() + 20000;
    int reps = g_reps; if (reps < 1) reps = 1; if (nt * reps > PMAX - 1) reps = (PMAX - 1) / nt;
    g_nsend = nt * reps; g_cb_n = 0; g_tok_sent = false; g_tok_to = next; g_fast = true;
    int i = 0;
    for (int r = 0; r < reps; r++) for (int t = 0; t < 6; t++) {
        if (!((tm >> t) & 1)) continue;
        g_st[i] = (uint8_t)t; g_sseq[i] = ++g_seq; i++;
        Payload q = {'P', 'G', 4, (uint8_t)g_id, (uint8_t)t, g_seq};
        { static uint8_t pb[250]; memset(pb, 0, sizeof(pb)); memcpy(pb, &q, sizeof(q)); int n = g_plen < (int)sizeof(q) ? (int)sizeof(q) : (g_plen > 250 ? 250 : g_plen);
          send_fc(NODE_MAC[t], pb, n); }   // PING の長さ ("PL <バイト>") を変えられるように
    }
    if (next >= 0) {
        Payload p = {'P', 'G', 6, (uint8_t)g_id, (uint8_t)next, tokseq};
        send_fc(NODE_MAC[next], &p, sizeof(p));   // 送った順に出るので、トークンは全 PING の後
    }
    int ncb = g_nsend + (next >= 0 ? 1 : 0);
    uint32_t t1 = (uint32_t)esp_timer_get_time();
    while (g_cb_n < ncb && (uint32_t)esp_timer_get_time() - t1 < 15000) delayMicroseconds(20);   // 全部のコールバック
    delayMicroseconds(300);   // 遅れて届く ACK の CSI
    g_ack_wait_until = 0; g_pipe = false; g_fast = false; g_ev_on = false;
    SendEv ev[PMAX]; AckC *ac = g_acks; int ne, na;
    portENTER_CRITICAL(&g_mux); ne = g_n_ev; na = g_n_acks; for (int k = 0; k < ne; k++) ev[k] = g_ev[k]; portEXIT_CRITICAL(&g_mux);
    for (int a = 1; a < na; a++) for (int b = a; b > 0 && (int32_t)(ac[b].ts - ac[b - 1].ts) < 0; b--) { AckC x = ac[b]; ac[b] = ac[b - 1]; ac[b - 1] = x; }
    int nok = 0; for (int k = 0; k < ne; k++) nok += ev[k].ok;
    bool ok = (ne == g_nsend) && ((next >= 0) ? (g_tok_sent && na == nok + 1) : (na == nok));
    if (ok) {
        int k = 0;
        for (int e = 0; e < ne; e++) if (ev[e].ok) {
            Record &r = ac[k].r; r.tx_id = ev[e].t; r.type = 5; r.target = ev[e].t; r.seq = ev[e].seq; r.t_aux_us = 0;
            if (xQueueSend(g_out_q, &r, 0) != pdTRUE) g_drop++;
            k++; g_n_pipe_match++;
        }
    } else g_n_pipe_unmatched++;
    g_tok_to = -1;
}

// 同時送信方式 ("FR <周期 µs>"、0 で止める): 送信役 0〜4 がそれぞれ、周期ごとに fast_turn をトークンなしで行う
static volatile uint32_t g_free_period = 0;
static void free_task(void *) {
    while (true) {
        if (g_free_period == 0 || g_id > LAST_SENDER) { vTaskDelay(10); continue; }
        uint32_t t0 = (uint32_t)esp_timer_get_time();
        fast_turn(5 - g_id, -1, 0);
        while (true) {   // 周期の残りを待つ (長いときは CPU を手放す。C3 は 1 コアなので PC への送出を止めないように)
            int32_t rem = (int32_t)(g_free_period - ((uint32_t)esp_timer_get_time() - t0));
            if (rem <= 0) break;
            if (rem > 2500) vTaskDelay(1); else break;   // 残りが短ければ抜けて、下の vTaskDelay(1) で待つ
        }
        vTaskDelay(1);   // 毎回かならず CPU を手放す (C3 は 1 コアなので、手放さないと PC への送出が止まる)
    }
}

static void token_task(void *) {
    uint32_t cyc;
    while (true) if (xQueueReceive(g_tok_q, &cyc, portMAX_DELAY) == pdTRUE) {
        if (!g_run) continue;
        if (cyc == 0xFFFFFFFE) { delayMicroseconds(1500); hop_next(); continue; }   // ノード 5 のチャネル切り替え
        if (g_id == 0 && cyc != 0xFFFFFFFF && cyc == g_cycle && !g_cycle_done) {   // 一巡して戻ってきた
            uint32_t now = (uint32_t)esp_timer_get_time();
            g_cycle_done = true; g_n_cycles++; g_cycle_us_sum += now - g_cycle_start;
            if (now - g_cycle_start < g_period_us) continue;   // まだ周期の途中なら、loop() の cycle_tick が始める
            g_cycle_start = now; g_cycle++; g_cycle_done = false;   // 周期を過ぎていれば、待たずに次の巡回を始める
            cyc = 0xFFFFFFFF;
        }
        // 古い巡回のトークン (遅れて戻ってきたもの) はノード 0 で捨てる。捨てないと複数のトークンが同時に回り、送信が重なる
        else if (g_id == 0 && cyc != 0xFFFFFFFF) { g_n_stale++; continue; }
        // 1. 自分より番号の大きい全員へ、続けてユニキャスト PING を送る
        int nt = 5 - g_id;
        uint32_t t_wake = (uint32_t)esp_timer_get_time(); bool by_tok = (cyc != 0xFFFFFFFF);
        portENTER_CRITICAL(&g_mux); g_n_ev = 0; g_n_acks = 0; portEXIT_CRITICAL(&g_mux);
        g_pipe = true; g_ev_on = true; g_ack_wait_until = (uint32_t)esp_timer_get_time() + 20000;
        uint8_t next = (g_id >= LAST_SENDER) ? 0 : (uint8_t)(g_id + 1);
        if (!g_hop) {   // 速い方式
            fast_turn(nt, (int8_t)next, (g_id == 0) ? g_cycle : cyc);
            if (by_tok) g_m_wake += t_wake - g_tok_rx_us;
            g_m_pings += (uint32_t)esp_timer_get_time() - t_wake; g_m_n++;
            continue;
        }
        for (int t = g_id + 1; t < 6; t++) {
            g_tseq[t] = ++g_seq;
            Payload q = {'P', 'G', 4, (uint8_t)g_id, (uint8_t)t, g_seq};
            esp_now_send(NODE_MAC[t], (const uint8_t *)&q, sizeof(q));
        }
        uint32_t t0 = (uint32_t)esp_timer_get_time();
        while (g_n_ev < nt && (uint32_t)esp_timer_get_time() - t0 < 8000) delayMicroseconds(20);   // 全員の送信完了を待つ
        uint32_t t_pings = (uint32_t)esp_timer_get_time();
        // 2. すぐに次の送信役へトークンを渡す (突き合わせは後で)
        Payload p = {'P', 'G', 6, (uint8_t)g_id, next, (g_id == 0) ? g_cycle : cyc};
        g_ev_on = false;   // トークンの送信完了は、往復の送信完了として数えない
        bool hop_first = g_hop && g_id == LAST_SENDER;
        // ノード 4: 遅れて届く CSI を待ってから次のチャネルへ移り、無線の切り替えが終わるのを待って (esp_wifi_set_channel はすぐ戻るが、実際の切り替えは後で起きる) トークンを渡す
        if (hop_first) { delayMicroseconds(1500); hop_next(); delayMicroseconds(HOP_SETTLE_US); }
        uint32_t t_tok = (uint32_t)esp_timer_get_time();
        g_tok_sent = false; g_tok_to = next;
        esp_now_send(NODE_MAC[next], (const uint8_t *)&p, sizeof(p));   // ユニキャスト (ACK と再送で確実に届く)
        // 3. 遅れて届く ACK の CSI を待ってから、送信完了の順番と時刻で対応づける
        { uint32_t tw = (uint32_t)esp_timer_get_time(); while (!g_tok_sent && (uint32_t)esp_timer_get_time() - tw < 20000) delayMicroseconds(10); }
        { uint32_t t_ts = (uint32_t)esp_timer_get_time();
          if (by_tok) g_m_wake += t_wake - g_tok_rx_us; g_m_pings += t_pings - t_wake; g_m_tok += t_ts - t_tok; g_m_n++;
          if (g_n_ev >= 1) g_m_first += g_ev[0].tau - t_wake;                                      // 最初の送信完了まで
          if (g_n_ev >= 2) { g_m_gap += (g_ev[g_n_ev - 1].tau - g_ev[0].tau) / (g_n_ev - 1); g_m_ngap++; } }   // 送信完了どうしの間隔
        if (!hop_first) delayMicroseconds(1500); else delayMicroseconds(300);
        g_ack_wait_until = 0; g_pipe = false;
        SendEv ev[PMAX]; AckC *ac = g_acks; int ne, na;
        portENTER_CRITICAL(&g_mux); ne = g_n_ev; na = g_n_acks; for (int i = 0; i < ne; i++) ev[i] = g_ev[i]; portEXIT_CRITICAL(&g_mux);
        // ACK を受信時刻の順に並べる (コールバックの到着順はずれることがある)
        for (int i = 1; i < na; i++) for (int j = i; j > 0 && (int32_t)(ac[j].ts - ac[j - 1].ts) < 0; j--) { AckC x = ac[j]; ac[j] = ac[j - 1]; ac[j - 1] = x; }
        int nok = 0; for (int i = 0; i < ne; i++) nok += ev[i].ok;
        // トークンの ACK を除く: 受信時刻 (rx の時計) をこのノードのタイマーに直して、トークンを送った後のものを捨てる。
        // 時計のずれ g_off は、きれいに対応がついた往復から推定する。まだ分からなければ、一番遅い ACK をトークンのものとみなす
        // まず個数で判断する (ACK が 1 つ多ければ、一番遅いものがトークンの ACK)。
        // 個数が同じときだけ、時刻で確認する: 一番遅い ACK がトークンを送ってから 300 µs 以上後なら、それはトークンの ACK で、
        // 往復の ACK が 1 つ欠けている → 対応づけない (時計のずれの推定には数百 µs の誤差があるので余裕を持たせる)
        bool skip = false;
        if (na == nok + 1) na--;
        else if (na == nok && na > 0 && g_off_valid && (int32_t)(ac[na - 1].ts + g_off - t_tok) > 300) skip = true;
        if (skip) na = -1;
        // 成功した送信の数と ACK の数が同じなら、順番どおりに対応づける。違えば、記録しない (取り違えを避ける)
        if (nok == na && na > 0) {
            int32_t offs[PMAX]; int no = 0;
            { int k = 0; for (int i = 0; i < ne; i++) if (ev[i].ok) offs[no++] = (int32_t)(ev[i].tau - ac[k++].ts); }
            for (int i = 1; i < no; i++) for (int j = i; j > 0 && offs[j] < offs[j - 1]; j--) { int32_t x = offs[j]; offs[j] = offs[j - 1]; offs[j - 1] = x; }
            int32_t med = offs[no / 2] - 130;   // 送信完了は ACK の受信の約 130 µs 後 (C3 で実測)
            g_off = g_off_valid ? (g_off + (med - g_off) / 8) : med; g_off_valid = true;
        }
        if (nok == na) {
            int k = 0;
            for (int i = 0; i < ne; i++) if (ev[i].ok) {
                Record &r = ac[k].r; r.tx_id = ev[i].t; r.type = 5; r.target = ev[i].t; r.seq = ev[i].seq; r.t_aux_us = ev[i].tau - ac[k].ts;
                if (xQueueSend(g_out_q, &r, 0) != pdTRUE) g_drop++;
                k++; g_n_pipe_match++;
            }
        } else g_n_pipe_unmatched++;
        if (g_hop && g_id < LAST_SENDER) {   // ノード 0〜3: この巡回の自分の役目が終わったので次のチャネルへ (トークンの送り直しが終わってから)
            uint32_t tw = (uint32_t)esp_timer_get_time();
            while (!g_tok_sent && (uint32_t)esp_timer_get_time() - tw < 20000) delayMicroseconds(50);
            hop_next();
        }
        g_tok_to = -1;
    }
}

static void out_task(void *) {   // PC への送出
    Record r;
    while (true) if (xQueueReceive(g_out_q, &r, portMAX_DELAY) == pdTRUE) {
        if (g_short) { r.magic = 0xB0B0C511; Serial.write((const uint8_t *)&r, sizeof(r) - 128); }
        else Serial.write((const uint8_t *)&r, sizeof(r));
        g_n_rec++;
    }
}

void setup() {
    Serial.setTxBufferSize(8192);
    Serial.begin(921600);
    WiFi.mode(WIFI_STA); WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
    uint8_t mac[6]; esp_wifi_get_mac(WIFI_IF_STA, mac); g_id = id_of(mac); memcpy(g_my_mac, mac, 6);
    esp_wifi_set_promiscuous(true); esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE); esp_wifi_set_promiscuous(false);
    esp_now_init();
    esp_now_peer_info_t peer = {}; memcpy(peer.peer_addr, BCAST, 6); peer.channel = 0; peer.ifidx = WIFI_IF_STA; esp_now_add_peer(&peer);
    for (int i = 0; i < 6; i++) if (i != g_id) { memcpy(peer.peer_addr, NODE_MAC[i], 6); esp_now_add_peer(&peer); }   // ユニキャスト用
    esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS0_LGI);
#if CONFIG_IDF_TARGET_ESP32C3
    esp_wifi_set_max_tx_power(44);   // C3 は 11 dBm (15 dBm 以上だと電波がまともに出ない。8.5 dBm だと遠い相手に届きにくい)
#endif
    wifi_csi_config_t cc = {}; cc.lltf_en = true; cc.htltf_en = true; cc.stbc_htltf2_en = true; cc.ltf_merge_en = false; cc.channel_filter_en = false; cc.manu_scale = false; cc.shift = 0;
    // 制御フレーム (ACK) も受けるようにする。既定のフィルタには制御フレームが入っていない
    wifi_promiscuous_filter_t flt = {}; flt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_CTRL; esp_wifi_set_promiscuous_filter(&flt);
    wifi_promiscuous_filter_t cflt = {}; cflt.filter_mask = WIFI_PROMIS_CTRL_FILTER_MASK_ACK; esp_wifi_set_promiscuous_ctrl_filter(&cflt);
    esp_wifi_set_promiscuous_rx_cb(&promisc_cb);   // 自分あての ACK を見つける
    esp_wifi_set_promiscuous(true);   // CSI を確実に出すため (C3 ではプロミスキャスの後に CSI を設定する必要があった)
    esp_wifi_set_csi_config(&cc); esp_wifi_set_csi_rx_cb(&csi_cb, NULL); esp_wifi_set_csi(true);
    esp_now_register_recv_cb(&recv_cb);
    esp_now_register_send_cb(&on_sent);
    g_out_q = xQueueCreate(64, sizeof(Record)); g_pong_q = xQueueCreate(8, sizeof(PongJob));
    xTaskCreate(pong_task, "pong", 4096, NULL, configMAX_PRIORITIES - 2, NULL);
    xTaskCreate(out_task, "out", 4096, NULL, 3, NULL);
    g_tok_q = xQueueCreate(4, sizeof(uint32_t));
    xTaskCreate(token_task, "token", 4096, NULL, configMAX_PRIORITIES - 3, NULL);
    xTaskCreate(free_task, "free", 4096, NULL, configMAX_PRIORITIES - 3, NULL);
}

static void cycle_tick() {   // ノード 0: 周期ごとに 1 巡を始める
    if (!g_run || g_id != 0) return;
    uint32_t now = (uint32_t)esp_timer_get_time();
    if (now - g_cycle_start < g_period_us) return;
    if (!g_cycle_done && now - g_cycle_start < 60000) return;
    if (!g_cycle_done) { g_n_tok_lost++; if (g_hop) { g_hop_i = 0; set_ch(HOP[0]); g_n_hop_reset++; } }
    g_cycle_start = now; g_cycle++; g_cycle_done = false;
    uint32_t start = 0xFFFFFFFF; xQueueSend(g_tok_q, &start, 0);   // 自分の分から始める
}

void loop() {
    if (Serial.available()) {
        String c = Serial.readStringUntil('\n'); c.trim();
        if (c.startsWith("P ")) { int j = c.substring(2).toInt(); if (j >= 0 && j < 6 && j != g_id) send_frame(1, (uint8_t)j, ++g_seq); }
        else if (c.startsWith("U ")) { int j = c.substring(2).toInt(); if (j >= 0 && j < 6 && j != g_id) send_unicast_ping((uint8_t)j, ++g_seq); }
        else if (c.startsWith("B ")) {   // バースト: 自分以外の全員に、gap µs おきにユニキャスト PING を続けて送る
            // gap は最短の間隔。ACK の CSI が届く (または 3 ms たつ) まで待ってから次の相手に送る (取り違えを防ぐ)
            uint32_t gap = (uint32_t)c.substring(2).toInt(); if (gap < 500) gap = 500;
            g_ack_window_us = 3000;
            for (int t = 0; t < 6; t++) {
                if (t == g_id) continue;
                uint32_t t0 = (uint32_t)esp_timer_get_time();
                send_unicast_ping((uint8_t)t, ++g_seq);
                while (g_ack_wait_until && (uint32_t)esp_timer_get_time() - t0 < 3000) delayMicroseconds(20);
                g_ack_wait_until = 0;   // 届かなかった場合も、ここで待ち受けを打ち切る
                while ((uint32_t)esp_timer_get_time() - t0 < gap) delayMicroseconds(20);
                delayMicroseconds(200);   // 遅れて届く CSI をやり過ごす
            }
            Serial.print("\n[BD]\n");   // バースト終了の合図 (PC はこれを待ってから次の送信役に指示する)
        }
        else if (c.startsWith("R ")) {   // 自律モード: "R <周期 µs>" で開始、"R 0" で停止
            uint32_t per = (uint32_t)c.substring(2).toInt();
            if (per == 0) g_run = false; else { g_period_us = per; g_cycle_done = true; g_run = true; }
        }
        else if (c == "H 1") { g_hop = true; g_hop_i = 0; set_ch(HOP[0]); g_last_rx_us = (uint32_t)esp_timer_get_time(); }   // チャネル切り替えを始める (ch 1 から)
        else if (c == "H 0") { g_hop = false; set_ch(CHANNEL); }
        else if (c.startsWith("HS ")) HOP_SETTLE_US = (uint32_t)c.substring(3).toInt();
        else if (c == "C1") g_short = true;
        else if (c == "C0") g_short = false;
        else if (c == "ST") Serial.printf("\n[ST] id %d n %u wake_us %.0f pings_us %.0f tok_us %.0f first_us %.0f gap_us %.0f nomem %u senderr %u tokfail %u gskip %u\n", g_id, (unsigned)g_m_n, g_m_n ? (double)g_m_wake / g_m_n : 0.0, g_m_n ? (double)g_m_pings / g_m_n : 0.0, g_m_n ? (double)g_m_tok / g_m_n : 0.0, g_m_n ? (double)g_m_first / g_m_n : 0.0, g_m_ngap ? (double)g_m_gap / g_m_ngap : 0.0, (unsigned)g_n_nomem, (unsigned)g_n_senderr, (unsigned)g_n_tok_fail, (unsigned)g_n_gskip);
        else if (c.startsWith("N ")) g_reps = c.substring(2).toInt();
        else if (c.startsWith("PL ")) g_plen = c.substring(3).toInt();
        // 時間枠の方式 (USB で一斉に開始): "SL <遅れ µs> <相手のマスク>" で自分の枠を設定し、"G" を受けたら遅れの後に 1 手番
        else if (c.startsWith("SL ")) { int sp = c.indexOf(' ', 3); g_slot_off = (uint32_t)c.substring(3, sp).toInt(); g_slot_mask = (uint8_t)c.substring(sp + 1).toInt(); }
        else if (c == "G") { if (g_slot_mask) { uint32_t g0 = (uint32_t)esp_timer_get_time();
            // 手番の最中に届いて待たされていた古い "G" は捨てる (弱い組の送り直しで手番が長引くと、次の G が手番の直後に遅れて実行され、
            // 他の台の時間枠とぶつかって連鎖的に抜ける。そうした G は手番の終わりの直後に読まれるので、終わりから 1.5 ms 以内なら捨てる)
            if (g_turn_end && g0 - g_turn_end < 1500) { g_n_gskip++; }
            else {
                while ((uint32_t)esp_timer_get_time() - g0 < g_slot_off) delayMicroseconds(20);
                g_tmask = g_slot_mask; fast_turn(0, -1, 0); g_tmask = 0; g_turn_end = (uint32_t)esp_timer_get_time();
            } } }
        else if (c.startsWith("FR ")) g_free_period = (uint32_t)c.substring(3).toInt();
        else if (c.startsWith("T ")) { int sp = c.indexOf(' ', 2); g_tmask = (uint8_t)c.substring(2, sp).toInt(); if (sp > 0) g_reps = c.substring(sp + 1).toInt();
            uint32_t u0 = g_n_pipe_match; fast_turn(0, -1, 0); g_tmask = 0; Serial.printf("\n[TD] %d %u\n", g_id, (unsigned)(g_n_pipe_match - u0)); }
        else if (c.startsWith("PR ")) { esp_err_t e = esp_wifi_config_espnow_rate(WIFI_IF_STA, (wifi_phy_rate_t)c.substring(3).toInt()); Serial.printf("\n[PR] id %d rate %d err 0x%x\n", g_id, (int)c.substring(3).toInt(), e); }   // PHY レート (0x10 = MCS0、0x13 = MCS3、0x0B = 6M、0x09 = 24M)
        else if (c == "SH") Serial.printf("\n[HOP] id %d hop %u reset %u ch %u err %u last_err 0x%x set_us %.0f\n", g_id, (unsigned)g_n_hop, (unsigned)g_n_hop_reset, HOP[g_hop_i], (unsigned)g_n_ch_err, g_ch_last_err, g_ch_n ? (double)g_ch_us_sum / g_ch_n : 0.0);
        else if (c == "S") Serial.printf("\n[PP] id %d csi %u rec %u pong %u drop %u ack %u unknown %u ack_me %u match %u cycles %u tok_lost %u pipe %u unmatched %u cycle_ms %.2f off %d\n", g_id, (unsigned)g_n_csi, (unsigned)g_n_rec, (unsigned)g_n_pong, (unsigned)g_drop, (unsigned)g_n_ack, (unsigned)g_n_unknown, (unsigned)g_n_ack_me, (unsigned)g_n_match, (unsigned)g_n_cycles, (unsigned)g_n_tok_lost, (unsigned)g_n_pipe_match, (unsigned)g_n_pipe_unmatched, g_n_cycles ? g_cycle_us_sum / 1000.0 / g_n_cycles : 0.0, (int)g_off);
    }
    cycle_tick();
    // ノード 1〜5: 何も受けない状態が 300 ms 続いたら ch 1 に戻る (チャネルの切り替えがずれたときの復帰)
    if (g_hop && g_run && g_id != 0 && (uint32_t)esp_timer_get_time() - g_last_rx_us > 300000) {
        g_hop_i = 0; set_ch(HOP[0]); g_last_rx_us = (uint32_t)esp_timer_get_time(); g_n_hop_reset++;
    }
    delay(1);
}
