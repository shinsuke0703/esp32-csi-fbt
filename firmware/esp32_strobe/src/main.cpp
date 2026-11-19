// CSI ストロボ送信機 (ESP32-DevKitC)
//  - ESP-NOW ブロードキャストを一定周期で送る (既定 100 Hz、チャンネル 6、HT20)
//  - 送信レートを HT MCS0 に固定 (802.11b だと受信側で CSI が出ないため OFDM にする)
//  - ペイロード: 'C','S','I','S' + 送信番号 (uint32) + 送信時刻 us (uint32)
//  - シリアル (115200) コマンド: C<ch> チャンネル, P<ms> 周期, S 統計
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

static uint8_t g_ch = 6;
static uint32_t g_period_ms = 10;
static uint32_t g_seq = 0, g_ok = 0, g_fail = 0;
static esp_err_t g_rate_err = ESP_OK;   // esp_wifi_config_espnow_rate の結果 (S で表示)
static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static void on_sent(const uint8_t *mac, esp_now_send_status_t st) { if (st == ESP_NOW_SEND_SUCCESS) g_ok++; else g_fail++; }

static void set_channel(uint8_t ch) {
    esp_wifi_set_promiscuous(true);   // 未接続の STA でチャンネルを変えるため
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    esp_now_del_peer(BCAST);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BCAST, 6); peer.channel = ch; peer.ifidx = WIFI_IF_STA; peer.encrypt = false;
    esp_now_add_peer(&peer);
    g_ch = ch;
}

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
    esp_now_init();
    esp_now_register_send_cb(on_sent);
    esp_err_t e = esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS0_LGI); g_rate_err = e;
    set_channel(g_ch);
    // C3 の小型ボードは送信出力が高いと電波がまともに出ない (実測: 15 dBm 以上で受信ほぼ 0、11 dBm 以下で全数受信)。8.5 dBm にする
    esp_wifi_set_max_tx_power(34);
    Serial.printf("\n[STROBE] MAC %s ch %u period %u ms rate MCS0 (%s)\n", WiFi.macAddress().c_str(), g_ch, (unsigned)g_period_ms, esp_err_to_name(e));
}

void loop() {
    static uint32_t next = 0;
    uint32_t now = millis();
    if ((int32_t)(now - next) >= 0) {
        next += g_period_ms;
        if ((int32_t)(now - next) > 100) next = now + g_period_ms;   // 大きく遅れたら追いかけない
        uint8_t pl[12] = {'C', 'S', 'I', 'S'};
        uint32_t us = (uint32_t)micros();
        memcpy(pl + 4, &g_seq, 4); memcpy(pl + 8, &us, 4); g_seq++;
        esp_now_send(BCAST, pl, sizeof(pl));
    }
    if (Serial.available()) {
        String c = Serial.readStringUntil('\n'); c.trim();
        if ((c[0] == 'C' || c[0] == 'c') && c.length() > 1) { int ch = c.substring(1).toInt(); if (ch >= 1 && ch <= 13) set_channel(ch); Serial.printf("[STROBE] ch %u\n", g_ch); }
        else if ((c[0] == 'P' || c[0] == 'p') && c.length() > 1) { int p = c.substring(1).toInt(); if (p >= 2) g_period_ms = p; Serial.printf("[STROBE] period %u ms\n", (unsigned)g_period_ms); }
        else if ((c[0] == 'T' || c[0] == 't') && c.length() > 1) {   // T<0.25 dBm 単位> 送信出力 (例: T34 = 8.5 dBm)
            esp_err_t e = esp_wifi_set_max_tx_power((int8_t)c.substring(1).toInt()); int8_t p = 0; esp_wifi_get_max_tx_power(&p);
            Serial.printf("[STROBE] tx power -> %s (%.2f dBm)\n", esp_err_to_name(e), p / 4.0);
        }
        else if (c.startsWith("RATE")) {   // RATE6 = レガシー OFDM 6M, RATEM0 = HT MCS0, RATE1 = 11b 1M
            wifi_phy_rate_t r = c == "RATE6" ? WIFI_PHY_RATE_6M : c == "RATE1" ? WIFI_PHY_RATE_1M_L : WIFI_PHY_RATE_MCS0_LGI;
            g_rate_err = esp_wifi_config_espnow_rate(WIFI_IF_STA, r);
            Serial.printf("[STROBE] rate %s -> %s\n", c.c_str(), esp_err_to_name(g_rate_err));
        }
        else if (c[0] == 'S' || c[0] == 's') Serial.printf("[STROBE] seq %u ok %u fail %u ch %u rate_cfg %s mac %s\n", (unsigned)g_seq, (unsigned)g_ok, (unsigned)g_fail, g_ch, esp_err_to_name(g_rate_err), WiFi.macAddress().c_str());
    }
    delay(1);
}
