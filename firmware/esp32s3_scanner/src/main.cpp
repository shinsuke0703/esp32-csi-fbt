#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_wifi_types.h>
#include <esp_system.h>
#include <esp_log.h>
#include <nvs_flash.h>

// ============================================================================
//  ESP32 Standalone 2.4GHz HT40 Wi-Fi CSI Scanner & Radar
//  Features:
//   1. Promiscuous 2.4GHz HT40 CSI capture (LLTF + HT-LTF: 114 subcarriers).
//   2. Active 802.11 Probe Pulse Injection (25Hz) for standalone radar operation
//      (works even with zero external TX nodes!).
//   3. Dual-mode UART output:
//      - Binary (0xC510C510 compatible with OptiTrack 289-byte format)
//      - Text CSV for real-time serial plotting / diagnostic.
//   4. On-board LED pulse indicator (GPIO 2).
//   5. Dynamic Channel Control via Serial (commands: C1, C6, C11, etc.).
// ============================================================================

#define LED_PIN                     2
// S3/C3 版: 失敗しても止めずに表示する (USB に出る前に abort すると何も見えないため)
#define CHK(x) do { esp_err_t _e = (x); Serial.printf("[%s] %s\n", _e == ESP_OK ? "ok" : "NG", #x); if (_e != ESP_OK) Serial.printf("   -> %s\n", esp_err_to_name(_e)); Serial.flush(); } while (0)
#define DEFAULT_WIFI_CHANNEL        6
#define CSI_UART_BAUD_RATE          921600
#define TRACKING_MAGIC              0xC510C510
#define TRACKING_CSI_BYTES          256
#define ACTIVE_PROBE_INTERVAL_MS    40   // 25 Hz active radar pinging

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;                 // 0xC510C510 Frame Sync Header
    uint32_t device_id;             // 0x5C4E ("SCN") Standalone Scanner
    uint32_t sequence;
    uint32_t timestamp;
    int8_t rssi;                    // Receiver RSSI (dBm)
    uint8_t version;                // Protocol version = 7
    uint8_t first_word_invalid;     // ESP32 first word invalid flag
    uint16_t csi_length_bytes;      // Raw CSI length reported by PHY
    uint8_t csi_payload_ltf;        // 1 = HT-LTF field
    uint8_t rx_cwb;                 // 0 = HT20, 1 = HT40
    uint8_t rx_sig_mode;            // PHY mode (HT/VHT)
    uint8_t rx_secondary_channel;   // Secondary channel
    uint8_t rx_stbc;
    int8_t rx_noise_floor;
    uint8_t source_mac[6];
    int8_t csi_data[TRACKING_CSI_BYTES]; // 114 subcarriers complex (real, imag)
} tracking_packet_t;
#pragma pack(pop)

// State variables
static uint8_t g_current_channel = DEFAULT_WIFI_CHANNEL;
static uint32_t g_packet_sequence = 0;
static uint8_t g_self_mac[6] = {0};
static bool g_binary_mode = true;
static bool g_active_probe_enabled = false;   // C3 版: ストロボを使うので自前の送信は既定で止める
static QueueHandle_t g_csi_queue = NULL;

// Standard 802.11 Probe Request Frame Template (Broadcast)
static uint8_t g_probe_frame[] = {
    0x40, 0x00,                         // Type: Management, Subtype: Probe Request
    0x00, 0x00,                         // Duration
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // Destination: Broadcast
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Source MAC (overwritten in setup)
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // BSSID: Broadcast
    0x00, 0x00,                         // Sequence Control
    0x00, 0x00,                         // SSID IE: Wildcard (Broadcast)
    0x01, 0x08, 0x82, 0x84, 0x8b, 0x96, 0x0c, 0x12, 0x18, 0x24 // Supported Rates
};

// S3 版: ネイティブ USB なので CP210x の制限はない。間隔は "R<ms>" で変更可 (既定 10 ms = 最大 100 Hz)
static volatile uint32_t g_rate_limit_ms = 0;
static volatile uint32_t g_last_csi_time_ms = 0;
// S3 版: 送信元 MAC の絞り込み ("M aa:bb:cc:dd:ee:ff" で設定、"M" だけで解除)。1 リンクに揃えるため
static volatile bool g_mac_filter_en = true;    // C3 版: 既定でストロボ (DevKitC) だけ受ける
static uint8_t g_mac_filter[6] = {0xCC, 0xDB, 0xA7, 0x32, 0x13, 0x80};

// CSI RX Callback (Triggered by ESP32 Wi-Fi PHY for each incoming frame)
static volatile uint32_t g_cb_all = 0, g_cb_short = 0, g_cb_sent = 0, g_promisc_rx = 0;
static volatile uint32_t g_promisc_ofdm = 0;
static void promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
    g_promisc_rx++;
    const wifi_promiscuous_pkt_t *p = (const wifi_promiscuous_pkt_t *)buf;
    if (p->rx_ctrl.sig_mode != 0 || p->rx_ctrl.rate >= 0x0B) g_promisc_ofdm++;   // 11b (1/2/5.5/11M) 以外
}
static volatile bool g_stat = false;
static void wifi_csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    g_cb_all++;
    if (!info || !info->buf || info->len < 128) { g_cb_short++; return; }
    if (g_mac_filter_en && memcmp(info->mac, g_mac_filter, 6) != 0) return;

    uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
    if (now_ms - g_last_csi_time_ms < g_rate_limit_ms) {
        return;
    }
    g_last_csi_time_ms = now_ms;

    tracking_packet_t packet;
    memset(&packet, 0, sizeof(packet));

    packet.magic = TRACKING_MAGIC;
    packet.device_id = 0x00005343; // "SC" Standalone CSI Node
    packet.sequence = g_packet_sequence++;
    packet.timestamp = now_ms;
    packet.rssi = info->rx_ctrl.rssi;
    packet.version = 7;
    packet.first_word_invalid = (uint8_t)info->first_word_invalid;
    packet.csi_length_bytes = info->len;
    packet.csi_payload_ltf = 1; // HT-LTF
    packet.rx_cwb = info->rx_ctrl.cwb; // 1 for HT40
    packet.rx_sig_mode = info->rx_ctrl.sig_mode;
    packet.rx_secondary_channel = info->rx_ctrl.secondary_channel;
    packet.rx_stbc = info->rx_ctrl.stbc;
    packet.rx_noise_floor = info->rx_ctrl.noise_floor;
    memcpy(packet.source_mac, info->mac, 6);

    // In ESP32 HT40: info->buf has LLTF (first 128 bytes) followed by HT-LTF (256 bytes)
    const int8_t *src_csi = (const int8_t*)info->buf;
    if (info->len >= 384) {
        src_csi += 128; // Focus on 114 tones HT-LTF
    }
    size_t copy_bytes = (info->len >= TRACKING_CSI_BYTES) ? TRACKING_CSI_BYTES : info->len;
    memcpy(packet.csi_data, src_csi, copy_bytes);

    if (g_csi_queue) {
        xQueueSendFromISR(g_csi_queue, &packet, NULL); g_cb_sent++;
    }
}

// Active Probe Injection Task (Radar Pulsing)
static void active_radar_probe_task(void *param) {
    while (true) {
        if (g_active_probe_enabled) {
            // Update sequence counter in 802.11 header
            static uint16_t seq_num = 0;
            g_probe_frame[22] = (seq_num << 4) & 0xff;
            g_probe_frame[23] = (seq_num >> 4) & 0xff;
            seq_num++;

            // Transmit raw 802.11 broadcast frame on current channel
            esp_wifi_80211_tx(WIFI_IF_STA, g_probe_frame, sizeof(g_probe_frame), false);
        }
        vTaskDelay(pdMS_TO_TICKS(50)); // 20 Hz active radar pulse
    }
}

// UART Dispatch Task (Protected with rate limiting and large TX buffer)
static void uart_dispatch_task(void *param) {
    tracking_packet_t packet;
    while (true) {
        if (xQueueReceive(g_csi_queue, &packet, portMAX_DELAY) == pdTRUE) {
            digitalWrite(LED_PIN, HIGH);

            if (g_binary_mode) {
                Serial.write((const uint8_t*)&packet, sizeof(packet));
            } else {
                // Text CSV diagnostic mode
                Serial.printf("CSI,%u,%d,%u,%02X:%02X:%02X:%02X:%02X:%02X,%u",
                    packet.sequence,
                    packet.rssi,
                    packet.rx_cwb ? 40 : 20,
                    packet.source_mac[0], packet.source_mac[1], packet.source_mac[2],
                    packet.source_mac[3], packet.source_mac[4], packet.source_mac[5],
                    packet.csi_length_bytes
                );
                for (int i = 0; i < 8; i++) {
                    Serial.printf(",%d,%d", packet.csi_data[i * 2], packet.csi_data[i * 2 + 1]);
                }
                Serial.println();
            }

            digitalWrite(LED_PIN, LOW);
        }
    }
}

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.setTxBufferSize(4096);
    Serial.begin(CSI_UART_BAUD_RATE);
    while (!Serial && millis() < 3000);
    delay(1500);   // ホストがポートを開くのを待つ

    // Initial log
    Serial.println("\n[+] Initializing ESP32 Standalone 2.4GHz HT40 CSI Scanner & Radar...");

    // 1. Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // 2. Wi-Fi Initialization in Station Mode + Promiscuous
    esp_netif_init();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    CHK(esp_wifi_init(&cfg));
    CHK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    CHK(esp_wifi_set_mode(WIFI_MODE_STA));
    CHK(esp_wifi_start());

    // 3. Configure 2.4GHz HT40 Bandwidth
    CHK(esp_wifi_set_channel(g_current_channel, WIFI_SECOND_CHAN_NONE));
    CHK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    CHK(esp_wifi_set_ps(WIFI_PS_NONE));

    // Get eFuse STA MAC and update probe template
    esp_wifi_get_mac(WIFI_IF_STA, g_self_mac);
    memcpy(&g_probe_frame[10], g_self_mac, 6);
    Serial.printf("[*] Device MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  g_self_mac[0], g_self_mac[1], g_self_mac[2],
                  g_self_mac[3], g_self_mac[4], g_self_mac[5]);

    // 4. Configure Wi-Fi CSI Hardware Engine
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = false,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0
    };

    // 5. Enable Promiscuous Mode to capture all ambient Wi-Fi packets on channel
    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_ALL
    };
    esp_wifi_set_promiscuous_filter(&filter);
    esp_wifi_set_promiscuous_rx_cb(&promisc_cb);
    CHK(esp_wifi_set_promiscuous(true));
    // C3: esp-csi の例と同じく、プロミスキャスを有効にしてから CSI を設定する
    CHK(esp_wifi_set_csi_config(&csi_config));
    CHK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_callback, NULL));
    CHK(esp_wifi_set_csi(true));

    // 6. Create Queue and Tasks
    g_csi_queue = xQueueCreate(64, sizeof(tracking_packet_t));
    xTaskCreatePinnedToCore(uart_dispatch_task, "uart_dispatch", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(active_radar_probe_task, "radar_probe", 4096, NULL, 3, NULL, 0);

    Serial.printf("[+] Standalone Scanner Ready on 2.4GHz Channel %u (HT40).\n", g_current_channel);
    Serial.println("[+] Active Radar Probe Injection: ACTIVE (25 Hz)");
    Serial.println("[+] Binary 289-byte Stream (0xC510C510) ACTIVE at 921600 bps.");
}

void loop() {
    // Check for interactive Serial commands
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.startsWith("C") || cmd.startsWith("c")) {
            int ch = cmd.substring(1).toInt();
            if (ch >= 1 && ch <= 13) {
                g_current_channel = (uint8_t)ch;
                wifi_second_chan_t sec = (ch <= 7) ? WIFI_SECOND_CHAN_ABOVE : WIFI_SECOND_CHAN_BELOW;
                esp_wifi_set_channel(g_current_channel, sec);
                Serial.printf("[CMD] Channel switched to %d (HT40)\n", g_current_channel);
            }
        } else if (cmd.equalsIgnoreCase("MODE")) {
            g_binary_mode = !g_binary_mode;
            Serial.printf("[CMD] Output format: %s\n", g_binary_mode ? "BINARY (0xC510C510)" : "CSV TEXT");
        } else if (cmd.startsWith("M") || cmd.startsWith("m")) {
            unsigned int b[6];
            if (sscanf(cmd.c_str() + 1, " %x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) == 6) {
                for (int i = 0; i < 6; i++) g_mac_filter[i] = (uint8_t)b[i];
                g_mac_filter_en = true;
                Serial.printf("[CMD] MAC filter: %02X:%02X:%02X:%02X:%02X:%02X\n", b[0], b[1], b[2], b[3], b[4], b[5]);
            } else {
                g_mac_filter_en = false;
                Serial.println("[CMD] MAC filter: OFF");
            }
        } else if (cmd.startsWith("R") && cmd.length() > 1 && isDigit(cmd[1])) {
            g_rate_limit_ms = (uint32_t)cmd.substring(1).toInt();
            Serial.printf("[CMD] Rate limit: %u ms\n", (unsigned)g_rate_limit_ms);
        } else if (cmd.equalsIgnoreCase("B20") || cmd.equalsIgnoreCase("B40")) {
            esp_err_t e = esp_wifi_set_bandwidth(WIFI_IF_STA, cmd.endsWith("20") ? WIFI_BW_HT20 : WIFI_BW_HT40);
            Serial.printf("[CMD] Bandwidth %s: %s\n", cmd.c_str(), esp_err_to_name(e));
        } else if (cmd.equalsIgnoreCase("STAT")) {
            g_stat = !g_stat;
        } else if (cmd.equalsIgnoreCase("RADAR")) {
            g_active_probe_enabled = !g_active_probe_enabled;
            Serial.printf("[CMD] Active Radar Pulse: %s\n", g_active_probe_enabled ? "ENABLED" : "DISABLED");
        }
    }
    static uint32_t last = 0;
    if (g_stat && millis() - last > 1000) { last = millis(); Serial.printf("[STAT] ofdm %u promisc %u csi_cb %u short %u sent %u\n", (unsigned)g_promisc_ofdm, (unsigned)g_promisc_rx, (unsigned)g_cb_all, (unsigned)g_cb_short, (unsigned)g_cb_sent); }
    delay(50);
}
