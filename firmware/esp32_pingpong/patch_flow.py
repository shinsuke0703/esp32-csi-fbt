# -*- coding: utf-8 -*-
"""速い方式の送信に流量制御を入れる: esp_now_send が断ったら (バッファがいっぱい)、少し待って送り直す。断られた回数を数える。"""
p = 'src/main.cpp'
raw = open(p, 'rb').read().decode('utf-8'); crlf = '\r\n' in raw; s = raw.replace('\r\n', '\n')
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:70])
    s = s.replace(a, b)
rep("static volatile int g_reps = 1;",
    "static volatile uint32_t g_n_nomem = 0, g_n_senderr = 0;\n"
    "static void send_fc(const uint8_t *mac, const void *d, int len) {   // 断られたら最大 5 ms まで待って送り直す\n"
    "    uint32_t t0 = (uint32_t)esp_timer_get_time();\n"
    "    while (true) {\n"
    "        esp_err_t e = esp_now_send(mac, (const uint8_t *)d, len);\n"
    "        if (e == ESP_OK) return;\n"
    "        if (e != ESP_ERR_ESPNOW_NO_MEM || (uint32_t)esp_timer_get_time() - t0 > 5000) { g_n_senderr++; return; }\n"
    "        g_n_nomem++; delayMicroseconds(50);\n"
    "    }\n"
    "}\n"
    "static volatile int g_reps = 1;")
rep("                esp_now_send(NODE_MAC[t], (const uint8_t *)&q, sizeof(q));\n            }\n            Payload p",
    "                send_fc(NODE_MAC[t], &q, sizeof(q));\n            }\n            Payload p")
rep("            esp_now_send(NODE_MAC[next], (const uint8_t *)&p, sizeof(p));   // 送った順に出るので",
    "            send_fc(NODE_MAC[next], &p, sizeof(p));   // 送った順に出るので")
open(p, 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print("ok")
