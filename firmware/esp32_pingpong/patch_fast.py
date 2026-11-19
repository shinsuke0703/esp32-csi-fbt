# -*- coding: utf-8 -*-
"""main.cpp に「速い方式」(PING に続けてすぐトークン、同じ相手に N 回) を入れる一回限りのパッチ (改行は CRLF のまま保つ)。"""
p = 'src/main.cpp'
raw = open(p, 'rb').read().decode('utf-8'); crlf = '\r\n' in raw; s = raw.replace('\r\n', '\n')

def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)

rep("#define RING 16", "#define RING 32")
rep("#define PMAX 8", "#define PMAX 32")
rep("static uint32_t g_tseq[6];",
    "static uint32_t g_tseq[6];\n"
    "// 速い方式 (ホッピングなし): PING に続けてすぐトークンを送る。送信完了のコールバックはまとめて遅れて届くので待たない。\n"
    "// コールバックは送った順に届くので、i 番目のコールバック = i 番目に送ったもの (最後の 1 つがトークン)。\n"
    "static volatile bool g_fast = false; static volatile int g_cb_n = 0, g_nsend = 0; static uint8_t g_st[PMAX]; static uint32_t g_sseq[PMAX];\n"
    "static volatile int g_reps = 1;   // 1 回の手番で同じ相手に何回送るか (\"N <回数>\" で変更)")
rep("static void on_sent(const uint8_t *mac, esp_now_send_status_t st) {   // ユニキャストは ACK を受けた (または諦めた) 直後に呼ばれる\n",
    "static void on_sent(const uint8_t *mac, esp_now_send_status_t st) {   // ユニキャストは ACK を受けた (または諦めた) 直後に呼ばれる\n"
    "    if (g_fast) {\n"
    "        int i = g_cb_n++;\n"
    "        if (i < g_nsend) { portENTER_CRITICAL(&g_mux); if (g_n_ev < PMAX) { g_ev[g_n_ev] = {g_st[i], g_sseq[i], (uint32_t)esp_timer_get_time(), st == ESP_NOW_SEND_SUCCESS}; g_n_ev++; } portEXIT_CRITICAL(&g_mux); }\n"
    "        else if (i == g_nsend) { g_tok_sent = true; if (st != ESP_NOW_SEND_SUCCESS) g_n_tok_fail++; }\n"
    "        return;\n"
    "    }\n")
rep("""        portENTER_CRITICAL(&g_mux); g_n_ev = 0; g_n_acks = 0; portEXIT_CRITICAL(&g_mux);
        g_pipe = true; g_ev_on = true; g_ack_wait_until = (uint32_t)esp_timer_get_time() + 20000;
        for (int t = g_id + 1; t < 6; t++) {""",
"""        portENTER_CRITICAL(&g_mux); g_n_ev = 0; g_n_acks = 0; portEXIT_CRITICAL(&g_mux);
        g_pipe = true; g_ev_on = true; g_ack_wait_until = (uint32_t)esp_timer_get_time() + 20000;
        uint8_t next = (g_id >= LAST_SENDER) ? 0 : (uint8_t)(g_id + 1);
        if (!g_hop) {   // 速い方式
            int reps = g_reps; if (reps < 1) reps = 1; if (nt * reps > PMAX - 1) reps = (PMAX - 1) / nt;
            g_nsend = nt * reps; g_cb_n = 0; g_tok_sent = false; g_tok_to = next; g_fast = true;
            int i = 0;
            for (int r = 0; r < reps; r++) for (int t = g_id + 1; t < 6; t++) {
                g_st[i] = (uint8_t)t; g_sseq[i] = ++g_seq; i++;
                Payload q = {'P', 'G', 4, (uint8_t)g_id, (uint8_t)t, g_seq};
                esp_now_send(NODE_MAC[t], (const uint8_t *)&q, sizeof(q));
            }
            Payload p = {'P', 'G', 6, (uint8_t)g_id, next, (g_id == 0) ? g_cycle : cyc};
            esp_now_send(NODE_MAC[next], (const uint8_t *)&p, sizeof(p));   // 送った順に出るので、トークンは全 PING の後
            uint32_t t1 = (uint32_t)esp_timer_get_time();
            while (g_cb_n < g_nsend + 1 && (uint32_t)esp_timer_get_time() - t1 < 15000) delayMicroseconds(20);   // 全部のコールバック
            delayMicroseconds(300);   // 遅れて届く ACK の CSI
            g_ack_wait_until = 0; g_pipe = false; g_fast = false; g_ev_on = false;
            if (by_tok) g_m_wake += t_wake - g_tok_rx_us;
            g_m_pings += (uint32_t)esp_timer_get_time() - t_wake; g_m_n++;
            SendEv ev[PMAX]; AckC *ac = g_acks; int ne, na;
            portENTER_CRITICAL(&g_mux); ne = g_n_ev; na = g_n_acks; for (int k = 0; k < ne; k++) ev[k] = g_ev[k]; portEXIT_CRITICAL(&g_mux);
            for (int a = 1; a < na; a++) for (int b = a; b > 0 && (int32_t)(ac[b].ts - ac[b - 1].ts) < 0; b--) { AckC x = ac[b]; ac[b] = ac[b - 1]; ac[b - 1] = x; }
            int nok = 0; for (int k = 0; k < ne; k++) nok += ev[k].ok;
            // トークンの ACK が最後に 1 つ多い (= 全部そろった) ときだけ、送った順に対応づける。それ以外は取り違えを避けて捨てる
            if (g_tok_sent && na == nok + 1 && ne == g_nsend) {
                int k = 0;
                for (int e = 0; e < ne; e++) if (ev[e].ok) {
                    Record &r = ac[k].r; r.tx_id = ev[e].t; r.type = 5; r.target = ev[e].t; r.seq = ev[e].seq; r.t_aux_us = 0;
                    if (xQueueSend(g_out_q, &r, 0) != pdTRUE) g_drop++;
                    k++; g_n_pipe_match++;
                }
            } else g_n_pipe_unmatched++;
            g_tok_to = -1;
            continue;
        }
        for (int t = g_id + 1; t < 6; t++) {""")
rep("""        uint8_t next = (g_id >= LAST_SENDER) ? 0 : (uint8_t)(g_id + 1);
        Payload p = {'P', 'G', 6, (uint8_t)g_id, next, (g_id == 0) ? g_cycle : cyc};""",
    """        Payload p = {'P', 'G', 6, (uint8_t)g_id, next, (g_id == 0) ? g_cycle : cyc};""")
rep('        else if (c == "SH")', '        else if (c.startsWith("N ")) g_reps = c.substring(2).toInt();\n        else if (c == "SH")')
open(p, 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print("ok")
