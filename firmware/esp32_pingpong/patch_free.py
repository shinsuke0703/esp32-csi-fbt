# -*- coding: utf-8 -*-
"""「同時送信」方式を入れる: トークンなしで、送信役 0〜4 がそれぞれ自分の周期で PING を送り続ける (CSMA/CA に順番を任せる)。
速い方式の 1 手番を関数 fast_turn にまとめ、トークン方式 (next ≥ 0) と同時送信方式 (next = −1) で共用する。改行は CRLF のまま保つ。"""
p = 'src/main.cpp'
raw = open(p, 'rb').read().decode('utf-8'); crlf = '\r\n' in raw; s = raw.replace('\r\n', '\n')
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)

i0 = s.index("        if (!g_hop) {   // 速い方式")
i1 = s.index("            continue;\n        }\n", i0) + len("            continue;\n        }\n")
s = s[:i0] + """        if (!g_hop) {   // 速い方式
            fast_turn(nt, (int8_t)next, (g_id == 0) ? g_cycle : cyc);
            if (by_tok) g_m_wake += t_wake - g_tok_rx_us;
            g_m_pings += (uint32_t)esp_timer_get_time() - t_wake; g_m_n++;
            continue;
        }
""" + s[i1:]

rep("static void token_task(void *) {", """// 速い方式の 1 手番: 自分より番号の大きい全員へ reps 回ずつユニキャスト PING を続けて送り、next ≥ 0 ならその後にトークンを送る。
// 送信完了のコールバックは送った順に (まとめて遅れて) 届くので、i 番目のコールバック = i 番目に送ったもの。
// ACK (宛先が自分のものだけ) を受信時刻の順に並べ、個数がちょうど合うときだけ送った順に対応づける (合わなければ取り違えを避けて捨てる)。
static void fast_turn(int nt, int8_t next, uint32_t tokseq) {
    portENTER_CRITICAL(&g_mux); g_n_ev = 0; g_n_acks = 0; portEXIT_CRITICAL(&g_mux);
    g_pipe = true; g_ev_on = true; g_ack_wait_until = (uint32_t)esp_timer_get_time() + 20000;
    int reps = g_reps; if (reps < 1) reps = 1; if (nt * reps > PMAX - 1) reps = (PMAX - 1) / nt;
    g_nsend = nt * reps; g_cb_n = 0; g_tok_sent = false; g_tok_to = next; g_fast = true;
    int i = 0;
    for (int r = 0; r < reps; r++) for (int t = g_id + 1; t < 6; t++) {
        g_st[i] = (uint8_t)t; g_sseq[i] = ++g_seq; i++;
        Payload q = {'P', 'G', 4, (uint8_t)g_id, (uint8_t)t, g_seq};
        send_fc(NODE_MAC[t], &q, sizeof(q));
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
            if (rem > 1500) vTaskDelay(1); else delayMicroseconds(rem);
        }
        if (g_free_period > 0 && ((uint32_t)esp_timer_get_time() - t0) < 1500) vTaskDelay(1);   // 周期が短すぎても最低 1 tick は手放す
    }
}

static void token_task(void *) {""")
rep("    xTaskCreate(token_task, \"token\", 4096, NULL, configMAX_PRIORITIES - 3, NULL);",
    "    xTaskCreate(token_task, \"token\", 4096, NULL, configMAX_PRIORITIES - 3, NULL);\n"
    "    xTaskCreate(free_task, \"free\", 4096, NULL, configMAX_PRIORITIES - 3, NULL);")
rep('        else if (c.startsWith("N ")) g_reps = c.substring(2).toInt();',
    '        else if (c.startsWith("N ")) g_reps = c.substring(2).toInt();\n'
    '        else if (c.startsWith("FR ")) g_free_period = (uint32_t)c.substring(3).toInt();')
open(p, 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print("ok")
