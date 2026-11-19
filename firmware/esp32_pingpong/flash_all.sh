#!/bin/sh
# つながっている全台に書き込む (C3 = VID 303A、DevKitC = CP210x の VID 10C4)。COM の番号は毎回自動で調べる。失敗したら 2 回までやり直す
fl() { for i in 1 2 3; do pio run -e $1 -t upload --upload-port $2 > /tmp/fl_$2.log 2>&1 && { echo "$2 OK"; return; }; echo "$2 失敗 ($i 回目)"; sleep 2; done; tail -5 /tmp/fl_$2.log; }
C3=$(python -P -c "import serial.tools.list_ports as L; print(' '.join(p.device for p in L.comports() if '303A' in p.hwid))")
DK=$(python -P -c "import serial.tools.list_ports as L; print(' '.join(p.device for p in L.comports() if '10C4' in p.hwid))")
echo "C3: $C3 / DevKitC: $DK"
for p in $C3; do fl c3 $p; done
for p in $DK; do fl esp32dev $p; done
