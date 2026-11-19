import serial, time, sys
s = serial.Serial(port=None, baudrate=115200, timeout=0.1); s.dtr = True; s.rts = False; s.port = sys.argv[1]; s.open(); time.sleep(2)
s.write(b"S\n"); time.sleep(0.5); s.write(b"S\n"); time.sleep(0.5); print(s.read(4096).decode("utf-8", "replace")); s.close()
