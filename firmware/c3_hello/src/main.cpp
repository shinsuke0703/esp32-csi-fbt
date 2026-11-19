#include <Arduino.h>
void setup() { Serial.begin(115200); }
void loop() { Serial.printf("hello %lu\n", millis()); delay(500); }
