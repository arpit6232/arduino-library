#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "helper.h"
#include "Arduino.h"

const char* kCategoryLabels[kCategoryCount] = {
    "notperson",
    "person",
};

/**
 * This section will be modified in future sprints to respond
 * back to the Jetson Nano, rather than detect on the ArduinoBLE
 * 
 * Sprint 1: Flash LED on the Arduino to detect a person 
 * Person Detected -> Blue
 * No Person Detected -> Red
 * 
 * -> Based on the pinout : https://content.arduino.cc/assets/NANO33BLE_V2.0_sch.pdf
 */
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t person_score, int8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  // Switch the person/not person LEDs off
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(LEDB, LOW);
  delay(100);
  digitalWrite(LEDB, HIGH);

  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (person_score > no_person_score) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Person score: %d No person score: %d",
                       person_score, no_person_score);
}

#endif  // ARDUINO_EXCLUDE_CODE
