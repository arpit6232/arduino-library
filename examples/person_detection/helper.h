#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


/**
 * Camera Parameters
 */
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;
constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;
constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
extern const char* kCategoryLabels[kCategoryCount];


// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void setup();

void loop();

#ifdef __cplusplus
}
#endif


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
                        int8_t person_score, int8_t no_person_score);
