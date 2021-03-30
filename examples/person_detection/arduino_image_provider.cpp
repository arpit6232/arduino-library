/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>

/**
 * Return the image cropped to appropriate size and quantized for processing
 * tflite::ErrorReporter* error_reporter: Logging used for tensorflow 
 * int image_width: Width of input image
 * int image_height: Height of the input image
 * int channels: Channels represent RGBD or Grayscale. All images here are 
 *                are converted to grayscale for easier processing and less 
 *                math. 
 * int8_t* image_data : Pointer to location of image data captured.
 */
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {

  byte data[176 * 144]; // Receiving QCIF grayscale from camera = 176 * 144 * 1

  static bool g_is_camera_initialized = false;
  static bool serial_is_initialized = false;

  // Initialize camera if necessary
  if (!g_is_camera_initialized) {
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

  // Read camera data
  Camera.readFrame(data);

  int min_x = (176 - 96) / 2;
  int min_y = (144 - 96) / 2;
  int index = 0;

  /**
   * Crop to 96x96 image and convert TF input image to signed 8-bit
   * The easiest and quickest way to convert from unsigned to
   * signed 8-bit integers is to subtract 128 from the unsigned value to get a
   * signed value.
   */
  for (int col_idx = min_y; col_idx < min_y + 96; col_idx++) {
    for (int row_idx = min_x; row_idx < min_x + 96; row_idx++) {
      image_data[index++] = static_cast<int8_t>(data[(col_idx * 176) + row_idx] - 128);
    }
  }

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE
