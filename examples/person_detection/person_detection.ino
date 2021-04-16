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

#include <TensorFlowLite.h>

#include "helper.h"
#include "image_provider.h"
#include "person_detect_model_data.h"


/**
 * Had to include a namespace to prevent err's
 */
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

/**
 * Tensor Area
 */
constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

char chr;
bool status;

/**
 * All arduino codes need a 
 * setup() and loop()
 */
void setup() {
  /* Tensorflow logger */
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  /**
   * Location to the model which is trianed of a powerful computer
   * and stored as bytes under <person_detect_model_data.cpp>
   */
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);

    return;
  }

  /**
   * Neural Network in micro-controller domain. 
   * This can be replaced to whatever model from tensorflow
   * model garden : https://github.com/tensorflow/models
   * Simple model used here
   */
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  /**
   * https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
   */
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  /**
   * Allocate memory 
   */
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  chr = 'n';
  status = true;

  Serial.begin(9600);
}


void loop() {

  if (Serial.available()) {

      while(status) {

        chr = Serial.read();
          switch(chr) {
            
            case 'a': /* Acknowledge to isConnected() API*/
                    if (Serial.available()) {
                      TF_LITE_REPORT_ERROR(error_reporter, "acknowledge");
                    }
                    status = true;
                    break;

            case 's': /* Display Output Once to executeSingle() API */
              if (Serial.available()) {
                
                    // Get image from provider.
                    if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                                              input->data.int8)) {
                      TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
                    }
                  
                    // Run the model on this input and make sure it succeeds.
                    if (kTfLiteOk != interpreter->Invoke()) {
                      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
                    }
                  
                    TfLiteTensor* output = interpreter->output(0);
                  
                    // Process the inference results.
                    int8_t person_score = output->data.uint8[kPersonIndex];
                    int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
                    RespondToDetection(error_reporter, person_score, no_person_score);
                    
              }
              status = true;
              break;  

            case 'c': /* Display Output Continniously to executeContinuous() API */
              while(chr != 'x') {
                  if (Serial.available()) {
                    
                        // Get image from provider.
                        if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                                                  input->data.int8)) {
                          TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
                        }
                      
                        // Run the model on this input and make sure it succeeds.
                        if (kTfLiteOk != interpreter->Invoke()) {
                          TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
                        }
                      
                        TfLiteTensor* output = interpreter->output(0);
                      
                        // Process the inference results.
                        int8_t person_score = output->data.uint8[kPersonIndex];
                        int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
                        RespondToDetection(error_reporter, person_score, no_person_score);
    
                        chr = Serial.read();
                        if(chr == 'x') {
                          status = false;
                          break;
                        }
                        
                    } /* End of if (Serial.available()) */ 
                } /* End of continious output of while loop */

            case 'x': /* Stop Program and return */
                if (Serial.available()) {
                  TF_LITE_REPORT_ERROR(error_reporter, "acknowledge");
                }
                status = false;
                return;
          } /* End of switch case */

      } /* End of while loop */
      
  } /* End of if (Serial.available())  */
    
} /* End of loop */
