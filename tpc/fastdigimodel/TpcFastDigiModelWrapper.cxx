//#include <onnxruntime_cxx_api.h>

#include "TpcFastDigiModelWrapper.h"
#include "model.h"


int TpcFastDigiModelWrapper::get_batch_size() {
  return ::get_batch_size();
}

void TpcFastDigiModelWrapper::model_init(int num_threads) {
  ::model_init(num_threads);
}

int TpcFastDigiModelWrapper::model_run(float *input, float *output, int input_size, int output_size) {
  return ::model_run(input, output, input_size, output_size);
}

void TpcFastDigiModelWrapper::model_free() {
  ::model_free();
}

