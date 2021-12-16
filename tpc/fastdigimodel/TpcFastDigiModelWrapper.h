#ifndef TPCFASTDIGIMODELWRAPPER_HH
#define TPCFASTDIGIMODELWRAPPER_HH

namespace TpcFastDigiModelWrapper {
  int get_batch_size();
  void model_init(int num_threads = 0);
  int model_run(float *input, float *output, int input_size, int output_size);
  void model_free();
}

#endif
