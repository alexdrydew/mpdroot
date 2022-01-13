#ifndef TPCFASTDIGIMODELWRAPPER_HH
#define TPCFASTDIGIMODELWRAPPER_HH


#include <onnxruntime/core/session/onnxruntime_cxx_api.h>


class OnnxruntimeTpcFastDigiModelWrapper {
private:
    Ort::Session* session;
    Ort::MemoryInfo memoryInfo;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;

    void printInfo();

public:
    OnnxruntimeTpcFastDigiModelWrapper(int num_threads);
    ~OnnxruntimeTpcFastDigiModelWrapper();

    int get_batch_size() { return 1; }
    int model_run(float *input, float *output, int input_size, int output_size);
};

#endif
