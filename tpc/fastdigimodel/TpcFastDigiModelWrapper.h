#ifndef TPCFASTDIGIMODELWRAPPER_HH
#define TPCFASTDIGIMODELWRAPPER_HH

#include <TString.h>
#include <onnxruntime_cxx_api.h>


class ONNXRuntimeTpcFastDigiModelWrapper {
private:
    Ort::Session* session;
    Ort::MemoryInfo memoryInfo;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;

    void printInfo();

public:
    ONNXRuntimeTpcFastDigiModelWrapper(int num_threads, TString modelVersion);
    ~ONNXRuntimeTpcFastDigiModelWrapper();

    int get_batch_size() { return 1; }
    int model_run(float *input, float *output, int input_size, int output_size);
};

#endif
