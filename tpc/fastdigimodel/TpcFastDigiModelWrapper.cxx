#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <cmath>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

#include "TpcFastDigiModelWrapper.h"
#include "model.h"


// from example
template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

OnnxruntimeTpcFastDigiModelWrapper::OnnxruntimeTpcFastDigiModelWrapper(int num_threads) :
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                          OrtMemType::OrtMemTypeDefault)) {
    std::string modelFilepath = std::string(__FILE__);
    modelFilepath = modelFilepath.substr(0, modelFilepath.rfind('/'));
    modelFilepath.append("/baseline.onnx");

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tpc digitizer");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(num_threads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = new Ort::Session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    const char* inputName = session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputDims = inputTensorInfo.GetShape();
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues = std::vector<float>(inputTensorSize, 0.);

    const char* outputName = session->GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    outputDims = outputTensorInfo.GetShape();
    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues = std::vector<float>(outputTensorSize);

    inputNames = {inputName};
    outputNames = {outputName};

    printInfo();
}

OnnxruntimeTpcFastDigiModelWrapper::~OnnxruntimeTpcFastDigiModelWrapper() {
    delete session;
}

void OnnxruntimeTpcFastDigiModelWrapper::printInfo() {
    std::cout << "Initialized GAN neural network TPC fast digitizer. Implemented by HSE MPD TPC project group.\n";
}

int OnnxruntimeTpcFastDigiModelWrapper::model_run(float *input, float *output, int input_size, int output_size) {
    return ::model_run(input, output, input_size, output_size);


    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, input, input_size, inputDims.data(),
            inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, output, output_size,
            outputDims.data(), outputDims.size()));

    //Run: Running the session is done in the Run() method:
    session->Run(Ort::RunOptions{nullptr},
                inputNames.data(), inputTensors.data(), 1,
                outputNames.data(), outputTensors.data(), 1);
}
