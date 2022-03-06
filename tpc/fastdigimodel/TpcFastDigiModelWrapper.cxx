// This Class' Header ----------------
#include "TpcFastDigiModelWrapper.h"

// ROOT Headers ----------------------
#include "TRandom.h"
#include <TString.h>

// C/C++ Headers ---------------------
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <cstdlib>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include "curl/curl.h"
#include <string>

// ONNXRuntime Headers ---------------
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>


template <typename T>
T vectorProduct(const std::vector<T>& v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

size_t writeCallback(void* ptr, size_t size, size_t nmemb, std::string* data) {
   data->append((char*) ptr, size * nmemb);
   return size * nmemb;
}

std::vector<char> downloadModel() {
   curl_global_init(CURL_GLOBAL_DEFAULT);
   std::string onnx;

   std::string mlflowUrl = std::string(std::getenv("MLFLOW_URL"));
   std::string minioUrl = std::string(std::getenv("MINIO_URL"));
   std::string modelName = std::string(std::getenv("ONNX_MODEL_NAME"));

   std::cout << "[INFO] Getting last model version..." << std::endl;
   auto versionsRequest = "http://www.example.com";// mlflowUrl + "/api/2.0/mlflow/registered-models/get-latest-versions?name=" + modelName;
   std::cout << versionsRequest << std::endl;
   std::string versionsResponse;

   auto curl = curl_easy_init();
   curl_easy_setopt(curl, CURLOPT_URL, versionsRequest);
   curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
   curl_easy_setopt(curl, CURLOPT_WRITEDATA, &versionsResponse);
   std::cout << curl_easy_perform(curl) << std::endl;
   curl_easy_cleanup(curl);

   std::cout << versionsResponse << std::endl;
   pt::ptree root;
   std::stringstream ss;
   ss << versionsResponse;
   pt::read_json(ss, root);
   int lastVersion = -1;
   std::string lastVersionUrl;
   for (pt::ptree::value_type &versionDict : root.get_child("model_versions")) {
      int version = versionDict.second.get<int>("version");
      if (lastVersion < version) {
         lastVersion = version;
      }
   }

//   std::cout << "[INFO] Getting model download url..." << std::endl;
//   curl_easy_setopt(curl, CURLOPT_URL,
//                    mlflowUrl + "/api/2.0/preview/mlflow/model-versions/get-download-uri?name" + modelName + "?version=" + std::to_string(lastVersion));
//
//   std::string artifactResponse;
//   curl = curl_easy_init();
//   curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
//   curl_easy_setopt(curl, CURLOPT_WRITEDATA, &artifactResponse);
//   curl_easy_perform(curl);
//   curl_easy_cleanup(curl);
//
//
//   ss << artifactResponse;
//   pt::read_json(ss, root);
//   std::string artifactUri = root.get<std::string>("artifact_uri");
//
//   auto artifactUrl = minioUrl + artifactUri.substr(5) + "/model.onnx"; // remove "s3://" prefix
//
//   std::cout << "[INFO] Downloading model..." << std::endl;
//   curl = curl_easy_init();
//   curl_easy_setopt(curl, CURLOPT_URL, artifactUrl);
//   curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
//   curl_easy_setopt(curl, CURLOPT_WRITEDATA, &onnx);
//   curl_easy_perform(curl);
//
//   curl_easy_cleanup(curl);
   curl_global_cleanup();

   return std::vector<char>(onnx.begin(), onnx.end());
}

ONNXRuntimeTpcFastDigiModelWrapper::ONNXRuntimeTpcFastDigiModelWrapper(int num_threads, TString modelVersion) :
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                          OrtMemType::OrtMemTypeDefault)) {

    std::cout << "[INFO] Downloading model" << std::endl;
    auto onnx = downloadModel();
    std::cout << "[DEBUG] Downloaded model of size " << onnx.size() * sizeof(onnx[0]) << std::endl;

//    std::string modelFilepath = std::string(__FILE__);
//    modelFilepath = modelFilepath.substr(0, modelFilepath.rfind('/'));
//    modelFilepath.append("/baseline.onnx");

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tpc digitizer");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(num_threads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // session = new Ort::Session(env, modelFilepath.c_str(), sessionOptions);
    session = new Ort::Session(env, static_cast<void*>(onnx.data()), onnx.size(), sessionOptions);
    std::cout << "[DEBUG] ONNXRuntime session created" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
   std::cout << "DEBUG1" << std::endl;

    const char* inputName = session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputDims = inputTensorInfo.GetShape();
    size_t inputTensorSize = vectorProduct(inputDims);
    std::cout << inputTensorSize << std::endl;
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

ONNXRuntimeTpcFastDigiModelWrapper::~ONNXRuntimeTpcFastDigiModelWrapper() {
    delete session;
}

void ONNXRuntimeTpcFastDigiModelWrapper::printInfo() {
    std::cout << "[INFO] Initialized GAN neural network TPC fast digitizer. Implemented by HSE MPD TPC project group." << std::endl;
}

int ONNXRuntimeTpcFastDigiModelWrapper::model_run(float *input, float *output, int input_size, int output_size) {
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, input, input_size, inputDims.data(),
            inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, output, output_size,
            outputDims.data(), outputDims.size()));

    session->Run(Ort::RunOptions{nullptr},
                inputNames.data(), inputTensors.data(), 1,
                outputNames.data(), outputTensors.data(), 1);
}
