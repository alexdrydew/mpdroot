// This Class' Header --------------------------
#include "ONNXRuntimeTpcFastDigiModelWrapper.h"

// ROOT Headers --------------------------------
#include "TRandom.h"
#include "TString.h"
#include "FairLogger.h"

// C/C++ Headers -------------------------------
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <exception>

// Boost Headers -------------------------------
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <string>

// ONNXRuntime Headers -------------------------
#include <onnxruntime_cxx_api.h>


namespace pt = boost::property_tree;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = boost::asio::ip::tcp;


template <typename T>
T vectorProduct(const std::vector<T>& v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

auto doGetRequest(const std::string& host, const std::string& port, const std::string& path) {
   boost::asio::io_context ioc;
   tcp::resolver resolver(ioc);
   tcp::socket socket(ioc);
   boost::asio::connect(socket, resolver.resolve(host, port));
   http::request<http::string_body> request(http::verb::get, path, 11);
   request.set(http::field::host, host);
   http::write(socket, request);
   boost::beast::flat_buffer buffer;
   http::response<http::string_body> response;
   http::read(socket, buffer, response);
   return response;
}

std::vector<char> downloadModel() {
   std::string mlflowUrl = std::string(std::getenv("MLFLOW_URL"));
   std::string minioUrl = std::string(std::getenv("MINIO_URL"));
   std::string modelName = std::string(std::getenv("ONNX_MODEL_NAME"));

   LOG(INFO) << "Getting last model version...";
   pt::ptree root;
   std::stringstream ss;
   auto getVersionPath = "/api/2.0/mlflow/registered-models/get-latest-versions?name=" + modelName;
   auto getVersionResponse = doGetRequest(mlflowUrl, "5000", getVersionPath);
   if (getVersionResponse.result() != http::status::ok) {
      throw std::runtime_error("Model wasn't found!");
   }
   ss << getVersionResponse.body();
   pt::read_json(ss, root);
   int lastVersion = -1;
   for (pt::ptree::value_type &versionDict : root.get_child("model_versions")) {
      int version = versionDict.second.get<int>("version");
      if (lastVersion < version) {
         lastVersion = version;
      }
   }
   std::cout << "\t" << modelName << " version: " << lastVersion << std::endl;

   LOG(INFO) << "Getting model download url..." << std::endl;
   auto getDownloadUriPath = "/api/2.0/preview/mlflow/model-versions/get-download-uri?name=" + modelName + "&version=" + std::to_string(lastVersion);
   auto response = doGetRequest(mlflowUrl, "5000", getDownloadUriPath);
   if (response.result() != http::status::ok) {
      throw std::runtime_error("Failed to retrieve model download url!");
   }
   ss << response.body();
   pt::read_json(ss, root);
   auto artifactUri = root.get<std::string>("artifact_uri");
   auto modelDownloadPath = artifactUri.substr(4) + "/model.onnx"; // s3://<path-to-model> -> /<path-to-model>
   LOG(INFO) << "Downloading ONNX model..." << std::endl;
   auto onnxModelResponse = doGetRequest(minioUrl, "9000", modelDownloadPath);
   if (onnxModelResponse.result() != http::status::ok) {
      throw std::runtime_error("Failed to download ONNX model!");
   }
   auto onnxModel = onnxModelResponse.body();
   return std::vector<char>(onnxModel.begin(), onnxModel.end());
}

ONNXRuntimeTpcFastDigiModelWrapper::ONNXRuntimeTpcFastDigiModelWrapper(int num_threads, TString modelVersion) :
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                          OrtMemType::OrtMemTypeDefault)) {

    LOG(INFO) << "Downloading TPC fast digitizer model" << std::endl;
    auto onnx = downloadModel();

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tpc digitizer");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(num_threads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // session = new Ort::Session(env, modelFilepath.c_str(), sessionOptions);
    session = new Ort::Session(env, static_cast<void*>(onnx.data()), onnx.size(), sessionOptions);

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

ONNXRuntimeTpcFastDigiModelWrapper::~ONNXRuntimeTpcFastDigiModelWrapper() {
    delete session;
}

void ONNXRuntimeTpcFastDigiModelWrapper::printInfo() {
    LOG(INFO) << "Initialized GAN neural network TPC fast digitizer. Implemented by HSE MPD TPC project group." << std::endl;
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
