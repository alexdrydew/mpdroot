#ifndef TPCFASTDIGIMODELWRAPPER_HH
#define TPCFASTDIGIMODELWRAPPER_HH

#include <TString.h>
#include <onnxruntime_cxx_api.h>

class ONNXRuntimeTpcFastDigiModelWrapper {
private:
   Ort::Session *  session;
   Ort::MemoryInfo memoryInfo;

   std::vector<const char *> inputNames;
   std::vector<const char *> outputNames;

   std::vector<int64_t> inputDims;
   std::vector<int64_t> outputDims;

   int     numThreads;
   TString mlflowHost;
   int     mlflowPort;
   TString s3Host;
   int     s3Port;
   TString modelName;
   int     modelVersion;

   TString onnxFilePath;

public:
   ONNXRuntimeTpcFastDigiModelWrapper(int numThreads, const TString &mlflowHost, int mlflowPort, const TString &s3Host,
                                      int s3Port, const TString &modelName, int modelVersion = -1);
   ONNXRuntimeTpcFastDigiModelWrapper(int numThreads, const TString &onnxFilePath);
   ~ONNXRuntimeTpcFastDigiModelWrapper();

   void init();

   int getBatchSize() { return 1; }
   int modelRun(float *input, float *output, size_t input_size, size_t output_size);
};

#endif
