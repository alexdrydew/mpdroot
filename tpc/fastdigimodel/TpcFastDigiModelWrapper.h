#ifndef TPCFASTDIGIMODELWRAPPER_HH
#define TPCFASTDIGIMODELWRAPPER_HH

#include <TString.h>
#include <onnxruntime_cxx_api.h>


class TpcFastDigiModelWrapper {
public:
   virtual void init() = 0;
   virtual int getBatchSize() = 0;
   virtual int modelRun(float *input, float *output, size_t input_size, size_t output_size) = 0;
   virtual ~TpcFastDigiModelWrapper() = default;
};


class ONNXTpcFastDigiModelWrapper : public TpcFastDigiModelWrapper {
protected:
   int numThreads;
   Ort::Session *session;
   Ort::MemoryInfo memoryInfo;
   std::vector<const char *> inputNames;
   std::vector<const char *> outputNames;

   std::vector<int64_t> inputDims;
   std::vector<int64_t> outputDims;
   void initSession(std::vector<char>& onnxModel);

public:
   ONNXTpcFastDigiModelWrapper(int numThreads);
   virtual ~ONNXTpcFastDigiModelWrapper();

   int getBatchSize() override { return 1; }
   int modelRun(float *input, float *output, size_t input_size, size_t output_size) override;
};


class LocalONNXTpcFastDigiModelWrapper : public ONNXTpcFastDigiModelWrapper {
private:
   TString onnxFilePath;

public:
   LocalONNXTpcFastDigiModelWrapper(int numThreads, const TString &onnxFilePath);
   void init() override;
};


class RemoteONNXTpcFastDigiModelWrapper : public ONNXTpcFastDigiModelWrapper {
private:
   TString mlflowHost;
   int     mlflowPort;
   TString s3Host;
   int     s3Port;
   TString modelName;
   int     modelVersion;
public:
   RemoteONNXTpcFastDigiModelWrapper(int numThreads, const TString &mlflowHost, int mlflowPort, const TString &s3Host, int s3Port,
                          const TString &modelName, int modelVersion = -1);
   void init() override;
};

class XLATpcFastDigiModelWrapper : public TpcFastDigiModelWrapper {
private:
   int numThreads;
public:
   XLATpcFastDigiModelWrapper(int numThreads);
   void init() override;
   int getBatchSize() override;
   int modelRun(float *input, float *output, size_t input_size, size_t output_size) override;
   virtual ~XLATpcFastDigiModelWrapper();
};

#endif
