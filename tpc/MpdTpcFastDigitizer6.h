#ifndef MPDROOT_MPDTPCFASTDIGITIZER6_H
#define MPDROOT_MPDTPCFASTDIGITIZER6_H

#include "MpdTpcFastDigitizer.h"
#include "fastdigimodel/ONNXRuntimeTpcFastDigiModelWrapper.h"

class MpdTpcFastDigitizer6: public MpdTpcFastDigitizer {
public:
   MpdTpcFastDigitizer6(ONNXRuntimeTpcFastDigiModelWrapper* onnxModelWrapper): MpdTpcFastDigitizer(onnxModelWrapper) { };

private:
   vector<float> prepareModelInput() const override;

   ClassDef(MpdTpcFastDigitizer6, 0)
};

#endif // MPDROOT_MPDTPCFASTDIGITIZER6_H
