#ifndef MPDROOT_MPDTPCFASTDIGITIZER6_H
#define MPDROOT_MPDTPCFASTDIGITIZER6_H

#include "MpdTpcFastDigitizer.h"
#include "fastdigimodel/TpcFastDigiModelWrapper.h"

class MpdTpcFastDigitizer6: public MpdTpcFastDigitizer {
public:
   MpdTpcFastDigitizer6(TpcFastDigiModelWrapper *onnxModelWrapper): MpdTpcFastDigitizer(onnxModelWrapper) { };

private:
   vector<float> prepareModelInput() const override;

   ClassDefOverride(MpdTpcFastDigitizer6, 0)
};

#endif // MPDROOT_MPDTPCFASTDIGITIZER6_H
