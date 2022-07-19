// This Class' Header ------------------
#include "MpdTpcFastDigitizer6.h"

// C/C++ Headers ----------------------
#include <vector>

// Class Member definitions -----------

using namespace std;

//---------------------------------------------------------------------------

vector<float> MpdTpcFastDigitizer6::prepareModelInput() const
{
   return {static_cast<float>(modelInputParameters.cross),
           static_cast<float>(modelInputParameters.dip),
           static_cast<float>(modelInputParameters.tbin),
           static_cast<float>(modelInputParameters.pad0),
           static_cast<float>(modelInputParameters.row0),
           static_cast<float>(modelInputParameters.mom3.Pt())};
}

//---------------------------------------------------------------------------

ClassImp(MpdTpcFastDigitizer6)
