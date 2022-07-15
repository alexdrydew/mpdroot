//-----------------------------------------------------------
//
// Description:
//      Implementation of class MpdTpcFastDigitizer
//      see MpdTpcFastDigitizer.h for details
//
// Author List:
//      Alexander Zinchenko LHEP, JINR, Dubna - 03-December-2021
//      (modified version of MpdTpcDigitizerAZlt.cxx)                           
//
//-----------------------------------------------------------

// This Class' Header ------------------
#include "MpdTpcFastDigitizer6.h"

// MPD Headers ----------------------
#include "MpdTpcDigit.h"
#include "MpdMultiField.h"
#include "MpdTpcSector.h"
#include "MpdTpcSectorGeo.h"
#include "TpcGas.h"
#include "TpcPoint.h"
#include "fastdigimodel/ONNXRuntimeTpcFastDigiModelWrapper.h"

// FAIR Headers ----------------------
#include "FairRunAna.h"
#include "FairEventHeader.h"
#include "FairRootManager.h"
#include "FairRunSim.h"

// ROOT Headers ----------------------
#include "TClonesArray.h"
#include <TGeoManager.h>
#include "TLorentzVector.h"
#include "TRandom.h"
#include <TRefArray.h>
#include "TMath.h"
#include "TSystem.h"
#include "TaskHelpers.h"
#include <TVirtualFFT.h>
#include <TFile.h>
// C/C++ Headers ----------------------
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

// Class Member definitions -----------

using namespace std;
using namespace TMath;

static Int_t nOverlapDigit;
static Int_t nAllLightedDigits;

static clock_t tStart = 0;
static clock_t tFinish = 0;
static clock_t tAll = 0;

//FILE *lunAZ = nullptr; //fopen("gasGain.dat","w");
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
