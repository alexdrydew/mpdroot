//--------------------------------------------------------------------
//
// Description:
//      Tpc Digitizer reads array of MC points and produces TpcDigits
//
//
// Author List:
//      Alexander Zinchenko LHEP, JINR, Dubna - 93-December-2021
//      (modified version of MpdTpcDigitizerAZlt.h)                             
//
//--------------------------------------------------------------------

#ifndef MPDTPCFASTDIGITIZER_HH
#define MPDTPCFASTDIGITIZER_HH

// Base Class Headers ----------------

#include "TpcPoint.h"
#include "MpdTpcHit.h"
#include "MpdTpcDigitizerTask.h"
#include "MpdTpcDigitizerQAHistograms.h"
#include "FairTask.h"
#include "MpdMCTrack.h"
#include "FairField.h"

class TpcGas;
class MpdTpcSectorGeo;

using namespace std;

/*
struct DigOrigArray {
    Float_t signal;
    map<Int_t, Float_t> origins; // array of pairs <MC-track ID, its signal> for calculating origin
    Int_t origin; // MC-track index for testing
    Bool_t isOverlap;
};
*/

class MpdTpcFastDigitizer : public FairTask {
public:

    // Constructors/Destructors ---------
    MpdTpcFastDigitizer();
    virtual ~MpdTpcFastDigitizer();

    Bool_t isSubtrackInInwards(const TpcPoint *p1, const TpcPoint *p2);

    void SetPrimBranchName(const TString& name) { fInputBranchName = name; }
    void SetPersistence(Bool_t opt = kTRUE) { fPersistence = opt; }
    void SetAttach(Bool_t opt = kTRUE) { fAttach = opt; }
    void SetDiffuse(Bool_t opt = kTRUE) { fDiffuse = opt; }
    void SetDistort(Bool_t opt = kTRUE) { fDistort = opt; }
    void SetDebug(Bool_t opt = kTRUE) { fPrintDebugInfo = opt; }
    void SetMakeQA(Bool_t opt = kFALSE) { fMakeQA = opt; }
    void SetDistribute(Bool_t opt = kFALSE) { fDistribute = opt; }
    void SetResponse(Bool_t opt = kFALSE) { fResponse = opt; }
    void SetOnlyPrimary(Bool_t opt = kFALSE) { fOnlyPrimary = opt; }
    void SetNoiseThr(Double_t thresh) { fNoiseThreshold = thresh; }
    void SetOverflow(Double_t overfl) { fOverflow = overfl; }

    virtual InitStatus Init();
    virtual void Exec(Option_t* opt);
    virtual void Finish();
    
private:
    void PadResponse(Float_t x, Float_t y, UInt_t timeID, Int_t origin, DigOrigArray ***arr);
    TF1* padResponseFunction();
    Float_t CalculatePadResponse(Int_t iloop, Int_t nLoop, UInt_t padID, UInt_t rowID, Float_t x, Float_t y);
    void GetArea(Float_t xEll, Float_t yEll, Float_t radius, vector<UInt_t> &padIDs, vector<UInt_t> &rowIDs);
    //AZ Int_t CalcOrigin(const DigOrigArray dig);
    Int_t CalcOrigin(DigOrigArray& dig);
    void TpcProcessing(const TpcPoint* prePoint, const TpcPoint* curPoint, const UInt_t secID, const UInt_t iPoint, const UInt_t nPoints);
    void Check4Edge(UInt_t iSec, TpcPoint* &prePoint, TpcPoint* virtPoint); // check for edge-effect
    Double_t Polya(); // gas gain fluctuations
    void SignalShaping(); // electronics response
    void FastDigi(int isec, const MpdTpcHit* curHit); // interface to fast digitizer

private:

    // Private Data Members ------------
    TString fInputBranchName;
    TString fOutputBranchName;
    
    TClonesArray* fMCPointArray;          // input array of MC points
    TClonesArray* fMCTracksArray;         // input array of MC tracks
    TClonesArray* fHitArray;              // input array of TPC hits from the hit producer
    TClonesArray* fDigits;                // output array of TPC digits (write into output tree)
    TpcGas* fGas;                         // pointer to gas system
    DigOrigArray ***fDigits4dArray;       // output array of digital signals (don't write into output tree)
    TpcSector *fSector;                   // object for getting geometrical parameters of TPC sector
    MpdTpcDigitizerQAHistograms *fHisto;  // pointer to object needed only for QA creating
    TF1* fPRF;                            // pad response function
    FairField* fMagField;                 // magnetic field
    
    Float_t fGain;                        // coefficient for avalanches calculating
    Float_t zCathode;                     // length of TPC
    Float_t fNoiseThreshold;              // threshold for signal separation
    Float_t fOverflow;                    // ADC overflow value
    const Int_t *fNumOfPadsInRow;         // array of pads quantity in every TPC rows
    UInt_t nSectors;                      // number of TPC sectors
    UInt_t fNTimeBins;                    // number of time samples in TPC
    UInt_t nRows;                         // number of rows in TPC sector
    UInt_t nInRows;                       // number of rows in inner part of TPC sector
    UInt_t nOutRows;                      // number of rows in outer part of TPC sector
    Float_t fSectInHeight;                // height of inner part of TPC sector
    Float_t fSectHeight;                  // height of TPC sector
    Float_t r_min;                        // minimal radius of TPC
    Float_t fSpread;                      // sigma for pad response function    
    Float_t k1, k2;                       // coefficients for padRespose calculating    
    Float_t pwIn;                         // inner pad width
    Float_t phIn;                         // inner pad height    
    Float_t pwOut;                        // outer pad width
    Float_t phOut;                        // outer pad height
    MpdTpcSectorGeo* fSecGeo;             // sector geometry
    
    // set of boolean flags for manage of work process 
    Bool_t fIsHistogramsInitialized; // is QA histograms initialized or not
    Bool_t fMakeQA;                  // create or not in output tree branch with QA histograms
    Bool_t fOnlyPrimary;             // Take into account only primary particle or not 
    Bool_t fPersistence;             // print or not output array into tree
    Bool_t fAttach;                  // attach electrons in gas or not
    Bool_t fDiffuse;                 // diffuse electrons in TPC  or not
    Bool_t fDistort;                 // not implemented yet
    Bool_t fResponse;                // to do pad response or not
    Bool_t fDistribute;              // distribute electrons between two MC points or not
    Bool_t fPrintDebugInfo;          // print or not additional information in output
    Bool_t fOneRow;                  // distribute charge only in one padrow - for debug

    ClassDef(MpdTpcFastDigitizer, 0)
};

#endif

//--------------------------------------------------------------
// $Log$
//--------------------------------------------------------------
