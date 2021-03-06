// -------------------------------------------------------------------------
// -----                 MpdZdcHitproducer header file                 -----
// -----                 Created 14/08/06  by S.Spataro                -----
// -----                 Modified March 2021 by A.Strijak                -----
// -------------------------------------------------------------------------

#ifndef MPDZDCDIGIPRODUCER_H
#define MPDZDCDIGIPRODUCER_H 1


#include <map>
#include "FairTask.h"
#include "TClonesArray.h"
#include "MpdZdcDigi.h"
#include "MpdZdcGeoPar.h"
#include "MpdZdcDigiScheme.h"

#include "TParameter.h"
#include "TH2F.h"

#include "TRandom3.h"

class MpdZdcDigiProducer : public FairTask
{

 public:

  /** Default constructor **/  
  MpdZdcDigiProducer(const char* name="MpdZdc Digi Producer");


  /** Destructor **/
  ~MpdZdcDigiProducer();


  /** Virtual method Init **/
  virtual InitStatus Init();


  /** Virtual method Exec **/
  virtual void Exec(Option_t* opt);

  MpdZdcDigi* AddHit(Int_t detID, Int_t modID, Int_t chanID,Float_t energy);

  inline void SetPix2Mip (Double_t setValue) { fPix2Mip = setValue; }
  inline void SetMIPEnergy (Double_t setValue) { fMIPEnergy = setValue; }
  inline void SetMIPNoise (Double_t setValue) { fMIPNoise = setValue; }
  inline void SetMIP2GeV (Double_t setValue) { fMIP2GeV = setValue; }


 private:

  virtual void SetParContainers();


 private:

  TRandom3* fRandom3;

  Double_t fPix2Mip;    // MPPC pixels per MIP
  Double_t fMIPEnergy;  // MIP energy (5 MeV)
  Double_t fMIPNoise;   // MIP noise level
  Double_t fMIP2GeV;    // MIP to GeV

  Double_t RecoEnergy (Double_t pfELoss);

  /** Input array of MpdZdcPoints **/
  TClonesArray* fPointArray;

  /** Output array of MpdZdcDigi **/
  TClonesArray* fDigiArray; 

//  TClonesArray* fELossZdc1Value;
//  TClonesArray* fELossZdc2Value;

  /** Input geometry parameters container**/
  MpdZdcGeoPar* fGeoPar;

  /** Output Histograms of X-Y energy map **/
//  TH2F *fHistZdc1En;
//  TH2F *fHistZdc2En;


  ClassDef(MpdZdcDigiProducer,2);

};

#endif
