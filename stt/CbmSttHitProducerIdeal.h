// -------------------------------------------------------------------------
// -----                 CbmStsHitProducerIdel header file             -----
// -----                  Created 10/01/06  by V. Friese               -----
// -------------------------------------------------------------------------


/** CbmStsHitProducerIdeal.h
 *@author V.Friese <v.friese@gsi.de>
 **
 ** The ideal hit producer produces hits of type CbmStsMapsHit by copying
 ** the MCPoint position. The position error is set to 1 mum, much 
 ** smaller than can be obtained by any detector. Using the hits from 
 ** this HitProducer is thus equivalent to using MC information
 ** directly, but with the correct data interface.
 **/


#ifndef CBMSTTHITPRODUCERIDEAL_H
#define CBMSTTHITPRODUCERIDEAL_H 

#include "FairTask.h"

#include "TVector3.h"
#include "TRandom.h"
 
class TClonesArray;

class CbmSttHitProducerIdeal : public FairTask
{
 public:

  /** Default constructor **/  
  CbmSttHitProducerIdeal();


  /** Destructor **/
  ~CbmSttHitProducerIdeal();


  /** Virtual method Init **/
  virtual InitStatus Init();


  /** Virtual method Exec **/
  virtual void Exec(Option_t* opt);


 private:
  /** Private method GetClostestApproachToWire;
   **
   ** Returns the measured values of the straw, radius and z position
   *@param closestDistanceinPos the radial position of the hit in straw frame
   *@param zPosInStrawFrame the z-position of the hit in straw frame
   *@param inPos entry point of the track into the straw
   *@param outPos exit point of the track into the straw
   **/
  void GetClostestApproachToWire(Double_t &closestDistance, 
				 Double_t &closestDistanceError,
				 TVector3 inPos, TVector3 outPos);
  void FoldZPosWithResolution(Double_t &zpos, Double_t &zposError, 
			      TVector3 localInPos, TVector3 localOutPos);

  Double_t GetRadialResolution(Double_t radius);
  Double_t GetLongitudinalResolution(Double_t zpos);

  /** Input array of CbmSttPoints **/
  TClonesArray* fPointArray;

  /** Output array of CbmSttHits **/
  TClonesArray* fHitArray;  
  
  /** Output array of CbmSttHitInfo **/
  TClonesArray* fHitInfoArray;

  ClassDef(CbmSttHitProducerIdeal,1);

};

#endif
