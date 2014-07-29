// -------------------------------------------------------------------------
// -----                         CbmStt header file                    -----
// -----                  Created 25/03/06  by R. Castelijns           -----
// -----                          Based on FairTof                      -----
// -------------------------------------------------------------------------

/**  CbmStt.h
 *@author R.Castelijns <r.castelijns@fz-juelich.de>
 **
 ** Defines the active detector STT. Constructs the geometry and
 ** registers MCPoints.
 **/



#ifndef CBMSTT_H
#define CBMSTT_H


#include "TClonesArray.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "FairDetector.h"
#include "CbmGeoSttPar.h"
#include "FairRun.h"

#include <string>
#include <sstream>

using std::string;

class TClonesArray;
class CbmSttPoint;
class FairVolume;

class CbmStt : public FairDetector
{

 public:

  /** Default constructor **/
  CbmStt();


  /** Standard constructor.
   *@param name    detetcor name
   *@param active  sensitivity flag
   **/
  CbmStt(const char* name, Bool_t active);


  /** Destructor **/
  virtual ~CbmStt();


  /** Virtual method ProcessHits
   **
   ** Defines the action to be taken when a step is inside the
   ** active volume. Creates FairTrdPoints and adds them to the
   ** collection.
   *@param vol  Pointer to the active volume
   **/
  virtual Bool_t  ProcessHits(FairVolume* vol = 0);


  /** Virtual method EndOfEvent
   **
   ** If verbosity level is set, print hit collection at the
   ** end of the event and resets it afterwards.
   **/
  virtual void EndOfEvent();


  /** Virtual method Register
   **
   ** Registers the hit collection in the ROOT manager.
   **/
  virtual void Register();


  /** Accessor to the hit collection **/
  virtual TClonesArray* GetCollection(Int_t iColl) const;


  /** Virtual method Print
   **
   ** Screen output of hit collection.
   **/
  virtual void Print() const;


  /** Virtual method Reset
   **
   ** Clears the hit collection
   **/
  virtual void Reset();


  /** Virtual method CopyClones
   **
   ** Copies the hit collection with a given track index offset
   *@param cl1     Origin
   *@param cl2     Target
   *@param offset  Index offset
   **/
  virtual void CopyClones(TClonesArray* cl1, TClonesArray* cl2, 
			  Int_t offset);


  /** Virtual method Construct geometry
   **
   ** Constructs the STT geometry
   **/
  virtual void ConstructGeometry();


  
 private:

  /** Track information to be stored until the track leaves the
      active volume. **/
  Int_t          fTrackID;           //!  track index
  Int_t          fVolumeID;          //!  volume id
  TLorentzVector fPos;               //!  wire position in global frame
  TLorentzVector fPosIn;             //!  entry position in global frame
  TLorentzVector fPosOut;            //!  exit position in global frame
  TLorentzVector fPosInLocal;        //!  entry position in straw frame
  TLorentzVector fPosOutLocal;       //!  exit position in straw frame
  TLorentzVector fMomIn;             //!  momentum
  TLorentzVector fMomOut;            //!  momentum
  Double32_t     fTime;              //!  time
  Double32_t     fLength;            //!  length
  Double32_t     fELoss;             //!  energy loss
  Double32_t     fMass;              //!  particle mass
  Bool_t         fIsInitialized;
  TObjArray *    fSensNodes;
  TObjArray *    fPassNodes;
  CbmGeoSttPar * fParameters;
  //  FairRun *       fRun;
  Int_t fPosIndex;                   //!
  TClonesArray* fSttCollection;      //! Hit collection
  TLorentzVector fpostot;   //! global frame hit position (in)// da cancellare
  TLorentzVector fpostotin;   //! global frame hit position (in)// da cancellare
  TLorentzVector fpostotout;   //! global frame hit position (in)// da cancellare



  /** Private method AddHit
   **
   ** Adds a FairTrdPoint to the HitCollection
   **/
  CbmSttPoint* AddHit(Int_t trackID, Int_t detID, TVector3 pos,
		      TVector3 posInLocal, TVector3 posOutLocal,
		      TVector3 momIn, TVector3 momOut, TVector3 wireDir,
		      Double_t time, Double_t length, Double_t eLoss, Double_t mass, TVector3 postot); // da cancellare postot

  string GetStringPart(string &aSrc, Int_t part, char aDelim);
  bool Split(string &aDest, string &aSrc, char aDelim);

  /** Private method ResetParameters
   **
   ** Resets the private members for the track parameters
   **/
  void ResetParameters();

  Bool_t  InitProcessHits();

  /** Private method GetSquaredDistanceFromWire
   **
   ** Returns the square of the distance of the current trackpoint to the wire
   *@return     distance
   **/
  float GetSquaredDistanceFromWire();

  ClassDef(CbmStt,2)

};


inline void CbmStt::ResetParameters() {
  fTrackID = fVolumeID = 0;
  fPos.SetXYZM(0.0, 0.0, 0.0, 0.0);
  fPosInLocal.SetXYZM(0.0, 0.0, 0.0, 0.0);
  fPosOutLocal.SetXYZM(0.0, 0.0, 0.0, 0.0);
  fMomIn.SetXYZM(0.0, 0.0, 0.0, 0.0);
  fMomOut.SetXYZM(0.0, 0.0, 0.0, 0.0);
  fTime = fLength = fELoss = 0;
  fPosIndex = 0;
  fpostot.SetXYZM(0.0, 0.0, 0.0, 0.0); // da cancellare
};


#endif

