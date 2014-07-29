// -------------------------------------------------------------------------
// -----                      CbmStsPoint source file                  -----
// -----                  Created 26/07/04  by V. Friese               -----
// -------------------------------------------------------------------------


#include <iostream>
#include "CbmSttPoint.h"


// -----   Default constructor   -------------------------------------------
CbmSttPoint::CbmSttPoint() : FairMCPoint() {
  fX_in_local  = fY_in_local   = fZ_in_local   = 0.;
  fX_out_local = fY_out_local  = fZ_out_local  = 0.;
  fPx_out      = fPy_out       = fPz_out       = 0.;
  fX_wire_dir  = fY_wire_dir   = fZ_wire_dir   = 0.;
}
// -------------------------------------------------------------------------



// -----   Standard constructor   ------------------------------------------
CbmSttPoint::CbmSttPoint(Int_t trackID, Int_t detID, TVector3 pos, 
			 TVector3 posInLocal, TVector3 posOutLocal, 
			 TVector3 momIn, TVector3 momOut, TVector3 wireDir,
			 Double_t tof, Double_t length, Double_t eLoss, Double_t mass, TVector3 postot)  // da cancellare postot
  : FairMCPoint(trackID, detID, pos, momIn, tof, length, eLoss)
{
  fX_out_local  = posOutLocal.X();
  fY_out_local  = posOutLocal.Y();
  fZ_out_local  = posOutLocal.Z();

  fX_in_local  = posInLocal.X();
  fY_in_local  = posInLocal.Y();
  fZ_in_local  = posInLocal.Z();

  fPx_out = momOut.Px();
  fPy_out = momOut.Py();
  fPz_out = momOut.Pz();

  fX_wire_dir = wireDir.X(); 
  fY_wire_dir = wireDir.Y(); 
  fZ_wire_dir = wireDir.Z(); 

  fMass = mass; // stt1

  // da cancellare
  fxtot = postot.X();
  fytot = postot.Y();
  fztot = postot.Z();
  /////

}
// -------------------------------------------------------------------------



// -----   Destructor   ----------------------------------------------------
CbmSttPoint::~CbmSttPoint() { }
// -------------------------------------------------------------------------




// -----   Public method Print   -------------------------------------------
void CbmSttPoint::Print(const Option_t* opt) const {
  cout << "-I- CbmStsPoint: STS Point for track " << fTrackID 
       << " in detector " << fDetectorID << endl;
  cout << "    Position (" << fX << ", " << fY << ", " << fZ
       << ") cm" << endl;
  cout << "    Momentum (" << fPx << ", " << fPy << ", " << fPz
       << ") GeV" << endl;
  cout << "    Time " << fTime << " ns,  Length " << fLength 
       << " cm,  Energy loss " << fELoss*1.0e06 << " keV" << endl;
}
// -------------------------------------------------------------------------



ClassImp(CbmSttPoint)
