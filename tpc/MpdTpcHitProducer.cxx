/// \class MpdTpcHitProducer
///
/// Hit producer in MPD TPC
/// \author Alexander Zinchenko (LHEP, JINR, Dubna)

//---------------------------------------------------------------------------

#include "MpdTpcHitProducer.h"
#include "MpdTpcSectorGeo.h"
#include "MpdKalmanFilter.h"
#include "TpcGeoPar.h"
#include "TpcPoint.h"

#include "FairEventHeader.h"
#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairRunAna.h"
#include "FairRuntimeDb.h"

#include <TF1.h>
#include <TGeoManager.h>
#include <TGeoTube.h>
#include <TSpline.h>
//#include "Math/Interpolator.h"
#include <iostream>

using namespace std;

//---------------------------------------------------------------------------
MpdTpcHitProducer::MpdTpcHitProducer()
  : FairTask("TPC Hit Producer"),
    fModular(0),
    fPersistance(kFALSE)
{
    fPointArray = nullptr, fHitArray = nullptr;
}

//---------------------------------------------------------------------------
MpdTpcHitProducer::~MpdTpcHitProducer() { }

//---------------------------------------------------------------------------
InitStatus MpdTpcHitProducer::Init()
{
  FairRootManager* ioman = FairRootManager::Instance();
  if(!ioman) {
      cout << "\n-E- [MpdTpcHitProducer::Init]: RootManager not instantiated!" << endl;
      return kFATAL;
  }

  fPointArray = (TClonesArray*) ioman->GetObject("TpcPoint");
  if(!fPointArray) {
      cout << "\n-W- [MpdTpcHitProducer::Init]: No TpcPoint array!" << endl;
      return kERROR;
  }

  // Create and register output array
  fHitArray = new TClonesArray("MpdTpcHit");
  if (fPersistance) ioman->Register("TpcHit", "TPC", fHitArray, kTRUE);
  else ioman->Register("TpcHit", "TPC", fHitArray, kFALSE);

  cout << "-I- MpdTpcHitProducer: Intialisation successfull." << endl;
  return kSUCCESS;
}

// -----   Private method SetParContainers   -------------------------------
void MpdTpcHitProducer::SetParContainers() {

  //return;
  // Get run and runtime database
  FairRunAna* run = FairRunAna::Instance();
  if ( ! run ) Fatal("SetParContainers", "No analysis run");

  FairRuntimeDb* db = run->GetRuntimeDb();
  if ( ! db ) Fatal("SetParContainers", "No runtime database");

  // Get TPC geometry parameter container
  db->getContainer("TpcGeoPar");
  //db->getContainer("MpdTofGeoPar");
}

//---------------------------------------------------------------------------
void MpdTpcHitProducer::Exec(Option_t* opt)
{
  static Int_t eventNo = 0;

  if (!fHitArray) Fatal("Exec", "No MpdTpcHitArray");

  cout << "\n-I- MpdTpcHitProducer: Event No. " << FairRun::Instance()->GetEventHeader()->GetMCEntryNumber() << " " << ++eventNo << endl;

  // Reset output array
  fHitArray->Delete();

  /// Merge hits if they belong to the same track and
  /// not too far from each other in Z: digitization emulation (interim solution).

  //fhLays->Reset();

  static Int_t first = 1;//, version3 = 0;
  static Double_t rMin = 99999.0, rMax = 0.0, dR;
  Int_t lay, layMax = 0, nPoints = fPointArray->GetEntriesFast();
  if (first) {
    first = 0;
    //TpcPoint *point = (TpcPoint*) fPointArray->First();
    //TVector3 posOut;
    //AZ point->PositionOut(posOut);
    //if (posOut != TVector3(0,0,0)) version3 = 1; // sectored and layered sensitive volume
    FairRuntimeDb* rtdb = FairRun::Instance()->GetRuntimeDb();
    //rtdb->printParamContexts();
    //cout << rtdb->findContainer("TpcGeoPar") << " " << rtdb->findContainer("TpcContFact:") << endl;
    TpcGeoPar *geoPar = (TpcGeoPar*) rtdb->getContainer("TpcGeoPar");
    TString volName = "tpc01sv";
    TObjArray* sensNodes = geoPar->GetGeoSensitiveNodes();
    //cout << sensNodes->GetEntriesFast() << " " << geoPar->GetGeoPassiveNodes()->GetEntriesFast() << endl;
    //FairGeoNode* sensVol = (FairGeoNode*) (sensNodes->FindObject(volName));
    Int_t nSens = sensNodes->GetEntriesFast(), nLays = 0;
    FairGeoNode* sensVol0 = nullptr;
    for (Int_t i = 0; i < nSens; ++i) {
      FairGeoNode* sensVol = (FairGeoNode*) (sensNodes->UncheckedAt(i));
      TString name = sensVol->GetName();
      if (!name.Contains("tpc")) continue;
      TArrayD* params = sensVol->getParameters();
      rMin = TMath::Min (params->At(0), rMin);
      rMax = TMath::Max (params->At(1), rMax);
      sensVol0 = sensVol;
      nLays = TMath::Max (nLays, TString(name(name.First('r')+1,15)).Atoi());
    }
    ++nLays;
    TObjArray* passNodes = geoPar->GetGeoPassiveNodes();
    if (nSens) {
      // Old geometry scheme
      if (sensVol0->getShape() == "PGON") {
        fModular = 1; // force using modular geometry
        FairGeoNode *inWall = (FairGeoNode*) passNodes->FindObject("tpc01InWall");
        TArrayD* params = inWall->getParameters();
        fZtpc = params->At(2);
      } else {
        FairGeoNode *tpc = (FairGeoNode*) passNodes->FindObject("tpcChamber1");
        rMax = tpc->getParameters()->At(1);
        dR = (rMax - rMin) / nLays;
        fZtpc = sensVol0->getParameters()->At(2);
        MpdTpcSectorGeo *geo = MpdTpcSectorGeo::Instance();
        geo->SetNofRows(nLays);
        geo->SetPadHeight(dR);
        geo->SetMinY(rMin);
      }
      cout << " *** TPC sensitive volume: " << sensVol0->GetName() << " "
       << rMin << " " << rMax << " " << fZtpc << " " << nLays << " " << dR << endl;
    } else {
      // New geometry scheme (ROOT geo)
      fModular = 1; // force using modular geometry
      TGeoVolume *inW = gGeoManager->GetVolume("tpc01InWall");
      TGeoTube *tube = (TGeoTube*) inW->GetShape();
      fZtpc = tube->GetDZ();
    }

    /*
    MpdTofGeoPar *tofPar = (MpdTofGeoPar*) rtdb->getContainer("MpdTofGeoPar");
    cout << tofPar << endl;
    //TString volName = "tpc01#1";
    //TString volName = "tpc01sv";
    TObjArray* sensNodesTof = tofPar->GetGeoSensitiveNodes();
    cout << sensNodesTof->GetEntriesFast() << " " << tofPar->GetGeoPassiveNodes()->GetEntriesFast() << endl;
    */
    //exit(0);

  }
  // !!!!!!!!! Geometry dependent values
  //Double_t rMin = 34.19, rMax = 99.81, dR = (rMax-rMin)/50; // 1.3124; // new version (with dead material)
  // !!!!!!!!!

  //AZ if (version3) { ExecNew(); return; } // sectored and layered sensitive volume
  if (1) { ExecNew(); return; } // more streamlined procedure
  if (fModular) { ExecModular(); return; } // emulate geometry of readout chambers

  TVector3 p3, p30, p3err(0.05, 0., 0.1); // R-Phi error 500 um, Z error 1 mm
  multiset<Int_t> layset;
  for (Int_t j = 0; j < nPoints; ++j ) {
    TpcPoint* point = (TpcPoint*) fPointArray->UncheckedAt(j);
    point->Position(p3);
    Double_t rad = p3.Pt();
    lay = (Int_t) ((rad - rMin) / dR);
    lay = TMath::Max (lay, 0);
    layMax = TMath::Max (lay, layMax);
    //MpdTpcHit *hit = new ((*fHitArray)[j]) MpdTpcHit();
    MpdTpcHit *hit = AddRawHit(j, point->GetDetectorID(), p3, p3err, j, point->GetTrackID());
    layset.insert(lay);
    hit->SetLayer(lay);
    hit->SetR(rad);
    hit->SetRphi(rad * p3.Phi());
    hit->SetLocalZ(p3.Z());
    hit->SetEnergyLoss(point->GetEnergyLoss());
    hit->SetStep(point->GetStep());
    //fhLays->Fill(lay+0.1);
  }
  cout << " Max layer = " << layMax << " " << fPointArray->GetEntriesFast() << endl;
  fHitArray->Sort(); // according to layer No. or radius (local y)

  Int_t ipos = 0;
  Int_t nHits = 0;
  for (Int_t i = layMax; i >= 0; --i) {
    ipos += nHits;
    nHits = layset.count(i);
    //nHits = TMath::Nint(fhLays->GetCellContent(i+1,0));
    if (nHits < 2) continue;
    Int_t *iHits = new Int_t [nHits];
    Int_t *index = new Int_t [nHits];
    for (Int_t j = 0; j < nHits; ++j) {
      Int_t k = j + ipos;
      iHits[j] = ((MpdTpcHit*)fHitArray->UncheckedAt(k))->GetTrackID();
    }
    TMath::Sort(nHits,iHits,index);

    for (Int_t j = 0; j < nHits; ++j) {
      // Loop over first hits
      MpdTpcHit *hit0 = (MpdTpcHit*) fHitArray->UncheckedAt(index[j]+ipos);
      if (hit0 == 0x0) continue;
      Int_t nMerge = 1;
      TpcPoint* point0 = (TpcPoint*) fPointArray->UncheckedAt(hit0->GetRefIndex());
      Double_t rPhi = hit0->GetRphi();
      Double_t z = hit0->GetLocalZ();
      Double_t rad = hit0->GetR();
      hit0->Position(p30);
      Double_t leng = point0->GetLength();
      Double_t edep = point0->GetEnergyLoss();
      Double_t step = point0->GetStep();
      for (Int_t j1 = j + 1; j1 < nHits; ++j1) {
        // Second hit
        Int_t k = index[j1] + ipos;
        MpdTpcHit *hit = (MpdTpcHit*) fHitArray->UncheckedAt(k);
        if (hit == 0x0) continue;
        TpcPoint* point = (TpcPoint*) fPointArray->UncheckedAt(hit->GetRefIndex());
        if (point->GetTrackID() != point0->GetTrackID()) break;
        //if (TMath::Abs(point->GetZ()-point0->GetZ()) > 5) continue;
        //if (TMath::Abs(hit0->GetRphi()-Proxim(hit0,hit)) > 5) continue;
        if (TMath::Abs(point->GetZ()-point0->GetZ()) > 2.5) continue;
        if (TMath::Abs(hit0->GetRphi()-Proxim(hit0,hit)) > 2.5) continue;
        // Merge hits
        hit->Position(p3);
        p30 += p3;
        rPhi += Proxim(hit0,hit);
        z += hit->GetLocalZ();
        rad += hit->GetR();
        leng += point->GetLength();
        edep += point->GetEnergyLoss();
        step += point->GetStep();
        hit0->AddLinks(hit->GetLinks()); // copy links
        fHitArray->RemoveAt(k);
        ++nMerge;
      }
      if (nMerge == 1) continue;
      //cout << i << " nMerge: " << nMerge << " " << hit0->GetRphi() << " " << hit0->GetZ() << " " << hit0->GetR() << endl;
      p30.SetMag(p30.Mag()/nMerge);
      hit0->SetPosition(p30);
      hit0->SetRphi(rPhi/nMerge);
      hit0->SetLocalZ(z/nMerge);
      hit0->SetR(rad/nMerge);
      //((TpcPoint*)fTpcPoints->UncheckedAt(hit0->GetRefIndex()))->SetLength(leng/nMerge); //
      hit0->SetLength(leng/nMerge);
      hit0->SetEnergyLoss(edep);
      hit0->SetStep(step);
    //cout << hit0->GetRphi() << " " << hit0->GetZ() << " " << hit0->GetR() << " " << hit0->GetTrackID() << endl;
      //cout << nMerge << " " << hit0->GetTrackID() << " " << hit0->GetLayer() << endl;
    }
    delete [] iHits;
    delete [] index;
  } // for (Int_t i = layMax; i >= 0;
  fHitArray->Compress();
  fHitArray->UnSort();
  fHitArray->Sort();

  /*
  fhLays->Reset();
  nHits = fHits->GetEntriesFast();
  cout << " Number of merged hits: " << nHits << endl;
  Int_t nKh = 0;
  for (int j = 0; j < nHits; ++j ) {
    TpcLheHit* hit = (TpcLheHit*) fHits->UncheckedAt(j);
    //if (hit->GetTrackID() != 1783) continue; // keep only one track
    lay = hit->GetLayer();
    fhLays->Fill(lay+0.1);
    // Add errors
    Double_t dRPhi = 0, dZ = 0;
    gRandom->Rannor(dRPhi,dZ);
    hit->SetZ(hit->GetZ()+dZ*hit->GetZerr()); //add error
    hit->SetRphi(hit->GetRphi()+dRPhi*hit->GetRphiErr()); //add error
    Double_t phi = hit->GetRphi() / hit->GetR();
    hit->SetX(hit->GetR()*TMath::Cos(phi));
    hit->SetY(hit->GetR()*TMath::Sin(phi));
    // Create Kalman hits
    //(Int_t detID, Int_t nDim, HitType hitType, Double_t *meas, Double_t *err, Double_t *cosSin, Double_t signal, Double_t dist, Int_t index)
    Double_t meas[2] = {hit->GetRphi(), hit->GetZ()};
    Double_t err[2] = {hit->GetRphiErr(), hit->GetZerr()};
    Double_t cossin[2] = {1., 0.};
    MpdKalmanHit *hitK = new ((*fKHits)[nKh++])
      MpdKalmanHit(lay*1000000+nKh-1, 2, MpdKalmanHit::kFixedR, meas, err, cossin, hit->GetEdep()/hit->GetStep(), hit->GetR(), hit->GetRefIndex());
    hitK->SetLength(((TpcPoint*)fTpcPoints->UncheckedAt(hit->GetRefIndex()))->GetLength());
    //hitK->SetDedx (hit->GetEdep()/hit->GetStep());
    //MpdKalmanFilter::Instance()->GetGeo()->SetGlobalPos(hitK,TVector3(hit->GetX(),hit->GetY(),hit->GetZ()));
  }

  fLayPointers = new Int_t [layMax+1];
  ipos = 0;
  for (Int_t i = layMax; i >= 0; --i) {
    //cout << i << " " << fhLays->GetCellContent(i+1,0) << endl;
    //if (ipos) cout << ((TpcLheHit*)fHits->UncheckedAt(ipos))->GetLayer() << " "
    //     << ((TpcLheHit*)fHits->UncheckedAt(ipos-1))->GetLayer() << endl;
    fLayPointers[i] = ipos;
    ipos += (Int_t) fhLays->GetCellContent(i+1,0);
  }
  */

  // Event summary
  cout << "-I- MpdTpcHitProducer: " << nPoints << " TpcPoints, " << fHitArray->GetEntriesFast() << " Hits created." << endl;
}

//---------------------------------------------------------------------------
void MpdTpcHitProducer::ExecModular()
{
  // Emulate geometry of readout chambers

  MpdTpcSectorGeo *secGeo = MpdTpcSectorGeo::Instance();
  //ROOT::Math::Interpolator inter(3, ROOT::Math::Interpolation::kPOLYNOMIAL);

  Int_t lay, layMax = 0, nPoints = fPointArray->GetEntriesFast(), nHits = 0;
  TVector3 p3, p30, p3local, p3local0, pmom3, pmom3loc, p3extr, p3err(0.05, 0., 0.1); // X error 500 um, Z error 1 mm
  multiset<Int_t> layset;
  multimap<Int_t,Int_t> midIndx;
  cout << " MC poins: " << fPointArray->GetEntriesFast() << endl;

  for (Int_t j = 0; j < nPoints; ++j ) {
  //for (Int_t j = 0; j < 1000; ++j ) {
    TpcPoint* point = (TpcPoint*) fPointArray->UncheckedAt(j);
    //if (point->GetTrackID() != 1) continue; ///
    if (point->GetTrackID() < 0) continue; /// strange case - protection
    point->Position(p3);
    Int_t padID = secGeo->Global2Local(p3, p3local);
    //if (padID < 0) {cout << j << " " << p3.X() << " " << p3.Y() << " " << p3local.X() << " " << p3local.Y() << endl; continue; }// outside sector boundaries
    if (padID < 0) continue; // outside sector boundaries
    point->Momentum(pmom3);
    pmom3loc.SetXYZ(0.,0.,0.);
    //if (point->GetTrackID() <= 2 && point->GetTrackID() > 0) printf("%2d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %6d %4d %4d\n", point->GetTrackID(), p3.X(), p3.Y(), p3.Z(), p3local.X(), p3local.Y(), p3local.Z(), padID, TpcPadID::numberToPadID(padID).row(), TpcPadID::numberToPadID(padID).sector());
    lay = secGeo->PadRow(padID);
    layMax = TMath::Max (lay, layMax);
    //MpdTpcHit *hit = AddRawHit(j, point->GetDetectorID(), p3, p3err, j, point->GetTrackID());
    MpdTpcHit *hit = AddRawHit(nHits++, padID, p3, p3err, j, point->GetTrackID());
    hit->SetLayer(lay);
    hit->SetLocalPosition(p3local); // point position
    //cout << p3local.X() << " " << p3local.Y() << " " << p3local.Z() << " " << hit->GetLocalY() << endl;
    hit->SetEnergyLoss(point->GetEnergyLoss());
    hit->SetStep(point->GetStep());
    hit->SetModular(1); // modular geometry flag
    hit->SetLength(point->GetLength());
    midIndx.insert(pair<Int_t,Int_t>(point->GetTrackID(),nHits-1));
  } // for (Int_t j = 0; j < nPoints;

  // Perform track interpolation to put hits on padrow median planes
  //
  multimap<Int_t,Int_t>::iterator mit;
  Int_t id0 = -1, i0 = 0, nh = 0, *hindx = nullptr, *ord = nullptr, lastIndx = 0;
  Double_t *times = nullptr, *ys = nullptr, *zs = nullptr, *xx = nullptr, *yy = nullptr, *zz = nullptr;
  for (mit = midIndx.begin(); mit != midIndx.end(); ++mit) {
    // Loop over track hits
    if (mit->first != id0) {
      id0 = mit->first;
      nh = midIndx.count(id0);
      times = new Double_t [nh];
      ys = new Double_t [nh];
      zs = new Double_t [nh];
      xx = new Double_t [nh];
      yy = new Double_t [nh];
      zz = new Double_t [nh];
      hindx = new Int_t [nh];
      ord = new Int_t [nh];
      i0 = 0;
    }
    MpdTpcHit *h = (MpdTpcHit*) fHitArray->UncheckedAt(mit->second);
    TpcPoint* point = (TpcPoint*) fPointArray->UncheckedAt(h->GetRefIndex());
    times[i0] = point->GetTime();
    hindx[i0++] = mit->second;

    if (i0 != nh) continue;
    TMath::Sort (nh, times, ord, kFALSE);
    // Take 3 consecutive (in time) points and use parabolic interpolation
    // to put the hit on a padrow median plane
    Int_t sec0 = -99, np = 0, ibeg = 0;
    Double_t ypad0 = -999.0, dir = 0.0, xhit = 0.0, zhit = 0.0;
    Bool_t ok = kTRUE;

    for (Int_t i3 = 0; i3 < nh; ++i3) {
      h = (MpdTpcHit*) fHitArray->UncheckedAt(hindx[ord[i3]]);
      Int_t sec = secGeo->Sector(h->GetDetectorID());
      if (sec != sec0 || i3 > lastIndx) {
        //if (sec != sec0) {
        // Store coordinates of all hits in this sector
        //cout << " ***** New sector: " << sec << endl;
        sec0 = sec;
        np = ibeg = 0;
        ypad0 = -999.0;
        for (Int_t jj = i3; jj < nh; ++jj) {
          MpdTpcHit *h1 = (MpdTpcHit*) fHitArray->UncheckedAt(hindx[ord[jj]]);
          sec = secGeo->Sector(h1->GetDetectorID());
          if (sec != sec0) break; // different sector
          yy[np] = h1->GetLocalY();
          xx[np] = h1->GetLocalX();
          if (np > 0 && TMath::Abs(xx[np]-xx[np-1]) > 5.0) break; // broken track
          zz[np++] = h1->GetLocalZ();
          lastIndx = jj;
        }
        /*for (Int_t jj = 0; jj < 6; ++jj) cout << xx[jj] << " " << yy[jj] << " " << zz[jj] << " ";
          cout << endl;*/
      }
      Double_t ypad = secGeo->LocalPadPosition(h->GetDetectorID()).Y(); // padrow position
      if (TMath::Abs(ypad-ypad0) < 0.1) {
        // The same padrow
        if (!ok) { fHitArray->RemoveAt(hindx[ord[i3]]); continue; }
        times[i3] = times[i3-1];
        ys[i3] = ys[i3-1];
        zs[i3] = zs[i3-1];
      } else {
        ypad0 = ypad;
        //cout << "---------------------------\n";
        ok = Interpolate(np, ibeg, yy, xx, zz, ypad0, dir, xhit, zhit);
        if (!ok) { fHitArray->RemoveAt(hindx[ord[i3]]); continue; }
        times[i3] = xhit; // reuse array times
        ys[i3] = ypad0;
        zs[i3] = zhit;
      }

      /*cout << " np, ibeg, y, x, z: " << np << " " << ibeg << " " << yy[ibeg] << " " << xx[ibeg] << " " << zz[ibeg] << endl;
      cout << ys[i3] << " " << times[i3] << " " << zs[i3] << endl;*/

      if (ibeg < np - 2) ++ibeg;
      Double_t dy = ypad0 - h->GetLocalY();
      //h->SetLocalY(ypad0);
      Double_t dx = times[i3] - h->GetLocalX();
      Double_t dz = zs[i3] - h->GetLocalZ();
      Double_t dl = dx * dx + dy * dy + dz * dz;
      dl = TMath::Sign (TMath::Sqrt(dl), dy);
      if (dir > 0) h->SetLength(h->GetLength()+dl); // going outward
      else h->SetLength(h->GetLength()-dl); // inward
    } // for (Int_t i3 = 0; i3 < nh;
    // Change hit coordinates
    for (Int_t i3 = 0; i3 < nh; ++i3) {
      h = (MpdTpcHit*) fHitArray->UncheckedAt(hindx[ord[i3]]);
      if (h == nullptr) continue;
      Int_t padID = h->GetDetectorID();
      h->SetLocalX(times[i3]);
      h->SetLocalY(ys[i3]);
      h->SetLocalZ(zs[i3]);
      h->LocalPosition(p30);
      secGeo->Local2Global(secGeo->Sector(padID), p30, p3);
      h->SetPosition(p3);
    }
    delete [] times;
    delete [] ys;
    delete [] zs;
    delete [] xx;
    delete [] yy;
    delete [] zz;

    // Merge hits
    for (Int_t i3 = 0; i3 < nh; ++i3) {
      // Loop over first hits
      MpdTpcHit *hit0 = (MpdTpcHit*) fHitArray->UncheckedAt(hindx[ord[i3]]);
      if (hit0 == 0x0) continue;
      Int_t padID0 = hit0->GetDetectorID();
      Int_t nMerge = 1;
      hit0->LocalPosition(p3local0);
      hit0->LocalPosition(p3extr);
      hit0->Position(p30);
      Double_t leng = hit0->GetLength();
      Double_t edep = hit0->GetEnergyLoss();
      Double_t step = hit0->GetStep();

      for (Int_t i31 = i3 + 1; i31 < nh; ++i31) {
        //continue; ///
        // Second hit
        MpdTpcHit *hit = (MpdTpcHit*) fHitArray->UncheckedAt(hindx[ord[i31]]);
        if (hit == 0x0) continue;
        Int_t padID = hit->GetDetectorID();
        if (secGeo->PadRow(padID) != secGeo->PadRow(padID0)) break; // in different padrows
        if (secGeo->Sector(padID) != secGeo->Sector(padID0)) break; // in different sectors
        hit->LocalPosition(p3local);
        hit->Position(p3);
        if (TMath::Abs(p3extr.Z()-p3local.Z()) > 2.5) break;
        if (TMath::Abs(p3extr.X()-p3local.X()) > 2.5) break;
        // Merge hits
        //cout << " Merge: " << p3extr.X() << " " << p3extr.Y() << " " << p3extr.Z() << " " << p3local.X() << " " << p3local.Y() << " " << p3local.Z() << endl;
        p3local0 += p3local;
        p30 += p3;
        leng += hit->GetLength();
        edep += hit->GetEnergyLoss();
        step += hit->GetStep();
        //hit0->AddLinks(hit->GetLinks()); // copy links
        fHitArray->RemoveAt(hindx[ord[i31]]);
        ++nMerge;
      }

      if (nMerge == 1) continue;
      p3local0.SetMag(p3local0.Mag()/nMerge);
      hit0->SetLocalPosition(p3local0);
      p30.SetMag(p30.Mag()/nMerge);
      hit0->SetPosition(p30);
      hit0->SetLength(leng/nMerge);
      hit0->SetEnergyLoss(edep);
      hit0->SetStep(step);
    } // for (Int_t i3 = 0; i3 < nh;
    delete [] hindx;
    delete [] ord;

  } // for (mit = midIndx.begin(); mit != midIndx.end();

  fHitArray->Compress();
  fHitArray->UnSort();
  fHitArray->Sort();
  cout << " Merged hits: " << fHitArray->GetEntriesFast() << endl;

  // Debug
  /*
  nh = fHitArray->GetEntriesFast();
  multimap<Int_t,Int_t> mm;
  for (Int_t i = 0; i < nh; ++i) {
    MpdTpcHit *h = (MpdTpcHit*) fHitArray->UncheckedAt(i);
    //cout << i << " " << h->GetTrackID() << " " << h->GetLayer() << endl;
    //mm.insert(pair<Int_t,Int_t>(h->GetTrackID(),h->GetLayer()));
    mm.insert(pair<Int_t,Int_t>(h->GetTrackID(),i));
  }
  multimap<Int_t,Int_t>::iterator it;
  for (it = mm.begin(); it != mm.end(); ++it) {
    if (it->first != 6 && it->first != 7 && it->first != 0) continue;
    MpdTpcHit *h = (MpdTpcHit*) fHitArray->UncheckedAt(it->second);
    cout << it->first << " " << h->GetLayer() << " " << h->GetLocalY() << " " << h->GetLocalX() << " "
        << h->GetLocalZ() << " " << h->GetY() << " " << h->GetX() << " " << h->GetZ() << endl;
  }
  */
}

//---------------------------------------------------------------------------
Double_t MpdTpcHitProducer::Proxim(const MpdTpcHit *hit0, const MpdTpcHit *hit)
{
  /// Adjust hit coord. R-Phi to be "around" hit0 R-Phi - to avoid
  /// discontinuity around +- Pi

  Double_t phi0 = hit0->GetRphi() / hit0->GetR();
  Double_t phi = hit->GetRphi() / hit->GetR();
  return hit->GetR() * MpdKalmanFilter::Instance()->Proxim(phi0, phi);
}

//---------------------------------------------------------------------------
MpdTpcHit* MpdTpcHitProducer::AddRawHit(Int_t indx, Int_t detUID, const TVector3 &posHit,
                                        const TVector3 &posHitErr, Int_t pointIndx,
                                        Int_t trackIndx)
{
  MpdTpcHit *pHit = new  ((*fHitArray)[indx]) MpdTpcHit(detUID, posHit, posHitErr, pointIndx);
  pHit->AddLink(FairLink(MpdTpcHit::PointIndex, pointIndx));
  pHit->AddLink(FairLink(MpdTpcHit::MCTrackIndex, trackIndx));
  pHit->AddID(trackIndx);
  return pHit;
}

//---------------------------------------------------------------------------
Bool_t MpdTpcHitProducer::Interpolate(Int_t np, Int_t& ibeg, Double_t *yp, Double_t *xp, Double_t *zp,
                                      Double_t y0, Double_t& dir, Double_t& xhit, Double_t& zhit)
{
  // Evaluate x and z at y0 by interpolation (either parabolic (if np=3), or linear (if np=2))
  // (here y0 is the padrow y-coordinate)

  if (np == 1) {
    xhit = xp[0];
    zhit = zp[0];
    return kTRUE;
  }

  if (np == 2) {
    dir = TMath::Sign (1.0,yp[1]-yp[0]);
    // Linear interpolation
    /*
    Double_t expr = (y0 - yp[0]) / (yp[1] - yp[0]);
    xhit = xp[0] + (xp[1] - xp[0]) * expr;
    zhit = zp[0] + (zp[1] - zp[0]) * expr;
    */
    xhit = (xp[0] + xp[1]) / 2.0;
    zhit = (zp[0] + zp[1]) / 2.0;
    return kTRUE;
  } else if (np < 2) {
    Fatal("Interpolate", "No points !!!");
  }

  Double_t *yp3{nullptr}, *xp3{nullptr}, *zp3{nullptr};
  Bool_t ok = kTRUE;
  // Go to the right
  for (Int_t j = ibeg; j < np; ++j) {
    ok = kTRUE;
    yp3 = &yp[j];
    xp3 = &xp[j];
    zp3 = &zp[j];
    if (j > np - 3) { --yp3; --xp3; --zp3; --ibeg; }
    if (j > np - 2) { --yp3; --xp3; --zp3; --ibeg; }
    Double_t dy0 = yp3[0] - y0;
    Double_t dy2 = yp3[2] - y0;
    if (dy0 * dy2 > 0.0) {
      // Both points on the same side from y0
      if (TMath::Abs(dy2) < TMath::Abs(dy0) && j < np - 3)  { ++ibeg; continue; }
      //ok = kFALSE;
      if (j < np - 3) ok = kFALSE; // go left or turning point
      break;
    } else if ((yp3[1]-yp3[0]) * (yp3[2]-yp3[1]) < 0) ok = kFALSE; // not monotone
    break;
  }
  // Go to the left
  if (!ok) {
    for (Int_t j = ibeg; j > -1; --j) {
      ok = kTRUE;
      yp3 = &yp[j-2];
      xp3 = &xp[j-2];
      zp3 = &zp[j-2];
      //if (j < 2) { ++xp3; ++yp3; ++ibeg; }
      //if (j < 1) { ++xp3; ++yp3; ++ibeg; }
      if (j < 2) { ++yp3; ++xp3; ++zp3; }
      if (j < 1) { ++yp3; ++xp3; ++zp3; }
      Double_t dy0 = yp3[0] - y0;
      Double_t dy2 = yp3[2] - y0;
      if (dy0 * dy2 > 0.0) {
        // Both points on the same side from y0
        if (TMath::Abs(dy0) < TMath::Abs(dy2) && j > 2) { --ibeg; continue; }
        if (j > 0) ok = kFALSE; // turning point
        break;
      } else if ((yp3[1]-yp3[0]) * (yp3[2]-yp3[1]) < 0) ok = kFALSE; // not monotone
      break;
    }
  }

  dir = TMath::Sign (1.0,yp3[1]-yp3[0]);
  if (!ok) {
    xhit = (xp3[0] + xp3[1] + xp3[2]) / 3.0;
    zhit = (zp3[0] + zp3[1] + zp3[2]) / 3.0;
    return kTRUE;
  }

  /*for (Int_t jj = 0; jj < 3; ++jj) cout << yp3[jj] << " ";
  cout << endl;
  for (Int_t jj = 0; jj < 3; ++jj) cout << xp3[jj] << " ";
  cout << endl;
  for (Int_t jj = 0; jj < 3; ++jj) cout << zp3[jj] << " ";
  cout << endl;*/

  // Parabolic interpolation
  Double_t dy01 = yp3[1] - yp3[0], dy02 = yp3[2] - yp3[0];
  Double_t y01 = yp3[1] + yp3[0], y02 = yp3[2] + yp3[0];
  //Double_t yp302 = yp3[0] * yp3[0];
  Double_t ydy = dy02 * y02 - dy02 * y01;

  Double_t dx01 = xp3[1] - xp3[0], dx02 = xp3[2] - xp3[0];
  Double_t slopex = dx01 / dy01;
  Double_t ax = dx02 - slopex * dy02;
  ax /= ydy;
  Double_t bx = slopex - ax * y01;
  Double_t cx = xp3[0] - ax * yp3[0] * yp3[0] - bx * yp3[0];
  xhit = ax * y0 * y0 + bx * y0 + cx;
  // Check if x is inside sector boundaries
  // (can happen when crossing sector boundaries (in some cases))
  MpdTpcSectorGeo *secGeo = MpdTpcSectorGeo::Instance();
  Double_t xEdge = (y0 + secGeo->GetMinY()) * TMath::Tan(secGeo->Dphi()/2) + 0.2; // safety margin 0.2
  if (TMath::Abs(xhit) > xEdge) {
    return kFALSE;
    // Find y corresponding to sector boundary
    /*Double_t d = bx * bx - 4 * ax * (cx - TMath::Sign(xEdge,xhit));
    y0 = (-bx - TMath::Sqrt(d)) / 2 / ax;
    xhit = ax * y0 * y0 + bx * y0 + cx;*/
  }

  Double_t dz01 = zp3[1] - zp3[0], dz02 = zp3[2] - zp3[0];
  Double_t slopez = dz01 / dy01;
  Double_t az = dz02 - slopez * dy02;
  az /= ydy;
  Double_t bz = slopez - az * y01;
  Double_t cz = zp3[0] - az * yp3[0] * yp3[0] - bz * yp3[0];
  zhit = az * y0 * y0 + bz * y0 + cz;
  if (TMath::Abs(zhit) > fZtpc + 1) return kFALSE;

  return kTRUE;
}

//---------------------------------------------------------------------------

void MpdTpcHitProducer::ExecNew()
{
  // More streamlined version

  MpdTpcSectorGeo *secGeo = MpdTpcSectorGeo::Instance();
  static TF1 func("func","[1]+[2]*(x-[0])+[3]*(x-[0])*(x-[0])",0,99999);
  static Double_t maxStep = -1;
  if (maxStep < 0) {
    // Get max step for TPC gas
    TGeoMedium *tpcMed = gGeoManager->GetMedium("TPCmixture");
    maxStep = tpcMed->GetParam(4) * 0.95;
    cout << " ***!!! Maximum step in TPC: " << maxStep << endl;
  }

  Int_t nPoints = fPointArray->GetEntriesFast(), nHits = 0, lay = 0;
  TVector3 p3, p30, p3loc, p3loc0, pmom3, pmom3loc, p3extr, p3err(0.05, 0., 0.1); // X error 500 um, Z error 1 mm
  map<Int_t,map<Double_t,Int_t> > idmap;
  //multimap<Int_t,Int_t> midIndx;
  cout << " MC poins: " << nPoints << endl;

  // Get all points according to trackID

  for (Int_t j = 0; j < nPoints; ++j ) {
  //for (Int_t j = 0; j < 1000; ++j ) {
    TpcPoint* point = (TpcPoint*) fPointArray->UncheckedAt(j);
    Int_t id = point->GetTrackID();
    //if (id != 1) continue; ///
    if (id < 0) continue; /// strange case - protection
    if (idmap.count(id) == 0) {
      map<Double_t,Int_t> aaa;
      idmap[id] = aaa;
    }
    idmap[id][point->GetTime()] = j;
  }

  // Loop over trackIDs

  for (map<Int_t,map<Double_t,Int_t> >::iterator mit = idmap.begin(); mit != idmap.end(); ++mit) {
    Int_t id = mit->first;
    map<Double_t,Int_t>& aaa = mit->second;
    map<Double_t,MpdTpcHit> hitMap;

    // Loop over points from one track and create hit for each point

    for (map<Double_t,Int_t>::iterator mit1 = aaa.begin(); mit1 != aaa.end(); ++mit1) {
      TpcPoint *point = (TpcPoint*) fPointArray->UncheckedAt(mit1->second);
      point->Position(p3);
      Int_t padID = secGeo->Global2Local(p3, p3loc), isec = -1;
      if (padID < 0) continue; // outside sector boundaries
      lay = -1;
      if (padID >= 0) {
        lay = secGeo->PadRow(padID);
        isec = secGeo->Sector(padID);
      }
      MpdTpcHit hit(padID, p3, p3err, mit1->second);
      hit.SetLayer(lay);
      hit.SetLocalPosition(p3loc); // point position
      hit.SetEnergyLoss(point->GetEnergyLoss());
      hit.SetStep(point->GetStep());
      hit.SetModular(1); // modular geometry flag
      hit.SetLength(point->GetLength());
      // Track direction in sector frame
      Int_t idir = 0;
      if (padID >= 0) {
        point->Momentum(pmom3);
        if (pmom3.Mag() > 1.e-6) {
          p3extr = p3;
          pmom3.SetMag(0.001);
          p3extr += pmom3;
          secGeo->Global2Local(p3extr, pmom3loc, isec);
          pmom3loc -= p3loc;
          if (pmom3loc[1] < -1.e-7) idir = -1; // going inward
          else if (pmom3loc[1] > 1.e-7) idir = 1;
        }
      }
      hit.SetFlag(idir);
      //hit.AddLink(FairLink(MpdTpcHit::PointIndex, mit1->second));
      //hit.AddLink(FairLink(MpdTpcHit::MCTrackIndex, id));
      hitMap[mit1->first] = hit;
    }
    // Add fake hit to indicate end-of-track
    if (hitMap.size()) {
      MpdTpcHit htmp;
      htmp.SetDetectorID(secGeo->PadID(30,0));
      htmp.SetFlag(hitMap.rbegin()->second.GetFlag());
      hitMap[hitMap.rbegin()->first+1.0] = htmp;
    }

    // Find track segment in one sector going in one direction in sector frame
    Int_t isec0 = -9, idir0 = -9, layb, lay0; //laye,
    vector<Double_t> xyzloct[4];
    map<Double_t,MpdTpcHit>::iterator mitb, mite;
    nHits = fHitArray->GetEntriesFast();

    for (map<Double_t,MpdTpcHit>::iterator mit1 = hitMap.begin(); mit1 != hitMap.end(); ++mit1) {
      MpdTpcHit& hit = mit1->second;
      Int_t isec = secGeo->Sector(hit.GetDetectorID());
      //cout << isec << " " << secGeo->PadRow(hit.GetDetectorID()) << " " << hit.GetEnergyLoss() << endl;
      Int_t idir = hit.GetFlag();
      if (isec0 < 0) {
        isec0 = isec;
        idir0 = idir;
      }
      if (isec != isec0 || idir != idir0) {
        // Process track segment
        if (xyzloct[0].size() < 2) mitb = mite = hitMap.lower_bound(xyzloct[3].front()); // Only one point
        else {
          mitb = hitMap.lower_bound(xyzloct[3].front());
          mite = hitMap.lower_bound(xyzloct[3].back());
        }
        layb = mitb->second.GetLayer();
        //laye = mite->second.GetLayer();
        map<Double_t,MpdTpcHit>::iterator mitt = mitb, mitend = mite, mithit;
        ++mitend;

        // Create hit on each padrow median plane

        Int_t spsize = TMath::Max (Int_t(xyzloct[0].size()),2);
        //TSpline3 tx("tx",xyzloct[3].data(),xyzloct[0].data(),spsize);
        //TSpline3 tz("tz",xyzloct[3].data(),xyzloct[2].data(),spsize);
        TSpline3* yt = nullptr, *yx = nullptr, *yz = nullptr;
        if (xyzloct[1].front() > xyzloct[1].back()) {
          // Reverse vectors (to have localY in ascending order)
          vector<Double_t> yrev(xyzloct[1]), trev(xyzloct[3]), xrev(xyzloct[0]), zrev(xyzloct[2]);
          std::reverse(yrev.begin(),yrev.end());
          std::reverse(trev.begin(),trev.end());
          std::reverse(xrev.begin(),xrev.end());
          std::reverse(zrev.begin(),zrev.end());
          yt = new TSpline3("yt",yrev.data(),trev.data(),spsize);
          yx = new TSpline3("yx",yrev.data(),xrev.data(),spsize);
          yz = new TSpline3("yz",yrev.data(),zrev.data(),spsize);
        }
        else {
          yt = new TSpline3("yt",xyzloct[1].data(),xyzloct[3].data(),spsize);
          yx = new TSpline3("yx",xyzloct[1].data(),xyzloct[0].data(),spsize);
          yz = new TSpline3("yz",xyzloct[1].data(),xyzloct[2].data(),spsize);
        }
        // Fit yloc vs time to find maximum and minimum of yloc achieved
        Double_t ylocMinMax[2] = {0, 100}, dt = TMath::Max(xyzloct[3].back()-xyzloct[3].front(),0.2);
        //Double_t ylocMinMax[2] = {0, 100}, dt = xyzloct[3].back()-xyzloct[3][xyzloct[3].size()-2];
        if (spsize > 2) {
          dt *= 0.2;
          //dt *= 5;
          //*
          TGraph gr(spsize,xyzloct[3].data(),xyzloct[1].data());
          func.SetParameters(0,xyzloct[1].front(),0,0);
          func.FixParameter(0,xyzloct[3].front());
          //gr.Fit("pol2","Q");
          gr.Fit("func","Q");
          ylocMinMax[0] = gr.GetFunction("func")->GetMinimum(xyzloct[3].front()-dt,xyzloct[3].back()+dt);
          ylocMinMax[1] = gr.GetFunction("func")->GetMaximum(xyzloct[3].front()-dt,xyzloct[3].back()+dt);
          //*/
        }

        Double_t time = 0.0;
        MpdTpcHit *hitok = nullptr;
        lay0 = -9;

        for ( ; mitt != mitend; ++mitt) {
          Int_t padID = mitt->second.GetDetectorID();
          if (mitb == mite) {
            // One hit
            for (Int_t j = 0; j < 3; ++j) p3loc[j] = xyzloct[j].front();
            time = xyzloct[3].front();
            hitok = &mitb->second;
            lay = layb;
          } else {
            lay = secGeo->PadRow(padID);
            if (lay == lay0) continue; // hit from the same padrow
            p3loc[1] = secGeo->LocalPadPosition(padID).Y(); // padrow position
            //if (p3loc[1] < ylocMinMax[0] || p3loc[1] > ylocMinMax[1]) continue; // curling track
            if (p3loc[1] < ylocMinMax[0]) p3loc[1] = ylocMinMax[0];
            else if (p3loc[1] > ylocMinMax[1]) p3loc[1] = ylocMinMax[1];

            time = yt->Eval(p3loc[1]);
            if ((idir0 == 0 && time > xyzloct[3].back()) || time < xyzloct[3].front()-dt || time > xyzloct[3].back()+dt)
              continue; // do not extrapolate curling track
            //p3loc[0] = tx.Eval(time);
            //p3loc[2] = tz.Eval(time);
            p3loc[0] = yx->Eval(p3loc[1]);
            p3loc[2] = yz->Eval(p3loc[1]);
            Double_t timh = TMath::Min (time,mite->first-0.001);
            timh = TMath::Max (timh,mitb->first);
            mithit = hitMap.lower_bound(timh);
            if (mithit != hitMap.end()) hitok = &mithit->second;
            else hitok = &hitMap.rbegin()->second;
          }
          lay0 = lay;
          secGeo->Local2Global(secGeo->Sector(padID),p3loc,p3);
          if (secGeo->Global2Local(p3, p30) < 0) continue; // cross-check
          MpdTpcHit *hitp = AddRawHit(nHits++, padID, p3, p3err, hitok->GetRefIndex(), id);
          //MpdTpcHit *hitp = new ((*fHitArray)[nHits++]) MpdTpcHit(*hitok); // copy constructor does not work
          hitp->SetLayer(lay);
          hitp->SetLocalPosition(p3loc); // point position
          hitp->SetPosition(p3);
          hitp->SetEnergyLoss(hitok->GetEnergyLoss());
          hitp->SetStep(hitok->GetStep());
          hitp->SetModular(1); // modular geometry flag
          hitp->SetLength(hitok->GetLength());
          // Compute correct step and energy loss (angular dependence) - if large step
          if (hitp->GetStep() > maxStep) {
            Double_t phi = secGeo->SectorAngle(isec0);
            //Double_t phi = secGeo->SectorAngle(secGeo->Sector(padID));
            TVector3 sec3 (TMath::Cos(phi), TMath::Sin(phi), 0.0);
            Double_t padh = (lay < secGeo->NofRowsReg(0)) ? secGeo->PadHeight(0) : secGeo->PadHeight(1);
            TpcPoint *point = (TpcPoint*) fPointArray->UncheckedAt(hitok->GetRefIndex());
            TVector3 mom3;
            point->Momentum(mom3);
            Double_t cosAngle = TMath::Cos (mom3.Angle(sec3));
            //cout << isec0 << " " << lay << " " << phi << " " << hitp->GetStep() << " " << cosAngle << endl;
            Double_t step = padh / TMath::Abs(cosAngle);
            step = TMath::Min (step, 3.5); //AZ-060822
            //AZ-060822 hitp->SetStep(TMath::Min(step, 3.5));
            //AZ-060822 hitp->SetEnergyLoss(hitp->GetEnergyLoss() * hitp->GetStep());
            hitp->SetEnergyLoss (hitp->GetEnergyLoss() / hitp->GetStep() * step); //AZ-060822
            hitp->SetStep(step); //AZ-060822
          }
        }
        for (Int_t j = 0; j < 4; ++j) xyzloct[j].clear();
        isec0 = isec;
        idir0 = idir;
        delete yt;
        delete yx;
        delete yz;
      } // if (isec != isec0 || idir != idir0)

      if (isec == 30) break;
      hit.LocalPosition(p3loc);
      for (Int_t j = 0; j < 3; ++j) xyzloct[j].push_back(p3loc[j]);
      xyzloct[3].push_back(mit1->first);
    } // for (map<Double_t,MpdTpcHit>::iterator mit1 = hitMap.begin();

  } // for (map<Int_t,map<Double_t,Int_t> >::iterator mit = idmap.begin();

}

//___________________________________________________________________________

ClassImp(MpdTpcHitProducer)
