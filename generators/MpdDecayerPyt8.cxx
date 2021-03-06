// -------------------------------------------------------------------------
// -----                 MpdDecayerPyt8 header file                    -----
// -----                      Created 23/07/2020                       -----
// -----                          A. Zinchenko                         -----
// -----             External decayer for MPD using Pythia8            -----
// -----                (adapted from TPythia8Decayer)                 -----
// -------------------------------------------------------------------------

#include "MpdDecayerPyt8.h"

#include <Pythia8/Pythia.h>

#include <TClonesArray.h>
#include <TDatabasePDG.h>
#include <TF1.h>
#include <TLorentzVector.h>
#include <TObjString.h>
#include <TParticle.h>
#include <TPythia8.h>
#include <TString.h>
#include <TSystem.h>
#include <TVirtualMC.h>

using namespace Pythia8; 

class MyRandomClass : public RndmEngine
{
  double flat() { return gRandom->Rndm(); }
};
  
ClassImp(MpdDecayerPyt8);

MpdDecayerPyt8* MpdDecayerPyt8::fgInstance = 0;
 
////////////////////////////////////////////////////////////////////////////////
/// Get the singleton object.

MpdDecayerPyt8* MpdDecayerPyt8::Instance()
{
  if (!fgInstance) fgInstance = new MpdDecayerPyt8;
  return fgInstance;
}

////////////////////////////////////////////////////////////////////////////////
///constructor

MpdDecayerPyt8::MpdDecayerPyt8():
  fPythia8(new TPythia8()),
  fDebug(0),
  fGlobalPolar(0),
  fLambda(NULL),
  fRandom(NULL)
{

  fPythia8->Pythia8()->readString("SoftQCD:elastic = on");

  // Disable cascade decays in one go
  fPythia8->Pythia8()->readString("ParticleDecays:limitRadius = on");
  fPythia8->Pythia8()->readString("ParticleDecays:rMax = 0.");

  fPythia8->Pythia8()->particleData.readString("59:m0 = 100.1396"); //??? - to avoid crash with Geant4
  
  fPythia8->Pythia8()->particleData.list(223);

  // Modify branching ratios
  ChangeBranchings();

  // Random number generator
  RndmEngine *rndm = new MyRandomClass();
  fPythia8->Pythia8()->setRndmEnginePtr(rndm);
    
  fPythia8->Pythia8()->init();

  Init();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the decayer

void MpdDecayerPyt8::Init()
{

  static Bool_t init = kFALSE;
  if (init) return;
  init = kTRUE;
  //ForceDecay();
  fParticles = new TClonesArray("TParticle");

  // Lambda branching to p+\pi-
  fPythia8->Pythia8()->particleData.list(3122);
  //ParticleDataEntry* part = fPythia8->Pythia8()->particleData.particleDataEntryPtr(3122); // Lambda
  fLambda = fPythia8->Pythia8()->particleData.particleDataEntryPtr(3122); // Lambda
  DecayChannel& channel = fLambda->channel(0);
  fBranch = channel.bRatio();

  // Random number generator
  fRandom = new TRandom(0); // time-dependent seed
  //fRandom = gRandom;

  //fPythia8->Pythia8()->particleData.isResonance(113,kTRUE);
  fPythia8->Pythia8()->particleData.list(113);
}

////////////////////////////////////////////////////////////////////////////////
/// Decay a single particle

void MpdDecayerPyt8::Decay(Int_t pdg, TLorentzVector* p)
{

  //{ cout << " !!! Out !!! " << endl; exit(0); }
  // Reset internal particle container
  fParticles->Delete();
  fSourceFlag = kPythia;
  fMother.SetXYZT(0,0,0,0);
  
  if (!p) return;

  TRandom *saveRandom = gRandom;
  gRandom = fRandom;
    
  TParticle *part = gMC->GetStack()->GetCurrentTrack();

  if (fMothersPdg.find(pdg) == fMothersPdg.end()) {
    // Particle is not defined
    new ((*fParticles)[0]) TParticle(*part); // store mother particle
    fSourceFlag = kCustom;
    return;
  }

  TVector3 polar;
  part->GetPolarisation(polar);
  Bool_t polarFlag = kFALSE;

  if (TMath::Abs(polar.X()) > 0.0001 || TMath::Abs(polar.Z()) > 0.0001) {
    // Polarized lambda - simulate anysotropic decay only to p + \pi-
    polarFlag = kTRUE;
    if (gRandom->Rndm() < fBranch) {
      // Anysotropic decay
      //cout << " ----------------- " << fBranch << endl;
      new ((*fParticles)[0]) TParticle(*part); // store mother particle
      Gdecay (pdg, p);
      gRandom = saveRandom;
      return;
    } else {
      //cout << " ++++++++++ " << fBranch << endl;
      // Force the other channels
      DecayChannel& channel = fLambda->channel(0);
      channel.bRatio(0.0);
    }
  }
    
  ClearEvent();
  AppendParticle(pdg, p);
  Int_t idPart = fPythia8->Pythia8()->event[0].id();
  fPythia8->Pythia8()->particleData.mayDecay(idPart,kTRUE);
  fPythia8->Pythia8()->moreDecays();
  if (fDebug > 0) fPythia8->EventListing();
  if (polarFlag) {
    DecayChannel& channel = fLambda->channel(0);
    channel.bRatio(fBranch);
  }

  gRandom = saveRandom;
}

////////////////////////////////////////////////////////////////////////////////
///import the decay products into particles array

Int_t MpdDecayerPyt8::ImportParticles(TClonesArray *particles)
{

  Int_t npart = 0;
  
  if (fSourceFlag == kPythia) {
    fParticles->Delete();
    npart = fPythia8->ImportParticles(fParticles, "All");
    TLorentzVector lvDaugh;
    
    for (Int_t j = 0; j < npart; ++j) {
      TParticle *part = (TParticle*) fParticles->UncheckedAt(j);
      part->ProductionVertex(lvDaugh);
      lvDaugh += fMother;
      part->SetProductionVertex(lvDaugh);
      new ((*particles)[j]) TParticle(*part);
    }
  } else {
    npart = fParticles->GetEntriesFast();
    for (Int_t j = 0; j < npart; ++j) new ((*particles)[j]) TParticle(*((TParticle*)fParticles->UncheckedAt(j)));
  }

  //return (fPythia8->ImportParticles(particles, "All"));
  return npart;
}

////////////////////////////////////////////////////////////////////////////////
/// Set forced decay mode

void MpdDecayerPyt8::SetForceDecay(Int_t /*type*/)
{
   printf("SetForceDecay not yet implemented !\n");
}
////////////////////////////////////////////////////////////////////////////////
/// ForceDecay not yet implemented

void MpdDecayerPyt8::ForceDecay()
{
  //printf("ForceDecay not yet implemented !\n");
}
////////////////////////////////////////////////////////////////////////////////

Float_t MpdDecayerPyt8::GetPartialBranchingRatio(Int_t /*ipart*/)
{
   return 0.0;
}
////////////////////////////////////////////////////////////////////////////////
///return lifetime in seconds of teh particle with PDG number pdg

Float_t MpdDecayerPyt8::GetLifetime(Int_t pdg)
{
   return (fPythia8->Pythia8()->particleData.tau0(pdg) * 3.3333e-12) ;
}

////////////////////////////////////////////////////////////////////////////////
///to read a decay table (not yet implemented)

void    MpdDecayerPyt8::ReadDecayTable()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Append a particle to the stack

void MpdDecayerPyt8::AppendParticle(Int_t pdg, TLorentzVector* p)
{
   fPythia8->Pythia8()->event.append(pdg, 11, 0, 0, p->Px(), p->Py(), p->Pz(), p->E(), p->M());
}


////////////////////////////////////////////////////////////////////////////////
/// Clear the event stack

void MpdDecayerPyt8::ClearEvent()
{
   fPythia8->Pythia8()->event.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Simulate two-body decay process of Lambda-hyperon to p+\pi-

void MpdDecayerPyt8::Gdecay(Int_t idpart, TLorentzVector* p)
{
        
  Double_t pcm[2][4] = {{0},{0}};
  Double_t xm0 = p->M(), xm1 = TDatabasePDG::Instance()->GetParticle(2212)->Mass();
  Double_t xm2 = TDatabasePDG::Instance()->GetParticle(-211)->Mass();
  
  Gdeca2 (xm0, xm1, xm2, pcm);

  // First, second decay products.
    
  TLorentzVector daughter1, daughter2;
  daughter1.SetPxPyPzE (pcm[0][0], pcm[0][1], pcm[0][2], pcm[0][3]);
  daughter2.SetPxPyPzE (pcm[1][0], pcm[1][1], pcm[1][2], pcm[1][3]);

  //Boost to lab frame
  //TVector3 boost = p->BoostVector(); //AZ - this is wrong !!!!
  TVector3 boost (0, 0, p->P()/p->E());
  daughter1.Boost(boost);
  daughter2.Boost(boost);
  daughter1.Transform(fRotation);
  daughter2.Transform(fRotation);

  TLorentzVector pos;
  gMC->TrackPosition(pos);
  //pos.Print();

  Int_t npart = fParticles->GetEntriesFast();
  new ((*fParticles)[npart++]) TParticle(2212, 1, 1, -1, 0, 0, daughter1, pos);
  new ((*fParticles)[npart]) TParticle(-211, 1, 1, -1, 0, 0, daughter2, pos);

  fSourceFlag = kCustom;

  // Modify mother particle parameters
  TParticle *mother = (TParticle*) fParticles->UncheckedAt(0);
  mother->SetFirstMother(1);
  mother->SetLastMother(-1);
  mother->SetFirstDaughter(2);
  mother->SetLastDaughter(3);
  mother->SetStatusCode(11);
}   

////////////////////////////////////////////////////////////////////////////////
/// Simulate two-body decay process with anisotropic angular distribution in CMS.

void MpdDecayerPyt8::Gdeca2(Double_t xm0, Double_t xm1, Double_t xm2, Double_t pcm[2][4])
{

  Double_t random[2], pvert[3], costh = 0.0, sinth = 0.0, phi = 0.0;
  
  Double_t e1 = (xm0 * xm0 + xm1 * xm1 - xm2 * xm2) / (2. * xm0);
  Double_t p1 = TMath::Sqrt (TMath::Abs ((e1 - xm1) * (e1 + xm1)));

  static TRandom *myRandom  = NULL;
  if (!myRandom) { myRandom = new TRandom(); myRandom->SetSeed(0); }
  TRandom *oldRandom = gRandom;
  gRandom = myRandom;

  gRandom->RndmArray (2, random);
    
  // Sanity check - should not happen (legacy code)
  TParticle *part = gMC->GetStack()->GetCurrentTrack();
  if (part->GetPdgCode() != 3122) { cout << " ??? Not Lambda - exit " << part->GetPdgCode() << endl; exit(0); }

  TLorentzVector lorV;
  part->Momentum(lorV);
  lorV.Vect().GetXYZ(pvert);
  
  TVector3 polar;
  part->GetPolarisation(polar);
  if (fGlobalPolar) polar.SetMag(part->GetWeight()); // particle polarization value is passed as its weight 
  
  if (TMath::Abs(polar.X()) < 0.0001 && TMath::Abs(polar.Z()) < 0.0001 && TMath::Abs(polar.Y()) < 0.0001) {
    costh = 2. * random[0] - 1.0;
    phi = TMath::TwoPi() * random[1];
  } else {
    // Anisotropic decay angular distribution (0.5*(1+alpha*P*cos)).
    //Anisotropy (pvert, random, polar.X(), phi, costh);
    Anisotropy (pvert, random, polar, phi, costh);
  }

  if (TMath::Abs(costh) >= 1.0) costh = TMath::Sign (1.0, costh);
  else sinth = TMath::Sqrt ((1. - costh) * (1. + costh));
		
  // Polar co-ordinates to momentum components.

  pcm[0][0] = p1 * sinth * TMath::Cos(phi);
  pcm[0][1] = p1 * sinth * TMath::Sin(phi);
  pcm[0][2] = p1 * costh;
  pcm[0][3] = e1;

  // Generate second decay product.

  pcm[1][0] = -pcm[0][0];
  pcm[1][1] = -pcm[0][1];
  pcm[1][2] = -pcm[0][2];
  pcm[1][3] = TMath::Sqrt (pcm[1][0] * pcm[1][0] + pcm[1][1] * pcm[1][1] + pcm[1][2] * pcm[1][2] + xm2 * xm2);

  gRandom = oldRandom;
}

////////////////////////////////////////////////////////////////////////////////
/// Simulate anisotropic (due to polarization) decay of lambda

//void MpdDecayerPyt8::Anisotropy (Double_t *pvert, Double_t *rndm, Double_t polar, Double_t &phi, Double_t &costh)
void MpdDecayerPyt8::Anisotropy (Double_t *pvert, Double_t *rndm, TVector3& polar3, Double_t &phi, Double_t &costh)
{
  // Simulate anisotropic (due to polarization) decay of lambda

  //exit(0);
  //std::ofstream outfile ("costh.txt");
  //freopen ("costh.txt", "w", stdout);

  Double_t polar = 0.0;
  if (fGlobalPolar) polar = polar3.Mag(); // global polarization - CHECK THIS
  else polar = polar3.X(); // transverse polar.

  //const Double_t alpha = 0.642;
  //static TF1 f("f","1+0.642*[0]*x",-1,1); 
  static TF1 f("f","1+0.732*[0]*x",-1,1); // AZ 10.06.21 - new value
  f.SetParameter(0,polar);
  f.SetNpx(1000);

  Double_t costhe = f.GetRandom();

  Double_t sinth = 0.0;
  if (TMath::Abs(costhe) >= 1.0) costhe = TMath::Sign (1.0,costhe);
  else sinth = TMath::Sqrt (1.0 - costhe * costhe);
  phi = TMath::TwoPi() * rndm[1];
  //std::cout << " Cos: " << costhe << " " << phi << " " << pvert[0] << " " << pvert[1] << " " << pvert[2] << std::endl;

  // Compute normal vector
  TVector3 beam(0.0, 0.0, 1.0);
  TVector3 lambda(pvert[0], pvert[1], pvert[2]);
  TVector3 norm;
  if (fGlobalPolar) norm = polar3.Unit(); // global polarization
  else norm = beam.Cross(lambda).Unit(); // transverse polarization
    
  // Unit vector with theta and phi w.r.t. normal
  TVector3 unit(sinth*TMath::Cos(phi), sinth*TMath::Sin(phi), costhe);

  // Coordinate system transformation
  // (from lambda to lab)
  TVector3 lambU = lambda.Unit(), lambZ(0,0,1), lambX(1,0,0), lambY(0,1,0);
  lambZ.RotateUz(lambU);
  lambX.RotateUz(lambU);
  lambY.RotateUz(lambU);
  TRotation rotL;
  rotL.RotateAxes(lambX,lambY,lambZ); // transformation to lab. system
  fRotation = rotL;
  rotL.Invert(); // from lab. to lambda
  unit.RotateUz(norm); // to lab. system
  unit.Transform(rotL); // to lambda system
  /*
  TVector3 normZ(0,0,1), normX(1,0,0), normY(0,1,0);
  normZ.RotateUz(norm);
  normX.RotateUz(norm);
  normY.RotateUz(norm);
  TRotation rotN;
  rotN.RotateAxes(normX,normY,normZ); // transformation to lab. system
  unit.Transform(rotN); // to lab. system
  unit.Transform(rotL); // to lambda system
  */
  costh = TMath::Cos(unit.Theta());
  phi = unit.Phi();
  //std::cout << costh << " " << phi << std::endl;
  //outfile << i << " " << costh << endl;
  //outfile.close();
}

///////////////////////////////////////////////////////////////////////////////
/// Add PDG of particle to be decayed by this package

void MpdDecayerPyt8::AddMotherPdg(Int_t pdg) 
{ 

  if (fMothersPdg.find(pdg) != fMothersPdg.end()) return;
  fMothersPdg.insert(pdg);
  gMC->SetUserDecay(pdg);
}

////////////////////////////////////////////////////////////////////////////////
/// Decay a TParticle 

void MpdDecayerPyt8::Decay(TParticle* part)
{
  // Decay function
  
  //{ cout << " !!! Out !!! " << endl; exit(0); }
  // Reset internal particle container
  fParticles->Delete();
  fSourceFlag = kPythia;

  /*
  if (!p) return;
  p->ProductionVertex(fMother);
  TPythia6 *pythia = TPythia6::Instance();
  
  pythia->Py1ent(0, p->GetPdgCode(), p->Energy(), p->Theta(), p->Phi());
  pythia->GetPrimaries();
  */
  if (!part) return;
  //part->Print();

  TRandom *saveRandom = gRandom;
  gRandom = fRandom;
    
  part->ProductionVertex(fMother);

  Int_t pdg = part->GetPdgCode();
  TLorentzVector p;
  part->Momentum(p);

  if (part->GetFirstMother() < 0) {
    // Primary particle - select mass
    Double_t mass = fPythia8->Pythia8()->particleData.mSel(pdg);
    //cout << "mmm " << mass << " " << part->GetFirstMother() << endl;
    p.SetE (TMath::Sqrt (p.P()*p.P() + mass*mass));
  }
  //p.Print();

  ClearEvent();
  AppendParticle(pdg, &p);
  Int_t idPart = fPythia8->Pythia8()->event[0].id();
  fPythia8->Pythia8()->particleData.mayDecay(idPart,kTRUE);
  fPythia8->Pythia8()->moreDecays();
  if (fDebug > 0) fPythia8->EventListing();

  gRandom = saveRandom;
} 

//__________________________________________________________________________

void MpdDecayerPyt8::ChangeBranchings()
{
  // Change branching ratios of some decay channels
  // (using description file)

  TString path = gSystem->ExpandPathName("$VMCWORKDIR");
  path += "/gconfig/MpdDecayConfig.txt";
  ifstream fin(path.Data());

  TString chline;
  TObjArray *tokens = NULL;

  while(1) {
    chline.ReadLine(fin);
    if (fin.eof()) break;
    cout << chline << endl;
    if (chline.Contains("#")) continue; // comment
    // Tokenize line                                                            
    tokens = chline.Tokenize(";"); // ";" - channel terminator / separator
    ChangeParticleBr(tokens);
    if (tokens) { tokens->Delete(); delete tokens; tokens = NULL; }
  }
}

//__________________________________________________________________________

void MpdDecayerPyt8::ChangeParticleBr(TObjArray *channels)
{
  // Change branching ratios of selected channels for single particle 

  Int_t nchan = channels->GetEntriesFast(); // number of channels to be changed

  Int_t pdg = ((TObjString*)channels->UncheckedAt(0))->String().Atoi();
  //cout << pdg << endl;
  ParticleDataEntryPtr part = fPythia8->Pythia8()->particleData.particleDataEntryPtr(pdg); // mother particle
  Int_t ndecays = part->sizeChannels();
  
  TObjArray *tokens = NULL;
  Double_t changedBrs = 0.0, unchangedBrs = 1.0;
  vector<Int_t> decayInds;
  vector<Double_t> bRatios;
  
  for (Int_t ich = 0; ich < nchan; ++ich) {
    TString chline = ((TObjString*)channels->UncheckedAt(ich))->String();
    cout << chline << endl;
    // Tokenize channel
    tokens = chline.Tokenize(":");

    // Get daughters' PDGs
    Int_t nprongs = tokens->GetEntriesFast() - 2;
    vector<Int_t> pdgs;
    for (Int_t id = 0; id < nprongs; ++id) {
      pdgs.push_back(((TObjString*)tokens->UncheckedAt(1+id))->String().Atoi());
    }
    TString scale = ((TObjString*)tokens->Last())->String();
    scale.Remove(0,1);
    Double_t scaleBr = scale.Atof();

    // Pick channel requested
    for (Int_t idecay = 0; idecay < ndecays; ++idecay) {
      DecayChannel& channel = part->channel(idecay);
      if (channel.multiplicity() != nprongs) continue;
      // Check daughters
      Int_t ok = 0;
      for (Int_t id = 0; id < nprongs; ++id) {
	if (channel.product(id) != pdgs[id]) break;
	++ok;
      }
      if (ok != nprongs) continue;
      changedBrs += channel.bRatio() * scaleBr;
      unchangedBrs -= channel.bRatio();
      decayInds.push_back(idecay);
      bRatios.push_back (channel.bRatio() * scaleBr);
      //cout << channel.bRatio() << endl;
    }
    //cout << changedBrs << endl;
    
    if (tokens) { tokens->Delete(); delete tokens; tokens = NULL; }
  } // for (Int_t ich = 0; ich < nchan;

  part->rescaleBR ((1.0 - changedBrs) / unchangedBrs);

  for (Int_t ich = 0; ich < nchan; ++ich) {
    DecayChannel& channel = part->channel(decayInds[ich]);
    channel.bRatio(bRatios[ich]);
  }

  // Check
  Double_t totalBr = 0;
  for (Int_t idecay = 0; idecay < ndecays; ++idecay) {
    DecayChannel& channel = part->channel(idecay);
    totalBr += channel.bRatio();
  }
  cout << " Total branching after rescaling: " << totalBr << endl;
}
