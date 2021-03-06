#include "MpdUnigenGenerator.h"

MpdUnigenGenerator::MpdUnigenGenerator() :
FairGenerator(),
fEventNumber(0),
fInFile(nullptr),
fInTree(nullptr),
fEvent(nullptr),
fParticle(nullptr),
fEventPlaneSet(kFALSE), 
fSpectatorsON(kFALSE),
fPhiMin(0.), 
fPhiMax(0.) { 
}

MpdUnigenGenerator::MpdUnigenGenerator(TString fileName, Bool_t isSpectator) :
FairGenerator(),
fEventNumber(0),
fInFile(nullptr),
fInTree(nullptr),
fEvent(nullptr),
fParticle(nullptr),
fEventPlaneSet(kFALSE), 
fPhiMin(0.), 
fPhiMax(0.) { 
  std::cout << "-I MpdUnigenGenerator: Opening input file " << fileName.Data() << std::endl;

  fInFile = new TFile(fileName,"read");
  if (!fInFile){
    Fatal("MpdUnigenGenerator", "Cannot open input file.");
    exit(1);
  }
  fInTree = (TTree*) fInFile->Get("events");

  if (!fInTree){
    Fatal("MpdUnigenGenerator", "Cannot open TTree from the file.");
    exit(1);
  }

  fSpectatorsON = isSpectator;
  if (fSpectatorsON)
    std::cout << "-I MpdUnigenGenerator: Spectators ON" << std::endl;
  
  Long64_t nEntries = fInTree->GetEntries();
  std::cout << "-I MpdUnigenGenerator: Number of entries in tree: " << nEntries << std::endl;
  fInTree->SetBranchAddress("event", &fEvent);

  if (fSpectatorsON){
    RegisterIons();
  }
}

Bool_t MpdUnigenGenerator::ReadEvent(FairPrimaryGenerator* primGen){
  // Check for input file
  if (!fInFile) {
    cout << "-E- MpdUnigenGenerator: Input file not open! " << endl;
    return kFALSE;
  }

  // Check for primary generator
  if (!primGen) {
    cout << "-E- MpdUnigenGenerator::ReadEvent: "
         << "No PrimaryGenerator!" << endl;
    return kFALSE;
  }

  fInTree->GetEntry(fEventNumber);
  std::cout << "-I- MpdUnigenGenerator: Event " << fEventNumber << std::endl;

  Double_t phi = fEvent->GetPhi();
  Double_t dphi = 0.;
  // ---> Generate rotation angle
  if (fEventPlaneSet) {
    gRandom->SetSeed(0);
    dphi = gRandom->Uniform(fPhiMin, fPhiMax);
    phi += dphi;
  }

  FairMCEventHeader* header = primGen->GetEvent();
  if (header && (!header->IsSet())) {
    header->SetEventID(fEvent->GetEventNr());
    header->SetB(fEvent->GetB());
    header->MarkSet(kTRUE);
    header->SetRotZ(phi);
  }

  UInt_t nTracks = fEvent->GetNpa();
  Int_t ionPdg;
  for (Int_t iTrack=0; iTrack<nTracks; iTrack++){
    fParticle = fEvent->GetParticle(iTrack);
    if (!fParticle)
      continue;
    
    Double_t px = fParticle->Px();
    Double_t py = fParticle->Py(); 
    
    if (fEventPlaneSet) {
      Double_t pt = TMath::Sqrt(px * px + py * py);
      Double_t azim = TMath::ATan2(py, px);
      azim += dphi;
      px = pt * TMath::Cos(azim);
      py = pt * TMath::Sin(azim);
    }

    if (fParticle->GetPdg() < 1e9) {
      primGen->AddTrack(fParticle->GetPdg(), px, py, fParticle->Pz(), 0., 0., 0.);
    } else {
      // Since hyper-nuclei are not (yet) supported by FairRoot, their PDG
      // is replaced by that of the non-strange analogue.
      ionPdg = fParticle->GetPdg();
      if (GetIonLambdas(fParticle->GetPdg())) {
        ionPdg = fParticle->GetPdg() - GetIonLambdas(fParticle->GetPdg()) * kPdgLambda;
        std::cout << "-W- MpdUnigenGenerator: Replacing hypernucleus (PDG " << fParticle->GetPdg() << ") by PDG " << ionPdg << std::endl;
      }
      // Charged ions can be registered
      if (GetIonCharge(ionPdg)) {
        primGen->AddTrack(ionPdg, px, py, fParticle->Pz(), 0., 0., 0.);
      } else {
        // Neutral ions are not supported by GEANT4.
        // They are thus decomposed into neutrons (as an approximation)
        Int_t mass = GetIonMass(ionPdg);
        for (Int_t iNeutron = 0; iNeutron < mass; iNeutron++) {
          Double_t pxNeutron = px/(Double_t)mass;
          Double_t pyNeutron = py/(Double_t)mass;
          Double_t pzNeutron = fParticle->Pz()/(Double_t)mass;
          Double_t eNeutron = fParticle->E()/(Double_t)mass;

          primGen->AddTrack(2112, pxNeutron, pyNeutron, pzNeutron, 0., 0., 0.);
        }
        std::cout << "-W- MpdUnigenGenerator: Neutral fragment (PDG " << ionPdg << ") is split into " << mass << " neutrons." << std::endl;
      } // Neutral ion ?
    } // ion ?
  }
  fEventNumber++;
  return kTRUE;
}

MpdUnigenGenerator::~MpdUnigenGenerator(){
  fInFile->Close();

  delete fInTree;
  delete fEvent;
  delete fParticle;
}

Int_t MpdUnigenGenerator::RegisterIons(void) {
  std::cout << "-I- MpdUnigenGenerator: Registering ions..." << std::endl;
  UParticle* particle {nullptr};
  Int_t nIons {0};
  fIonMap.clear();

  for (Int_t iEvent=0; iEvent<fInTree->GetEntries(); iEvent++) {
    fInTree->GetEntry(iEvent);

    for (Int_t iParticle=0; iParticle<fEvent->GetNpa(); iParticle++) {
      particle = fEvent->GetParticle(iParticle);

      Int_t pdgCode = particle->GetPdg();
      if (pdgCode > 1e9) {  // ion
        // For ions the PDG code is +-10LZZZAAAI,
        // where A = n_Lambda + n_proton + n_neutron, Z = n_proton, L = n_Lambda
        // and I = 0 - ground state, I>0 - excitations
        const Int_t nLambda = GetIonLambdas(pdgCode);
        const Int_t chrg    = GetIonCharge(pdgCode);
        const Int_t mass    = GetIonMass(pdgCode);

        // Neutral ions are skipped (not present in G4IonTable)
        if (chrg == 0) continue;

        // Add ion to ion map
        TString ionName = Form("Ion_%d_%d_%d", mass, chrg, nLambda);
        if (fIonMap.find(ionName) == fIonMap.end()) {  // new ion
          fIonMap[ionName] = new FairIon(ionName, chrg, mass, chrg);

          std::cout << "-I- MpdUnigenGenerator: Added new ion with PDG " << pdgCode
            << " (Charge " << chrg << ", Mass " << mass << ",  nLambda " << nLambda
            << ") from event " << iEvent << " at index " << iParticle << std::endl;

          nIons++;
        } //new ion ?
      } // ion ?
    } // particle loop
  } // event loop

  // Adding new ions to FairRunSim instance
  for (const auto& entry : fIonMap) {
    FairRunSim::Instance()->AddNewIon(entry.second);
  }

  return fIonMap.size();
}

ClassImp(MpdUnigenGenerator);
