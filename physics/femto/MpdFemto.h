#ifndef MPDFEMTO_H
#define MPDFEMTO_H 1

#include <iostream>
#include <TNamed.h>
#include <TFile.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <MpdEvent.h>
#include <FairMCTrack.h>
#include <MpdTrack.h>
#include <TMath.h>
#include <TParticle.h>
#include <TParticlePDG.h>
#include <TRandom.h>
#include <TLorentzVector.h>
#include <MpdFemtoHistos.h>

using namespace std;
using namespace TMath;

class MpdFemto : public TNamed {
public:
    MpdFemto();
    MpdFemto(const Char_t* fname);
    virtual ~MpdFemto();

    // Getters

    Int_t GetPdgCode() {
        return fPDG;
    }

    Int_t GetEntriesNum() {
        return fDstTree->GetEntries();
    }
    
    MpdFemtoHistos* GetHistos() {
        return fHisto;
    }
    
    Float_t GetQinv() {
        return fQinv;
    }

    // Setters

    void SetPdgCode(Int_t val) {
        fPDG = val;
    }

    void SetEtaCuts(Float_t low, Float_t up) {
        fEtaCutLow = low;
        fEtaCutUp = up;
    }

    void SetPtCuts(Float_t low, Float_t up) {
        fPtCutLow = low;
        fPtCutUp = up;
    }
    
    void SetKtCuts(Float_t low, Float_t up) {
        fKtCutLow = low;
        fKtCutUp = up;
    }

    void SetSourceSize(Float_t size) {
        fSourceSize = size;
    }
    
    void SetNumMixedEvents(Int_t num) {
        fMixedEvents = num;
    }
    
    void SetQinv(Float_t qinv) {
        fQinv = qinv;
    }
    
    void SetNbins(Int_t val) {
        fBins = val;
    }

    void SetUpLimit(Float_t xUp) {
        fxUp = xUp;
    }
    
    void MakeCFs_1D();


private:
    MpdFemtoHistos* fHisto;
    
    void ReadEvent(Int_t);

    Int_t fPDG;
    Float_t fMass;
    const Char_t* fFilename;
    TDatabasePDG* fPartTable;
    TParticlePDG* fMassPart;
    
    Float_t fQinv;
    Float_t fKtCutLow;
    Float_t fKtCutUp;
    Float_t fEtaCutLow;
    Float_t fEtaCutUp;
    Float_t fPtCutLow;
    Float_t fPtCutUp;
    Float_t fSourceSize;
    Int_t fMixedEvents;
    
    Int_t fBins;
    Float_t fxUp;

    TChain* fDstTree;
    MpdEvent* fMpdEvent;

    TClonesArray* fMcTracks;
    TClonesArray* fRecoTracks;
    TClonesArray* fFemtoContainerReco;
    TClonesArray* fFemtoContainerMc;

    MpdTrack* fMpdTrackReco;
    FairMCTrack* fMpdTrackMc;
    
    inline Float_t EposFemtoQinv4vec(TLorentzVector first, TLorentzVector second) {
        return Abs((first - second).M());
    }
    
    Float_t EposFemtoWeightQS(TLorentzVector, TLorentzVector, TLorentzVector, TLorentzVector); 

    ClassDef(MpdFemto, 1)
};

#endif