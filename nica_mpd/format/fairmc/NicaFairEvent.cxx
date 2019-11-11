/*
 * NicaEventFair.cxx
 *
 *  Created on: 05-07-2014
 *      Author: Daniel Wielanek
 *		E-mail: daniel.wielanek@gmail.com
 *		Warsaw University of Technology, Faculty of Physics
 */
#include "NicaFairEvent.h"
#include "NicaFairEventInterface.h"
#include "FairRootManager.h"
#include <iostream>

NicaFairEvent::NicaFairEvent(): NicaMCEvent("NicaFairTrack"){
}

NicaFairEvent::NicaFairEvent(TString trackname): NicaMCEvent(trackname){
}

NicaFairEvent::NicaFairEvent(const NicaFairEvent& other) :NicaMCEvent(other){
}

void NicaFairEvent::Update() {
	FairMCEventHeader *event = (FairMCEventHeader*)((NicaFairEventInterface*)fSource)->fEvent;
	TClonesArray *tracks = (TClonesArray*)((NicaFairEventInterface*)fSource)->fFairTracks->GetArray();
	fB = event->GetB();
	fVertex->SetXYZT(event->GetX(),event->GetY(),event->GetZ(),event->GetT());
	fTracks->Clear();
	fTotalTracksNo = tracks->GetEntriesFast();
	fTracks->ExpandCreateFast(fTotalTracksNo);
	for(int i=0;i<tracks->GetEntriesFast();i++){
		FairMCTrack *track = (FairMCTrack*)tracks->UncheckedAt(i);
		NicaMCTrack *mc = (NicaMCTrack*)fTracks->UncheckedAt(i);
		mc->GetMomentum()->SetPxPyPzE(track->GetPx(),track->GetPy(),track->GetPz(),track->GetEnergy());
		mc->SetMotherIndex(track->GetMotherId());
		if(track->GetMotherId()==-1){
			mc->SetPrimary(kTRUE);
		}else{
			mc->SetPrimary(kFALSE);
		}
		mc->SetCharge(CalculateCharge(track->GetPdgCode()));
		mc->SetPdg(track->GetPdgCode());
		mc->GetStartPosition()->SetXYZT(track->GetStartX(),track->GetStartY(),track->GetStartZ(),track->GetStartT());
		mc->GetLink()->Clear();
		mc->GetLink()->SetLink(0,i);
	}
}

void NicaFairEvent::Clear(Option_t* opt) {
	NicaMCEvent::Clear(opt);
}

void NicaFairEvent::Print() {
}

void NicaFairEvent::CreateSource() {
	std::cout<<"Create source"<<std::endl;
	fSource = new NicaFairEventInterface();
}

NicaFairEvent::~NicaFairEvent() {
}

TString NicaFairEvent::GetFormatName() const{
	return "FairMCFormat";
}

Bool_t NicaFairEvent::ExistInTree() const {
	FairRootManager *manager = FairRootManager::Instance();
	Int_t header = manager->CheckBranch("MCEventHeader.")+ manager->CheckBranch("EventHeader.");
	if(header >1) header = 1;
	Int_t tracks = manager->CheckBranch("MCTrack");
	if((header+tracks)==2){
		return kTRUE;
	}
	return kFALSE;
}
