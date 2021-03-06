#include "MpdGeneratorsFactory.h"

#include <TMath.h>
#include <TRandom.h>
#include <TDatabasePDG.h>

#include "MpdUrqmdGenerator.h"
#include "MpdVHLLEGenerator.h"
#include "Mpd3fdGenerator.h"
#include "FairParticleGenerator.h"
#include "FairIonGenerator.h"
#include "FairBoxGenerator.h"
#include "MpdPHSDGenerator.h"
#include "MpdLAQGSMGenerator.h"
#include "THadgen.h"
#include "MpdGetNumEvents.h"
#include "mpdloadlibs.C"


MpdGeneratorsFactory::MpdGeneratorsFactory()
{
	add<MpdUrqmdGenCreator>(MpdGeneratorType::URQMD);
	add<MpdVHLLEGenCreator>(MpdGeneratorType::VHLLE);
	add<MpdFLUIDGenCreator>(MpdGeneratorType::FLUID);
	//add<MpdPARTGenCreator>(MpdGeneratorType::PART);
	//add<MpdIONGenCreator>(MpdGeneratorType::ION);
	//add<MpdBOXGenCreator>(MpdGeneratorType::BOX);
	add<MpdHSDGenCreator>(MpdGeneratorType::HSD);
	add<MpdLAQGSMGenCreator>(MpdGeneratorType::LAQGSM);
	//add<MpdHADGENGenCreator>(MpdGeneratorType::HADGEN);
}
	
MpdGeneratorsFactory::~MpdGeneratorsFactory()
{
	for(CreatorsMap::iterator it = creators.begin(); it != creators.end(); ++it)
		delete it->second;
}

std::shared_ptr<MpdFactoryMadeGenerator> MpdGeneratorsFactory::create(MpdGenerator & Gen)
{
	typename CreatorsMap::iterator it = creators.find(Gen.genType);
	if (it != creators.end())
	{
		std::shared_ptr<FairGenerator> gen = std::shared_ptr<FairGenerator>(it->second->create(Gen.inFile, Gen.startEvent, Gen.nEvents));
		return std::make_shared<MpdFactoryMadeGenerator>(gen, it->second->postActions(Gen.inFile), Gen.inFile);
	}
	return std::shared_ptr<MpdFactoryMadeGenerator>(NULL);
}

template <class C> 
void MpdGeneratorsFactory::add(const MpdGeneratorType & GenType)
{
	typename CreatorsMap::iterator it = creators.find(GenType);
	if (it == creators.end())
		creators[GenType] = new C();
}

FairGenerator * MpdUrqmdGenCreator::create (TString & inFile, Int_t & nStartEvent, Int_t & nEvents)
{
		// ------- Urqmd  Generator
	if (!CheckFileExist(inFile)) return NULL;

	MpdUrqmdGenerator* urqmdGen = new MpdUrqmdGenerator(inFile);
	// Event plane angle will be generated by uniform distribution from min to max.
	// Angles are in degrees
	Float_t min = 0.0;
	Float_t max = 30.0;
	urqmdGen->SetEventPlane(min * TMath::DegToRad(), max * TMath::DegToRad());
	if (nStartEvent > 0) urqmdGen->SkipEvents(nStartEvent);

	// if nEvents is equal 0 then all events (start with nStartEvent) of the given file should be processed
	if (nEvents == 0)
		nEvents = MpdGetNumEvents::GetNumURQMDEvents(const_cast<char *>(inFile.Data())) - nStartEvent;

	return urqmdGen;
}

FairGenerator * MpdVHLLEGenCreator::create (TString & inFile, Int_t &, Int_t &)
{
	if (!CheckFileExist(inFile)) return NULL;
	
	MpdVHLLEGenerator* vhlleGen = new MpdVHLLEGenerator(inFile, kTRUE); // kTRUE corresponds to hydro + cascade, kFALSE -- hydro only
	vhlleGen->SkipEvents(0);
	return vhlleGen;
}

FairGenerator * MpdFLUIDGenCreator::create (TString & inFile, Int_t & nStartEvent, Int_t &)
{
	if (!CheckFileExist(inFile)) return NULL;
	
	Mpd3fdGenerator* fluidGen = new Mpd3fdGenerator(inFile);
    if (nStartEvent > 0) fluidGen->SkipEvents(nStartEvent);
    //fluidGen->SetPsiRP(0.); // set fixed Reaction Plane angle [rad] instead of random
    //fluidGen->SetProtonNumberCorrection(79./197.); // Z/A Au for Theseus 2018-03-17-bc2a06d
	return fluidGen;
}

/*FairGenerator * MpdPARTGenCreator::create (TString &, Int_t &, Int_t &)
{
	// ------- Particle Generator
	FairParticleGenerator* partGen = new FairParticleGenerator(211, 10, 1, 0, 3, 1, 0, 0);
	return partGen;
}*/

/*FairGenerator * MpdIONGenCreator::create (TString &, Int_t &, Int_t &)
{
	// ------- Ion Generator
	FairIonGenerator *fIongen = new FairIonGenerator(79, 197, 79, 1, 0., 0., 25, 0., 0., -1.);
	return fIongen;
}*/

/*FairGenerator * MpdBOXGenCreator::create (TString &, Int_t &, Int_t &)
{
	gRandom->SetSeed(0);
	// ------- Box Generator
	FairBoxGenerator* boxGen = new FairBoxGenerator(13, 100); // 13 = muon; 1 = multipl.
	boxGen->SetPRange(0.25, 2.5); // GeV/c //setPRange vs setPtRange
	boxGen->SetPhiRange(0, 360); // Azimuth angle range [degree]
	boxGen->SetThetaRange(0, 180); // Polar angle in lab system range [degree]
	boxGen->SetXYZ(0., 0., 0.); // mm o cm ??
	return boxGen;
}*/

FairGenerator * MpdHSDGenCreator::create (TString & inFile, Int_t & nStartEvent, Int_t & nEvents)
{
// ------- HSD/PHSD Generator
	if (!CheckFileExist(inFile)) return NULL;

	MpdPHSDGenerator *hsdGen = new MpdPHSDGenerator(inFile.Data());
	//hsdGen->SetPsiRP(0.); // set fixed Reaction Plane angle [rad] instead of random
	if (nStartEvent > 0) hsdGen->SkipEvents(nStartEvent);

	// if nEvents is equal 0 then all events (start with nStartEvent) of the given file should be processed
	if (nEvents == 0)
		nEvents = MpdGetNumEvents::GetNumPHSDEvents(const_cast<char *>(inFile.Data())) - nStartEvent;

	return hsdGen;
}

FairGenerator * MpdLAQGSMGenCreator::create (TString & inFile, Int_t & nStartEvent, Int_t & nEvents)
{
// ------- LAQGSM Generator
	if (!CheckFileExist(inFile)) return NULL;

	MpdLAQGSMGenerator* guGen = new MpdLAQGSMGenerator(inFile.Data(),kTRUE,0, 1+nStartEvent+nEvents);
	// kTRUE - for NICA/MPD, 1+nStartEvent+nEvents - search ions in selected part of file.
	if (nStartEvent > 0) guGen->SkipEvents(nStartEvent);

	// if nEvents is equal 0 then all events (start with nStartEvent) of the given file should be processed
	if (nEvents == 0)
		nEvents = MpdGetNumEvents::GetNumQGSMEvents(const_cast<char *>(inFile.Data())) - nStartEvent;

	return guGen;
}

std::function<Int_t(FairRunSim * fRun)> MpdLAQGSMGenCreator::postActions(TString & inFile)
{
    return [&inFile](FairRunSim * fRun)
    {
        TString Pdg_table_name = TString::Format("%s%s%c%s", gSystem->BaseName(inFile.Data()), ".g", (fRun->GetName())[6], ".pdg_table.dat");
        return (TDatabasePDG::Instance())->WritePDGTable(Pdg_table_name.Data());
    };
}

/*FairGenerator * MpdHADGENGenCreator::create (TString &, Int_t &, Int_t &)
{
	THadgen* hadGen = new THadgen();
	hadGen->SetRandomSeed(clock() + time(0));
	hadGen->SetParticleFromPdgCode(0, 196.9665, 79);
	hadGen->SetEnergy(6.5E3);
	MpdGeneralGenerator* generalHad = new MpdGeneralGenerator(hadGen);
	return generalHad;
}*/
