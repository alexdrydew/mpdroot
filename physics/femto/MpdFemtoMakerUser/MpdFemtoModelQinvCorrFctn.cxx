//
// One-dimensional correlation function for the model estimations
//

// MpdFemtoMaker headers
#include "MpdFemtoModelGausLCMSFreezeOutGenerator.h"
#include "MpdFemtoModelHiddenInfo.h"
#include "MpdFemtoModelManager.h"
#include "MpdFemtoPair.h"

// MpdFemtoMakerUser headers
#include "MpdFemtoModelQinvCorrFctn.h"

// ROOT headers
#include "TH1D.h"
#include "TString.h"

ClassImp(MpdFemtoModelQinvCorrFctn);

//_________________

MpdFemtoModelQinvCorrFctn::MpdFemtoModelQinvCorrFctn() :
MpdFemtoBaseCorrFctn(),
mManager(nullptr),
mNumeratorWeighted(nullptr),
mNumerator(nullptr),
mDenominatorWeighted(nullptr),
mDenominator(nullptr) {

    // Default constructor
    mNumeratorWeighted = new TH1D("ModelNumWeighted", "Weighted numerator;q_{inv} (GeV/c);dN/dq_{inv}", 80, 0., 0.8);
    mNumerator = new TH1D("ModelNum", "Unweighted numerator;q_{inv} (GeV/c);dN/dq_{inv}", 80, 0., 0.8);

    mDenominatorWeighted = new TH1D("ModelDenWeighted", "ModelNumTrueIdeal", 80, 0., 0.8);
    mDenominator = new TH1D("ModelDen", "ModelNumFakeIdeal", 80, 0., 0.8);

    mNumeratorWeighted->Sumw2();
    mNumerator->Sumw2();

    mDenominatorWeighted->Sumw2();
    mDenominator->Sumw2();
}

//_________________

MpdFemtoModelQinvCorrFctn::MpdFemtoModelQinvCorrFctn(const char *title, const Int_t& aNbins,
        const Double_t& aQinvLo, const Double_t& aQinvHi) :
MpdFemtoBaseCorrFctn(),
mManager(nullptr),
mNumeratorWeighted(nullptr),
mNumerator(nullptr),
mDenominatorWeighted(nullptr),
mDenominator(nullptr) {

    // Parametrized constructor
    char buf[100];
    snprintf(buf, 100, "ModelNumWeighted_%s", title);
    mNumeratorWeighted = new TH1D(buf, buf, aNbins, aQinvLo, aQinvHi);
    snprintf(buf, 100, "ModelNum_%s", title);
    mNumerator = new TH1D(buf, buf, aNbins, aQinvLo, aQinvHi);
    snprintf(buf, 100, "ModelDenWeighted_%s", title);
    mDenominatorWeighted = new TH1D(buf, buf, aNbins, aQinvLo, aQinvHi);
    snprintf(buf, 100, "ModelDen_%s", title);
    mDenominator = new TH1D(buf, buf, aNbins, aQinvLo, aQinvHi);

    mNumeratorWeighted->Sumw2();
    mNumerator->Sumw2();

    mDenominatorWeighted->Sumw2();
    mDenominator->Sumw2();
}

//_________________

MpdFemtoModelQinvCorrFctn::MpdFemtoModelQinvCorrFctn(const MpdFemtoModelQinvCorrFctn& corrFctn) :
MpdFemtoBaseCorrFctn(),
mManager(nullptr),
mNumeratorWeighted(nullptr),
mNumerator(nullptr),
mDenominatorWeighted(nullptr),
mDenominator(nullptr) {

    // Copy constructor
    if (corrFctn.mNumeratorWeighted) {
        mNumeratorWeighted = new TH1D(*(corrFctn.mNumeratorWeighted));
    }
    if (corrFctn.mNumerator) {
        mNumerator = new TH1D(*(corrFctn.mNumerator));
    }
    if (corrFctn.mDenominatorWeighted) {
        mDenominatorWeighted = new TH1D(*(corrFctn.mDenominatorWeighted));
    }
    if (corrFctn.mDenominator) {
        mDenominator = new TH1D(*(corrFctn.mDenominator));
    }

    mManager = corrFctn.mManager;
}

//_________________

MpdFemtoModelQinvCorrFctn::~MpdFemtoModelQinvCorrFctn() {
    // Destructor
    if (mNumeratorWeighted) {
        delete mNumeratorWeighted;
        mNumeratorWeighted = nullptr;
    }
    if (mNumerator) {
        delete mNumerator;
        mNumerator = nullptr;
    }
    if (mDenominatorWeighted) {
        delete mDenominatorWeighted;
        mDenominatorWeighted = nullptr;
    }
    if (mDenominator) {
        delete mDenominator;
        mDenominator = nullptr;
    }
}

//_________________

MpdFemtoModelQinvCorrFctn& MpdFemtoModelQinvCorrFctn::operator=(const MpdFemtoModelQinvCorrFctn& corrFctn) {

    // Assignment operator
    if (this == &corrFctn) {
        return *this;
    }

    if (mNumeratorWeighted) delete mNumeratorWeighted;
    mNumeratorWeighted = ((corrFctn.mNumeratorWeighted) ? new TH1D(*corrFctn.mNumeratorWeighted) : nullptr);

    if (mNumerator) delete mNumerator;
    mNumerator = ((corrFctn.mNumerator) ? new TH1D(*corrFctn.mNumerator) : nullptr);

    if (mDenominatorWeighted) delete mDenominatorWeighted;
    mDenominatorWeighted = ((corrFctn.mDenominatorWeighted) ? new TH1D(*corrFctn.mDenominatorWeighted) : nullptr);

    if (mDenominator) delete mDenominator;
    mDenominator = ((corrFctn.mDenominator) ? new TH1D(*corrFctn.mDenominator) : nullptr);

    mManager = corrFctn.mManager;

    return *this;
}

//_________________

void MpdFemtoModelQinvCorrFctn::connectToManager(MpdFemtoModelManager *manager) {
    mManager = manager;
}

//_________________

MpdFemtoString MpdFemtoModelQinvCorrFctn::report() {
    // Make a report
    MpdFemtoString tStr = "MpdFemtoModelQinvCorrFctn report";

    tStr += TString::Format("Number of entries in numerator (w/ weight)   :\t%E\n",
            mNumeratorWeighted->GetEntries());
    tStr += TString::Format("Number of entries in numerator (w/o weight)  :\t%E\n",
            mNumerator->GetEntries());
    tStr += TString::Format("Number of entries in denominator (w/ weight) :\t%E\n",
            mDenominatorWeighted->GetEntries());
    tStr += TString::Format("Number of entries in denominator (w/o weight):\t%E\n",
            mDenominator->GetEntries());

    if (mPairCut) {
        tStr += "Here is the PairCut specific to this CorrFctn\n";
        tStr += mPairCut->report();
    } else {
        tStr += "No PairCut specific to this CorrFctn\n";
    }

    return tStr;
}

//_________________

void MpdFemtoModelQinvCorrFctn::addRealPair(MpdFemtoPair* pair) {

    // Check if pair passes front-loaded cut if exists
    if (mPairCut && !mPairCut->pass(pair)) {
        return;
    }

    // Retrieve femtoscopic weight
    double weight = mManager->weight(pair);

    mNumeratorWeighted->Fill(fabs(pair->qInv()), weight);
    mNumerator->Fill(fabs(pair->qInv()), 1.);
}

//_________________

void MpdFemtoModelQinvCorrFctn::addMixedPair(MpdFemtoPair* pair) {

    // Check if pair passes front-loaded cut if exists
    if (mPairCut && !mPairCut->pass(pair)) {
        return;
    }

    double weight = mManager->weight(pair);
    mDenominatorWeighted->Fill(fabs(pair->qInv()), weight);
    mDenominator->Fill(fabs(pair->qInv()), 1.);
}

//_________________

void MpdFemtoModelQinvCorrFctn::eventBegin(const MpdFemtoEvent* /* event */) {
    /* empty */
}

//_________________

void MpdFemtoModelQinvCorrFctn::eventEnd(const MpdFemtoEvent* /* event */) {
    /* empty */
}

//_________________

void MpdFemtoModelQinvCorrFctn::finish() {
    /* empty */
}

//_________________

void MpdFemtoModelQinvCorrFctn::writeOutHistos() {

    // Write out data histos
    mNumeratorWeighted->Write();
    mNumerator->Write();
    mDenominatorWeighted->Write();
    mDenominator->Write();
}

//_________________

TList* MpdFemtoModelQinvCorrFctn::getOutputList() {

    // Prepare the list of objects to be written to the output
    TList *tOutputList = new TList();

    tOutputList->Add(mNumeratorWeighted);
    tOutputList->Add(mNumerator);
    tOutputList->Add(mDenominatorWeighted);
    tOutputList->Add(mDenominator);

    return tOutputList;
}
