//
// A pair cut that keeps most of the pair observables
//

// MpdFemtoMaker headers
#include "MpdFemtoPair.h"

// MpdFemtoMakerUser headers
#include "MpdFemtoBasicPairCut.h"

// ROOT headers
#include "TMath.h"

// C++ headers
#include <limits>

ClassImp(MpdFemtoBasicPairCut)

//_________________
MpdFemtoBasicPairCut::MpdFemtoBasicPairCut() : MpdFemtoBasePairCut() {

    // Default constructor
    mPrimaryVertex.SetXYZ(0., 0., 0.);
    mQuality[0] = -1e6;
    mQuality[1] = 1e6;
    mKt[0] = -1e6;
    mKt[1] = 1e6;
    mPt[0] = -1e6;
    mPt[1] = 1e6;
    mOpeningAngle[0] = -1e6;
    mOpeningAngle[1] = 1e6;
    mRapidity[0] = -1e6;
    mRapidity[1] = 1e6;
    mEta[0] = -1e6;
    mEta[1] = 1e6;
    mQinv[0] = -1e6;
    mQinv[1] = 1e6;
    mMinv[0] = -1e6;
    mMinv[1] = 1e6;
    mEntranceSeparation[0] = -1e6;
    mEntranceSeparation[1] = 1e6;
    mAngleToPrimaryVertex[0] = -1e6;
    mAngleToPrimaryVertex[1] = 1e6;
    mFracOfMergedRow[0] = -1e6;
    mFracOfMergedRow[1] = 1e6;
    mClosestRowAtDCA[0] = -1e6;
    mClosestRowAtDCA[1] = 1e6;
    mWeightedAvSep[0] = -1e6;
    mWeightedAvSep[1] = 1e6;
    mAverageSeparation[0] = -1e6;
    mAverageSeparation[1] = 1e6;
    mRValueLo = -1e6;
    mDPhiStarMin[0] = -1e6;
    mDPhiStarMin[1] = 1e6;
    mNPairsPassed = 0;
    mNPairsFailed = 0;
}

//_________________

MpdFemtoBasicPairCut::MpdFemtoBasicPairCut(const MpdFemtoBasicPairCut& c) : MpdFemtoBasePairCut(c) {

    // Copy constructor
    mNPairsPassed = 0;
    mNPairsFailed = 0;

    mPrimaryVertex = c.mPrimaryVertex;
    mQuality[0] = c.mQuality[0];
    mQuality[1] = c.mQuality[1];
    mKt[0] = c.mKt[0];
    mKt[1] = c.mKt[1];
    mPt[0] = c.mPt[0];
    mPt[1] = c.mPt[1];
    mOpeningAngle[0] = c.mOpeningAngle[0];
    mOpeningAngle[1] = c.mOpeningAngle[1];
    mQinv[0] = c.mQinv[0];
    mQinv[1] = c.mQinv[1];
    mMinv[0] = c.mMinv[0];
    mMinv[1] = c.mMinv[1];
    mRapidity[0] = c.mRapidity[0];
    mRapidity[1] = c.mRapidity[1];
    mEta[0] = c.mEta[0];
    mEta[1] = c.mEta[1];
    mEntranceSeparation[0] = c.mEntranceSeparation[0];
    mEntranceSeparation[1] = c.mEntranceSeparation[1];
    mAngleToPrimaryVertex[0] = c.mAngleToPrimaryVertex[0];
    mAngleToPrimaryVertex[1] = c.mAngleToPrimaryVertex[1];
    mFracOfMergedRow[0] = c.mFracOfMergedRow[0];
    mFracOfMergedRow[1] = c.mFracOfMergedRow[1];
    mClosestRowAtDCA[0] = c.mClosestRowAtDCA[0];
    mClosestRowAtDCA[1] = c.mClosestRowAtDCA[1];
    mWeightedAvSep[0] = c.mWeightedAvSep[0];
    mWeightedAvSep[1] = c.mWeightedAvSep[1];
    mAverageSeparation[0] = c.mAverageSeparation[0];
    mAverageSeparation[1] = c.mAverageSeparation[1];
    mRValueLo = c.mRValueLo;
    mDPhiStarMin[0] = c.mDPhiStarMin[0];
    mDPhiStarMin[1] = c.mDPhiStarMin[1];
}

//_________________

MpdFemtoBasicPairCut& MpdFemtoBasicPairCut::operator=(const MpdFemtoBasicPairCut& c) {
    // Assignment operator

    if (this != &c) {
        mNPairsPassed = 0;
        mNPairsFailed = 0;
        mPrimaryVertex = c.mPrimaryVertex;
        mQuality[0] = c.mQuality[0];
        mQuality[1] = c.mQuality[1];
        mKt[0] = c.mKt[0];
        mKt[1] = c.mKt[1];
        mPt[0] = c.mPt[0];
        mPt[1] = c.mPt[1];
        mOpeningAngle[0] = c.mOpeningAngle[0];
        mOpeningAngle[1] = c.mOpeningAngle[1];
        mQinv[0] = c.mQinv[0];
        mQinv[1] = c.mQinv[1];
        mMinv[0] = c.mMinv[0];
        mMinv[1] = c.mMinv[1];
        mRapidity[0] = c.mRapidity[0];
        mRapidity[1] = c.mRapidity[1];
        mEta[0] = c.mEta[0];
        mEta[1] = c.mEta[1];
        mEntranceSeparation[0] = c.mEntranceSeparation[0];
        mEntranceSeparation[1] = c.mEntranceSeparation[1];
        mAngleToPrimaryVertex[0] = c.mAngleToPrimaryVertex[0];
        mAngleToPrimaryVertex[1] = c.mAngleToPrimaryVertex[1];
        mFracOfMergedRow[0] = c.mFracOfMergedRow[0];
        mFracOfMergedRow[1] = c.mFracOfMergedRow[1];
        mClosestRowAtDCA[0] = c.mClosestRowAtDCA[0];
        mClosestRowAtDCA[1] = c.mClosestRowAtDCA[1];
        mWeightedAvSep[0] = c.mWeightedAvSep[0];
        mWeightedAvSep[1] = c.mWeightedAvSep[1];
        mAverageSeparation[0] = c.mAverageSeparation[0];
        mAverageSeparation[1] = c.mAverageSeparation[1];
        mRValueLo = c.mRValueLo;
        mDPhiStarMin[0] = c.mDPhiStarMin[0];
        mDPhiStarMin[1] = c.mDPhiStarMin[1];
    } // if (this != &c)

    return *this;
}

//_________________

MpdFemtoBasicPairCut::~MpdFemtoBasicPairCut() {
    // Destructor
    /* no-op */
}

//__________________

bool MpdFemtoBasicPairCut::pass(const MpdFemtoPair* pair) {
    // Check if pair passes pair-cut

    // For charged TRACKS only
    float dPhiStarMin = MpdFemtoPair::calculateDPhiStarMin(
            pair->track1()->momentum(),
            pair->track1()->track()->charge(),
            pair->track2()->momentum(),
            pair->track2()->track()->charge(),
            0.1, 0.6, 2., /* in meters */
            pair->track1()->track()->bField() * 0.1 /* in Tesla */
            );

    // // Uncomment next lines for debug
    // std::cout << "--MpdFemtoBasicPairCut--" << std::endl
    //     << "Quality         = " << pair->quality() << std::endl
    //     << "kT              = " << pair->kT() << std::endl
    //     << "pT              = " << pair->pT() << std::endl
    //     << "Opening angle   = " << pair->openingAngle() << std::endl
    //     << "Rapidity        = " << pair->rapidity()
    //     << std::endl
    //     << "Pseudorapity    = " << pair->eta()
    //     << std::endl
    //     << "Qinv            = " << pair->qInv() << std::endl
    //     << "Minv            = " << pair->mInv() << std::endl
    //     << "TpcEntranceSep  = " << pair->nominalTpcEntranceSeparation()
    //     << std::endl
    //     << "FracOfMergedRow = " << pair->fractionOfMergedRow() << std::endl
    //     << "ClosestRowAtDCA = " << pair->closestRowAtDCA() << std::endl
    //     << "WeightedAvSep   = " << pair->weightedAvSep() << std::endl
    //     << "AverageSep      = " << pair->nominalTpcAverageSeparation() << std::endl
    //     << "rValue          = " << pair->rValue() << std::endl
    //     << "dPhiStarMin     = " << dPhiStarMin
    //     << std::endl << std::endl;

    bool mGoodPair = (
            (pair->quality() >= mQuality[0]) &&
            (pair->quality() <= mQuality[1]) &&
            (pair->kT() >= mKt[0]) &&
            (pair->kT() <= mKt[1]) &&
            (pair->pT() >= mPt[0]) &&
            (pair->pT() <= mPt[1]) &&
            (pair->openingAngle() >= mOpeningAngle[0]) &&
            (pair->openingAngle() <= mOpeningAngle[1]) &&
            (pair->rapidity() >= mRapidity[0]) &&
            (pair->rapidity() <= mRapidity[1]) &&
            (pair->eta() >= mEta[0]) &&
            (pair->eta() <= mEta[1]) &&
            (TMath::Abs(pair->qInv()) >= mQinv[0]) &&
            (TMath::Abs(pair->qInv()) <= mQinv[1]) &&
            (TMath::Abs(pair->mInv()) >= mMinv[0]) &&
            (TMath::Abs(pair->mInv()) <= mMinv[1]) &&
            (pair->nominalTpcEntranceSeparation() >= mEntranceSeparation[0]) &&
            (pair->nominalTpcEntranceSeparation() <= mEntranceSeparation[1]) &&
            (pair->fractionOfMergedRow() >= mFracOfMergedRow[0]) &&
            (pair->fractionOfMergedRow() <= mFracOfMergedRow[1]) &&
            (pair->closestRowAtDCA() >= mClosestRowAtDCA[0]) &&
            (pair->closestRowAtDCA() <= mClosestRowAtDCA[1]) &&
            (pair->weightedAvSep() >= mWeightedAvSep[0]) &&
            (pair->weightedAvSep() <= mWeightedAvSep[1]) &&
            (pair->nominalTpcAverageSeparation() >= mAverageSeparation[0]) &&
            (pair->nominalTpcAverageSeparation() <= mAverageSeparation[1]) &&
            (pair->rValue() >= mRValueLo) &&
            (dPhiStarMin >= mDPhiStarMin[0]) &&
            (dPhiStarMin <= mDPhiStarMin[1])
            );


    // // Uncomment next lines for the dubug purpose
    // if(mGoodPair) {
    //   std::cout << "Pair cut passed!" << std::endl;
    // }
    // else {
    //   std::cout << "Pair cut failed!" << std::endl;
    // }

    mGoodPair ? mNPairsPassed++ : mNPairsFailed++;

    return mGoodPair;
}

//__________________

MpdFemtoString MpdFemtoBasicPairCut::report() {

    TString report;

    // Interesting option to print values
    // #define PRINTVAR(var) Form("%.4e <= " #var " <= %.4e\n", var[0], var[1])
    // report += PRINTVAR(mQuality);
    // #undef PRINTVAR

    report = "\n-- MpdFemtoBasicPairCut --\n";
    report += Form("%5.2f <= quality <= %5.2f\n", mQuality[0], mQuality[1]);
    report += Form("%5.2f <= kT <= %5.2f\n", mKt[0], mKt[1]);
    report += Form("%5.2f <= pT <= %5.2f\n", mPt[0], mPt[1]);
    report += Form("%5.2f <= opening angle <= %5.2f\n", mOpeningAngle[0], mOpeningAngle[1]);
    report += Form("%5.2f <= rapidity <= %5.2f\n", mRapidity[0], mRapidity[1]);
    report += Form("%5.2f <= eta <= %5.2f\n", mEta[0], mEta[1]);
    report += Form("%5.2f <= qInv <= %5.2f\n", mQinv[0], mQinv[1]);
    report += Form("%5.2f <= mInv <= %5.2f\n", mMinv[0], mMinv[1]);
    report += Form("%5.2f <= eta <= %5.2f\n", mEta[0], mEta[1]);
    report += Form("%5.2f <= entrance separation <= %5.2f\n", mEntranceSeparation[0], mEntranceSeparation[1]);
    report += Form("%5.2f <= angle to primary vertex <= %5.2f\n", mAngleToPrimaryVertex[0], mAngleToPrimaryVertex[1]);
    report += Form("%5.2f <= FMR <= %5.2f\n", mFracOfMergedRow[0], mFracOfMergedRow[1]);
    report += Form("%5.2f <= closest row at DCA <= %5.2f\n", mClosestRowAtDCA[0], mClosestRowAtDCA[1]);
    report += Form("%5.2f <= weighted average separation <= %5.2f\n", mWeightedAvSep[0], mWeightedAvSep[1]);
    report += Form("%5.2f <= average separation <= %5.2f\n", mAverageSeparation[0], mAverageSeparation[1]);
    report += Form("R >= %5.2f\n", mRValueLo);
    report += Form("%5.2f <= dphi* min <= %5.2f\n", mDPhiStarMin[0], mDPhiStarMin[1]);
    report += Form("NPairsPassed = %li\nNPairsFailed = %li\n", mNPairsPassed, mNPairsFailed);

    return MpdFemtoString((const char *) report);
}

//_________________

TList *MpdFemtoBasicPairCut::listSettings() {

    // Return a list of settings in a writable form
    TList *settings_list = new TList();

#define NEWOBJ(var) new TObjString( TString::Format( "MpdFemtoBasicPairCut." #var ".min=%f", var[0] ) ), \
                    new TObjString( TString::Format( "MpdFemtoBasicPairCut." #var ".max=%f", var[1] ) )

    settings_list->AddVector(NEWOBJ(mQuality),
            NEWOBJ(mKt),
            NEWOBJ(mPt),
            NEWOBJ(mOpeningAngle),
            NEWOBJ(mRapidity),
            NEWOBJ(mEta),
            NEWOBJ(mQinv),
            NEWOBJ(mMinv),
            NEWOBJ(mEntranceSeparation),
            NEWOBJ(mAverageSeparation),
            new TObjString(TString::Format("MpdFemtoBasicPairCut.mRValueLo=%f", mRValueLo)),
            NULL);

    return settings_list;
}
