//
// Reader for the McDst format that takes McDstReader
//

// MpdFemtoMaker headers
// Base
#include "MpdFemtoBaseEventCut.h"
#include "MpdFemtoBaseTrackCut.h"
#include "MpdFemtoBaseV0Cut.h"
#include "MpdFemtoBaseKinkCut.h"
#include "MpdFemtoBaseXiCut.h"
// Infrastructure
#include "MpdFemtoModelHiddenInfo.h"

// MpdFemtoMakerUser headers
#include "MpdFemtoMcDstReader.h"

// ROOT headers
#include "TMath.h"
#include "TMatrixDSymEigen.h"
#include "TVectorT.h"
#include "TMatrixTLazy.h"

//_________________

MpdFemtoMcDstReader::MpdFemtoMcDstReader() :
MpdFemtoBaseEventReader(),
mMcDstReader(nullptr),
mMagField(-5.),
mHbtEvent(nullptr),
mDoRotate(false),
mEventPlaneResolution(1.),
mRefMult(0),
mRefMultPos(0),
mRefMult2(0),
mRefMult2Pos(0),
mSphericity(-1.),
mSphericity2(-1.),
mEventsPassed(0) {
    // Default constructor
    mReaderStatus = 0;
    mRandom = new TRandom3(0);
    mPhi[0] = -100. * TMath::Pi();
    mPhi[1] = 100. * TMath::Pi();
}

//_________________

MpdFemtoMcDstReader::MpdFemtoMcDstReader(MpdMcDstReader *reader, int debug) :
MpdFemtoBaseEventReader(),
mMcDstReader(reader),
mMagField(-5.),
mHbtEvent(nullptr),
mDoRotate(false),
mEventPlaneResolution(1.),
mRefMult(0),
mRefMultPos(0),
mRefMult2(0),
mRefMult2Pos(0),
mSphericity(-1.),
mSphericity2(-1.),
mEventsPassed(0) {
    // Parametrized constructor
    mRandom = new TRandom3(0);
    // mDebug (defined in the base class)
    mDebug = debug;
    mPhi[0] = -100. * TMath::Pi();
    mPhi[1] = 100. * TMath::Pi();
}

//_________________

MpdFemtoMcDstReader::MpdFemtoMcDstReader(const MpdFemtoMcDstReader& copy) :
MpdFemtoBaseEventReader(),
mMcDstReader(copy.mMcDstReader),
mMagField(copy.mMagField),
mHbtEvent(nullptr),
mDoRotate(copy.mDoRotate),
mEventPlaneResolution(copy.mEventPlaneResolution),
mRefMult(0),
mRefMultPos(0),
mRefMult2(0),
mRefMult2Pos(0),
mSphericity(-1.),
mSphericity2(-1.),
mEventsPassed(0) {
    // Copy constructor
    if (mRandom) {
        delete mRandom;
        mRandom = new TRandom3(0);
    }
    mPhi[0] = copy.mPhi[0];
    mPhi[1] = copy.mPhi[1];
}

//_________________

MpdFemtoMcDstReader& MpdFemtoMcDstReader::operator=(const MpdFemtoMcDstReader& copy) {
    // Assignment operator
    if (this != &copy) {
        mMcDstReader = copy.mMcDstReader;
        mMagField = copy.mMagField;
        mHbtEvent = nullptr;
        mDoRotate = copy.mDoRotate;
        if (mRandom) {
            delete mRandom;
            mRandom = new TRandom3(0);
        }
        mPhi[0] = copy.mPhi[0];
        mPhi[1] = copy.mPhi[1];
        mEventPlaneResolution = copy.mEventPlaneResolution;
        mRefMult = 0;
        mRefMultPos = 0;
        mRefMult2 = 0;
        mRefMult2Pos = 0;
        mSphericity = -1.;
        mSphericity2 = -1.;
        mEventsPassed = 0;
    }

    return *this;
}

//_________________

MpdFemtoMcDstReader::~MpdFemtoMcDstReader() {
    // Destructor
    if (mMcDstReader) {
        delete mMcDstReader;
        mMcDstReader = nullptr;
    }
    if (mHbtEvent) {
        delete mHbtEvent;
        mHbtEvent = nullptr;
    }
    if (mRandom) {
        delete mRandom;
        mRandom = nullptr;
    }
}

//_________________

MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent() {
    // Read McDst and construct MpdFemtoEvent

    // Clean-up mHbtEvent from previous reading
    mHbtEvent = nullptr;

    // Check that McDstReader exists
    if (!mMcDstReader) {
        std::cout << "[ERROR] MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent() - no McDstReader is provided" << std::endl;
        mReaderStatus = 1;
        return mHbtEvent;
    }

    // Load event
    mMcDstReader->loadEntry(mEventsPassed);
    // Increment counter
    mEventsPassed++;

    // Check that McDst exists
    MpdMcDst *mcDst = mMcDstReader->mcDst();
    if (!mcDst) {
        std::cout << "[ERROR] MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent() - no McDst is in McDstReader" << std::endl;
        mReaderStatus = 1;
        return mHbtEvent;
    }

    // Check that McEvent exists
    if (!mcDst->event() || mcDst->numberOfParticles() <= 0) {
        if (!mcDst->event()) {
            std::cout << "[WARNING] MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent()"
                    << " - McEvent does not exist" << std::endl;
        }
        if (mcDst->numberOfParticles() <= 0) {
            std::cout << "[WARNING] MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent()"
                    << " - No particles in the event were found" << std::endl;
        }
        mReaderStatus = 1;
    } else {
        // Create empty HBT event and then fill it
        mHbtEvent = new MpdFemtoEvent();

        // Retrieve Monte Carlo event
        MpdMcEvent *mcEvent = (MpdMcEvent*) mcDst->event();

        mHbtEvent->setEventNumber(mcEvent->eventNr());
        mHbtEvent->setRunNumber(0);
        mHbtEvent->setMagneticField(mMagField);
        mHbtEvent->setNumberOfPrimaryTracks(0);
        mHbtEvent->setCent16(-1);

        double azimuthalAngle = 0.;
        double phi = mcEvent->phi();
        // Generate rotation angle by request
        if (mDoRotate) {
            phi = mRandom->Uniform(mPhi[0], mPhi[1]);
        }
        mHbtEvent->setEventPlaneAngle(phi);
        // Set ideal resolution
        mHbtEvent->setEventPlaneResolution(mEventPlaneResolution);

        // Function that calculates refMult, refMultPos, refMult2, refMult2Pos,
        // sphericity, sphericity2
        estimateEventProperties();

        // Track loop
        for (unsigned int iTrk = 0; iTrk < mcDst->numberOfParticles(); iTrk++) {

            // Retrieve i-th particle
            MpdMcParticle *particle = (MpdMcParticle*) mcDst->particle(iTrk);
            if (!particle) {
                std::cout << "[WARNING] MpdFemtoEvent* MpdFemtoMcDstReader::returnHbtEvent() - Particle does not exist"
                        << std::endl;
                continue;
            } // if(!trk)

            // Create new MpdFemtoTrack and fill it with required information
            MpdFemtoTrack *hbtTrack = new MpdFemtoTrack();
            hbtTrack->setId(particle->index());
            // Set number of hits (45 correspond to the number
            // of pad row in STAR TPC)
            // IMPORTANT!!!! nHits also stores charge as: nHits * charge
            // but charge in the TDataBasePDF is 3 for positive and -3 for negative
            if (particle->charge() > 0) {
                hbtTrack->setNHits(45);
            } else if (particle->charge() == 0) {
                hbtTrack->setNHits(0);
            } else {
                hbtTrack->setNHits(-45);
            }
            hbtTrack->setNHitsPossible(45);
            hbtTrack->setNHitsDedx(45);
            hbtTrack->setChi2(1.);
            hbtTrack->setDedx(dEdxMean(particle->mass(), particle->ptot()));
            if (TMath::Abs(particle->pdg()) == 211) {
                // Pion
                hbtTrack->setNSigmaElectron(-65.);
                hbtTrack->setNSigmaPion(0.);
                hbtTrack->setNSigmaKaon(65.);
                hbtTrack->setNSigmaProton(65.);

                hbtTrack->setPidProbElectron(0.);
                hbtTrack->setPidProbPion(1.);
                hbtTrack->setPidProbKaon(0.);
                hbtTrack->setPidProbProton(0.);
            } else if (TMath::Abs(particle->pdg()) == 321) {
                // Kaon
                hbtTrack->setNSigmaElectron(-65.);
                hbtTrack->setNSigmaPion(-65.);
                hbtTrack->setNSigmaKaon(0.);
                hbtTrack->setNSigmaProton(65.);

                hbtTrack->setPidProbElectron(0.);
                hbtTrack->setPidProbPion(0.);
                hbtTrack->setPidProbKaon(1.);
                hbtTrack->setPidProbProton(0.);
            } else if (TMath::Abs(particle->pdg()) == 2212) {
                // Proton
                hbtTrack->setNSigmaElectron(-65.);
                hbtTrack->setNSigmaPion(-65.);
                hbtTrack->setNSigmaKaon(-65.);
                hbtTrack->setNSigmaProton(0.);

                hbtTrack->setPidProbElectron(0.);
                hbtTrack->setPidProbPion(0.);
                hbtTrack->setPidProbKaon(0.);
                hbtTrack->setPidProbProton(1.);
            } else if (TMath::Abs(particle->pdg()) == 11) {
                // Electron
                hbtTrack->setNSigmaElectron(0.);
                hbtTrack->setNSigmaPion(65.);
                hbtTrack->setNSigmaKaon(65.);
                hbtTrack->setNSigmaProton(65.);

                hbtTrack->setPidProbElectron(1.);
                hbtTrack->setPidProbPion(0.);
                hbtTrack->setPidProbKaon(0.);
                hbtTrack->setPidProbProton(0.);
            } else {
                // Other particles
                hbtTrack->setNSigmaElectron(-65.);
                hbtTrack->setNSigmaPion(-65.);
                hbtTrack->setNSigmaKaon(-65.);
                hbtTrack->setNSigmaProton(-65.);

                hbtTrack->setPidProbElectron(0.);
                hbtTrack->setPidProbPion(0.);
                hbtTrack->setPidProbKaon(0.);
                hbtTrack->setPidProbProton(0.);
            }

            // McParticle stores emission point (last or decay point) in fm,
            // but DCA is stored in cm
            hbtTrack->setDca(particle->x() * 1e-13,
                    particle->y() * 1e-13,
                    particle->x() * 1e-13);

            if (mDoRotate) {
                hbtTrack->setPz(particle->pz());
                hbtTrack->setGlobalPz(particle->pz());
                azimuthalAngle = particle->phi();
                azimuthalAngle += phi;
                hbtTrack->setPx(particle->pt() * TMath::Cos(azimuthalAngle));
                hbtTrack->setPy(particle->pt() * TMath::Sin(azimuthalAngle));
                hbtTrack->setGlobalPx(particle->pt() * TMath::Cos(azimuthalAngle));
                hbtTrack->setGlobalPy(particle->pt() * TMath::Sin(azimuthalAngle));
            } else {
                hbtTrack->setP(particle->px(), particle->py(), particle->pz());
                hbtTrack->setGlobalP(particle->px(), particle->py(), particle->pz());
            }

            // McDst assumes that event happen at (0.,0.,0.)
            hbtTrack->setPrimaryVertex(0., 0., 0.);
            hbtTrack->setMagneticField(mMagField);
            // Set topology map (probably can insert probabilty function from data)
            hbtTrack->setTopologyMap(0, 0.);
            hbtTrack->setTopologyMap(1, 0.);

            hbtTrack->setBeta(particle->ptot() / particle->energy());

            // Create model hidden info and fill it
            MpdFemtoModelHiddenInfo *hiddenInfo = new MpdFemtoModelHiddenInfo();

            // px and py values are not a mistake and take into
            // account event plane angle rotation
            hiddenInfo->setTrueMomentum(hbtTrack->p().X(),
                    hbtTrack->p().Y(),
                    particle->pz());

            // Check if rotate event plane angle (IS IT NEEDED??)
            if (mDoRotate) {
                double radialPos = TMath::Sqrt(particle->x() * particle->x() +
                        particle->y() * particle->y());
                azimuthalAngle = TMath::ATan2(particle->y(), particle->x());
                azimuthalAngle += phi;
                hiddenInfo->setEmissionPoint(radialPos * TMath::Cos(azimuthalAngle),
                        radialPos * TMath::Sin(azimuthalAngle),
                        particle->z(),
                        particle->t());
            } else {
                hiddenInfo->setEmissionPoint(particle->x(),
                        particle->y(),
                        particle->z(),
                        particle->t());
            }
            hiddenInfo->setPdgPid(particle->pdg());
            hiddenInfo->setOrigin(0);
            hbtTrack->setHiddenInfo(hiddenInfo);

            // Check if a front-loaded cut exists. The mTrackCut
            // is inherited from MpdFemtoBaseReader
            if (mTrackCut && !mTrackCut->pass(hbtTrack)) {
                delete hbtTrack;
                continue;
            } // if ( mTrackCut )

            // Add track to track collection
            mHbtEvent->trackCollection()->push_back(hbtTrack);
        } // for(int iTrk=0; iTrk<mcDst->numberOfParticles(); iTrk++)

        mHbtEvent->setPrimaryVertex(0., 0., 0.);
        mHbtEvent->setRefMult(mRefMult);
        mHbtEvent->setRefMultPos(mRefMultPos);
        mHbtEvent->setRefMult2(mRefMult2);
        mHbtEvent->setRefMult2Pos(mRefMult2Pos);
        mHbtEvent->setGRefMult(mRefMult);
        mHbtEvent->setGRefMultPos(mRefMultPos);
        mHbtEvent->setBTofTrayMult(mRefMult);
        mHbtEvent->setNumberOfBTofMatched(mRefMult);
        mHbtEvent->setNumberOfBEMCMatched(mRefMult);
        mHbtEvent->setSphericity(mSphericity);
        mHbtEvent->setSphericity2(mSphericity2);

        // Check front-loaded cuts here, because now all event information
        // should be filled making event cut on it possible. The mEventCut
        // is inherited from MpdFemtoBaseReader
        if (mEventCut && !mEventCut->pass(mHbtEvent)) {
            delete mHbtEvent;
            mHbtEvent = nullptr;
            // return mHbtEvent; // Will be deleted 6 lines below
        } // if (mEventCut)

    } // else {

    return mHbtEvent;
}

//_________________

double MpdFemtoMcDstReader::dEdxMean(Double_t mass, Double_t momentum) {

    Double_t dedxMean = 0.;
    Double_t tpcDedxGain = 0.174325e-06;
    Double_t tpcDedxOffset = -2.71889;
    Double_t tpcDedxRise = 776.626;

    Double_t gamma = TMath::Sqrt(momentum * momentum / (mass * mass) + 1.);
    Double_t beta = TMath::Sqrt(1. - 1. / (gamma * gamma));
    Double_t rise = tpcDedxRise * beta * beta * gamma*gamma;
    if (beta > 0) {
        dedxMean = tpcDedxGain / (beta * beta) * (0.5 * TMath::Log(rise) -
                beta * beta - tpcDedxOffset);
    } else {
        dedxMean = 1000.;
    }
    return dedxMean;
}

//_________________

void MpdFemtoMcDstReader::estimateEventProperties() {

    // Initialize values
    mRefMult = 0;
    mRefMultPos = 0;
    mRefMult2 = 0;
    mRefMult2Pos = 0;
    mSphericity = -1.;
    mSphericity2 = -1.;

    // Check that reader exists
    if (!mMcDstReader) {
        std::cout << "[WARNING] void MpdFemtoMcDstReader::estimateEventProperties()"
                << " - No McDstReader is provided" << std::endl;
        return;
    }

    // Retrieve Monte Carlo DST
    MpdMcDst *mcDst = (MpdMcDst*) mMcDstReader->mcDst();
    if (!mcDst || mcDst->numberOfParticles() <= 0) {
        std::cout << "[WARNING] void MpdFemtoMcDstReader::estimateEventProperties()"
                << " - No McDst or particles in the event" << std::endl;
        return;
    }

    // Next matrices will be used for the sphericity calculation
    TMatrixTSym<double> *matrix = new TMatrixTSym<double>(2);
    TMatrixTSym<double> *matrix2 = new TMatrixTSym<double>(2);
    matrix->Zero();
    matrix2->Zero();

    float pTsum = 0.;
    float pTsum2 = 0.;
    float px = 0.;
    float py = 0.;
    float pt = 0.;

    // Loop over particles and calculate number of charged tracks
    // with |eta|<0.5 and pT>0.15 GeV/c (STAR way)
    for (unsigned int iTrk = 0; iTrk < mcDst->numberOfParticles(); iTrk++) {

        // Retrieve i-th particle
        MpdMcParticle *particle = (MpdMcParticle*) mcDst->particle(iTrk);
        // Particle must exist
        if (!particle) continue;

        // std::cout << "pT: " << particle->pt()
        //           << " charge: " << particle->charge()
        //           << " eta: " << particle->eta()
        //           << " parent: " << particle->parent() << std::endl;

        // In the experiment, one works only with charged particles
        if (particle->charge() == 0) continue;
        // Remove nucleons from incoming nuclei (should work for UrQMD)
        //if (particle->parent() == 0) continue;

        // Particle must have pT>0.15 GeV/c and |eta|<1 (STAR acceptance)
        if (TMath::Abs(particle->eta()) > 1. ||
                particle->pt() < 0.15) continue;


        mRefMult2++;
        if (particle->charge() > 0) {
            mRefMult2Pos++;
        }

        // Event properties in |eta|<1.
        px = particle->px();
        py = particle->py();
        pt = particle->pt();

        (*matrix2)(0, 0) += px * px / pt;
        (*matrix2)(1, 1) += py * py / pt;
        (*matrix2)(0, 1) += px * py / pt;
        (*matrix2)(1, 0) += px * py / pt;
        pTsum2 += pt;

        // Event properties in |eta|<0.5
        if (TMath::Abs(particle->eta()) <= 0.5) {
            mRefMult++;
            if (particle->charge() > 0) {
                mRefMultPos++;
            }

            (*matrix)(0, 0) += px * px / pt;
            (*matrix)(1, 1) += py * py / pt;
            (*matrix)(0, 1) += px * py / pt;
            (*matrix)(1, 0) += px * py / pt;
            pTsum += pt;
        } // if ( TMath::Abs( particle->eta() ) <= 0.5 )
    } // for ( int iTrk=0; iTrk<mMuDst->numberOfParticles(); iTrk++ )

    *matrix *= 1. / pTsum;
    *matrix2 *= 1. / pTsum2;
    TMatrixDSymEigen eigenEstimator(*matrix);
    TMatrixDSymEigen eigenEstimator2(*matrix2);
    TVectorD eigen = eigenEstimator.GetEigenValues();
    TVectorD eigen2 = eigenEstimator2.GetEigenValues();
    mSphericity = 2. * eigen.Min() / (eigen[0] + eigen[1]);
    mSphericity2 = 2. * eigen2.Min() / (eigen2[0] + eigen2[1]);

    // std::cout << "esitmate event properties\n"
    //           << "refMult: " << mRefMult << " refMultPos: " << mRefMultPos
    //           << "sph: " << mSphericity << std::endl;
}

//_________________

MpdFemtoString MpdFemtoMcDstReader::report() {
    // Make a report
    MpdFemtoString repstr = "\nMpdFemtoString MpdFemtoMcDstReader::report() - reporting\n";

    repstr += "---> Event cut in the reader: ";
    if (mEventCut) {
        repstr += mEventCut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n---> Track cut in the reader: ";
    if (mTrackCut) {
        repstr += mTrackCut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n---> V0 cut in the reader: ";
    if (mV0Cut) {
        repstr += mV0Cut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n---> V0 cut in the reader: ";
    if (mV0Cut) {
        repstr += mV0Cut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n---> Kink cut in the reader: ";
    if (mKinkCut) {
        repstr += mKinkCut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n---> Xi cut in the reader: ";
    if (mXiCut) {
        repstr += mXiCut->report();
    } else {
        repstr += "\tNONE";
    }

    repstr += "\n";
    return repstr;
}

ClassImp(MpdFemtoMcDstReader)

