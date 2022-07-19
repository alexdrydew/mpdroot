//-----------------------------------------------------------
//
// Description:
//      Implementation of class MpdTpcFastDigitizer
//      see MpdTpcFastDigitizer.h for details
//
// Author List:
//      Alexander Zinchenko LHEP, JINR, Dubna - 03-December-2021
//      (modified version of MpdTpcDigitizerAZlt.cxx)                           
//
//-----------------------------------------------------------

// This Class' Header ------------------
#include "MpdTpcFastDigitizer.h"

// MPD Headers ----------------------
#include "MpdTpcDigit.h"
#include "MpdMultiField.h"
#include "MpdTpcSector.h"
#include "MpdTpcSectorGeo.h"
#include "TpcGas.h"
#include "TpcPoint.h"
#include "fastdigimodel/ONNXRuntimeTpcFastDigiModelWrapper.h"

// FAIR Headers ----------------------
#include "FairRunAna.h"
#include "FairEventHeader.h"
#include "FairRootManager.h"
#include "FairRunSim.h"

// ROOT Headers ----------------------
#include "TClonesArray.h"
#include <TGeoManager.h>
#include "TLorentzVector.h"
#include "TRandom.h"
#include <TRefArray.h>
#include "TMath.h"
#include "TSystem.h"
#include "TaskHelpers.h"
#include <TVirtualFFT.h>
#include <TFile.h>
// C/C++ Headers ----------------------
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

// Class Member definitions -----------

using namespace std;
using namespace TMath;

static Int_t nOverlapDigit;
static Int_t nAllLightedDigits;

static clock_t tStart = 0;
static clock_t tFinish = 0;
static clock_t tAll = 0;

//FILE *lunAZ = nullptr; //fopen("gasGain.dat","w");
//---------------------------------------------------------------------------

MpdTpcFastDigitizer::MpdTpcFastDigitizer(ONNXRuntimeTpcFastDigiModelWrapper* onnxModelWrapper)
        : FairTask("TPC fast digitizer"),
          fMCPointArray(nullptr),
          fMCTracksArray(nullptr),
          fHitArray(nullptr),
          fDigits(nullptr),
          fDigits4dArray(nullptr),
          fSector(nullptr),
          fHisto(nullptr),
          fPRF(nullptr),
          fNoiseThreshold(3.0),
          fOverflow(1023.1),
          fNumOfPadsInRow(nullptr),
          fIsHistogramsInitialized(kFALSE),
          fMakeQA(kFALSE),
          fOnlyPrimary(kFALSE),
          fPersistence(kTRUE),
          fAttach(kFALSE),
          fDiffuse(kTRUE),
        //fDiffuse(kFALSE), // debug
          fDistort(kFALSE),
          fResponse(kTRUE),
          fDistribute(kTRUE),
        //fDistribute(kFALSE), // debug
          fPrintDebugInfo(kFALSE),
        //fOneRow(kTRUE), // debug
          fOneRow(kFALSE),
          modelWrapper(onnxModelWrapper) {
    fInputBranchName = "TpcPoint";
    fOutputBranchName = "MpdTpcDigit";

    string tpcGasFile = gSystem->Getenv("VMCWORKDIR");
    tpcGasFile += "/geometry/Ar-90_CH4-10.asc";
    fGas = new TpcGas(tpcGasFile, 130);
}

//---------------------------------------------------------------------------
MpdTpcFastDigitizer::~MpdTpcFastDigitizer() {
    if (fIsHistogramsInitialized) {
        delete fHisto;
    }
    delete fGas;
    delete fPRF;
    delete fSector;
}

//---------------------------------------------------------------------------
InitStatus MpdTpcFastDigitizer::Init() {

    modelWrapper->init();

    //Get ROOT Manager
    FairRootManager *ioman = FairRootManager::Instance();
    if (FairRunSim::Instance() == nullptr)
        fMagField = FairRunAna::Instance()->GetField();
    else fMagField = FairRunSim::Instance()->GetField();
    //fMagField = FairRunSim::Instance()->GetField();

    if (!ioman) {
        cout << "\n-E- [MpdTpcFastDigitizer::Init]: RootManager not instantiated!" << endl;
        return kFATAL;
    }
    fMCPointArray = (TClonesArray *) ioman->GetObject(fInputBranchName);
    fMCTracksArray = (TClonesArray *) ioman->GetObject("MCTrack");
    fHitArray = (TClonesArray *) ioman->GetObject("TpcHit");
    if (fHitArray == nullptr) {
        cout << "\n-E- [MpdTpcFastDigitizer::Init]: TpcHit branch is not found !" << endl;
        return kFATAL;
    }

    fSector = new TpcSector();
    /*
    nTimeBackets = fSector->GetNTimeBins();
    nSectors = fSector->GetNSectors();
    pwIn = fSector->GetInnerPadWidth();
    pwOut = fSector->GetOuterPadWidth();
    phIn = fSector->GetInnerPadHeight();
    phOut = fSector->GetOuterPadHeight();
    nRows = fSector->GetNumRows();
    nInRows = fSector->GetNumInnerRows();
    nOutRows = fSector->GetNumOuterRows();
    fSectInHeight = fSector->GetSectInnerHeight();
    fSectHeight = fSector->GetSectHeight();
    r_min = fSector->GetRmin();
    zCathode = fSector->GetLength(); //cm
    */
    fSecGeo = MpdTpcSectorGeo::Instance();
    fNTimeBins = fSecGeo->GetNTimeBins();
    nSectors = fSecGeo->NofSectors() * 2;
    pwIn = fSecGeo->PadWidth(0);
    pwOut = fSecGeo->PadWidth(1);
    phIn = fSecGeo->PadHeight(0);
    phOut = fSecGeo->PadHeight(1);
    nRows = fSecGeo->NofRows();
    nInRows = fSecGeo->NofRowsReg(0);
    nOutRows = fSecGeo->NofRowsReg(1);
    fSectInHeight = fSecGeo->GetRocY(1) - fSecGeo->GetRocY(0);
    fSectHeight = fSecGeo->GetRocY(2) - fSecGeo->GetRocY(0);
    r_min = fSecGeo->GetMinY();
    zCathode = fSecGeo->GetZmax(); //cm

    //fNumOfPadsInRow = fSector->GetArrayPadsInRow();
    fNumOfPadsInRow = fSecGeo->NPadsInRows();
    //if (fPrintDebugInfo) {
    if (1) {
        cout << "Number of pads in every rows is ";
        for (UInt_t k = 0; k < nRows; ++k)
            cout << fNumOfPadsInRow[k] * 2 << " ";
        cout << endl;
    }

    //memory allocating for output array
    fDigits4dArray = new DigOrigArray **[nRows];
    for (UInt_t iRow = 0; iRow < nRows; ++iRow) {
        fDigits4dArray[iRow] = new DigOrigArray *[fNumOfPadsInRow[iRow] * 2];
        for (UInt_t iPad = 0; iPad < (UInt_t) fNumOfPadsInRow[iRow] * 2; ++iPad) {
            fDigits4dArray[iRow][iPad] = new DigOrigArray[fNTimeBins];
            for (UInt_t iTime = 0; iTime < fNTimeBins; ++iTime) {
                fDigits4dArray[iRow][iPad][iTime].isOverlap = kFALSE;
                //AZ fDigits4dArray[iRow][iPad][iTime].origin = 0;
                fDigits4dArray[iRow][iPad][iTime].origin = -1;
                fDigits4dArray[iRow][iPad][iTime].origins.clear();
                fDigits4dArray[iRow][iPad][iTime].signal = 0.0;
            }
        }
    }

    fDigits = new TClonesArray(fOutputBranchName);
    ioman->Register(fOutputBranchName, "TPC", fDigits, fPersistence);

    //AZ fNoiseThreshold = 1000.0; // electrons
    //AZ-040720 fNoiseThreshold = 30.0; // ADC counts
    //(AZ-moved to constructor) fNoiseThreshold = 3.0; // ADC counts
    //AZ fGain = 5000.0; //electrons
    fGain = 1000.0; //electrons
    if (fResponse) {
        fSpread = 0.196; // cm  // Value is given by TPC group
        k1 = 1.0 / (Sqrt(TwoPi()) * fSpread);
        k2 = -0.5 / fSpread / fSpread;
    } else {
        fSpread = 0.0; // cm  // FOR TEST ONLY. NO RESPONSE.
        k1 = k2 = 1.0;
    }

    if (!fIsHistogramsInitialized && fMakeQA) {
        fHisto = new MpdTpcDigitizerQAHistograms();
        fHisto->Initialize();
        fIsHistogramsInitialized = kTRUE;
    }
    fPRF = padResponseFunction();
    nOverlapDigit = 0;
    nAllLightedDigits = 0;

    cout << "-I- MpdTpcFastDigitizer: Initialization successful." << endl;
    return kSUCCESS;
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::Exec(Option_t *opt) {

    tStart = clock();

    cout << "MpdTpcFastDigitizer::Exec started" << endl;
    fDigits->Delete();

    Int_t nPoints = fMCPointArray->GetEntriesFast();
    if (nPoints < 2) {
        Warning("MpdTpcFastDigitizer::Exec", "Not enough Hits in TPC for Digitization (<2)");
        return;
    }

    if (fPrintDebugInfo) cout << "Number of MC points is " << nPoints << endl << endl;

    const Float_t phiStep = TwoPi() / nSectors * 2;

    multimap <Double_t, Int_t> *pointsM = nullptr, *hitsM = nullptr;
    // Fast digitizer
    hitsM = new multimap<Double_t, Int_t>[nSectors];
    Int_t nHits = fHitArray->GetEntriesFast();

    for (Int_t ih = 0; ih < nHits; ++ih) {
        MpdTpcHit *hit = (MpdTpcHit *) fHitArray->UncheckedAt(ih);
        Int_t isec = fSecGeo->Sector(hit->GetDetectorID());
        hitsM[isec].insert(pair<Float_t, Int_t>(hit->GetLayer(), ih));
    }

    TpcPoint *virtPoint = new TpcPoint;
    TpcPoint tmpPoint;
    TpcPoint *ppp = &tmpPoint;

    for (UInt_t iSec = 0; iSec < nSectors; ++iSec) {
        // Fast digitizer
        multimap<Double_t, Int_t>::iterator mit = hitsM[iSec].begin();
        for (; mit != hitsM[iSec].end(); ++mit) {
            MpdTpcHit *hit = (MpdTpcHit *) fHitArray->UncheckedAt(mit->second);
            FastDigi(iSec, hit);
        }

        UInt_t maxTimeBin =
                (UInt_t) MpdTpcSectorGeo::Instance()->TimeMax() / MpdTpcSectorGeo::Instance()->TimeBin() + 1;
        for (UInt_t iRow = 0; iRow < nRows; ++iRow) {
            for (UInt_t iPad = 0; iPad < (UInt_t) fNumOfPadsInRow[iRow] * 2; ++iPad) {
                //AZ for (UInt_t iTime = 0; iTime < fNTimeBins; ++iTime) {
                for (UInt_t iTime = 0; iTime < maxTimeBin; ++iTime) {
                    if (fDigits4dArray[iRow][iPad][iTime].signal > fNoiseThreshold) {
                        Int_t outSize = fDigits->GetEntriesFast();
                        Int_t id = CalcOrigin(fDigits4dArray[iRow][iPad][iTime]);
                        if (id >= 0) {
                            Double_t ampl = fDigits4dArray[iRow][iPad][iTime].signal;

                            //if (ampl > 4095.1) ampl = 4095.1; // dynamic range 12 bits
                            // IR03 new scale and rounding
                            //Double_t ScaleFactor = 20./(550.*1.25); // 687.5
                            // В старой шкале пик от mip для 15mm пэда был 687 канале,
                            // тогда среднее (As=1.3MP) от рел-роста 1.01 в канале 550.*1.2\
5*1.3,                                                                          
                            // а в правильной шкале должен быть в 75 канале
                            Double_t ScaleFactor = 75. / (550. * 1.25 * 1.3); // 19-MAR-2020 Movc\
han                                                                             
                            // S.Movchan denies: if( iRow >= 26) ScaleFactor *=(1.2/1.8);
                            ampl *= ScaleFactor;
                            if (ampl > 1023.1) ampl = 1023.1;

                            //new((*fDigits)[outSize]) MpdTpcDigit(id, iPad, iRow, iTime, iSec, fDigits4dArray[iRow][iPad][iTime].signal);
                            new((*fDigits)[outSize]) MpdTpcDigit(id, iPad, iRow, iTime, iSec, ampl);
                        }
                    }
                    //if (fDigits4dArray[iRow][iPad][iTime].signal > 0.0) {
                    if (CalcOrigin(fDigits4dArray[iRow][iPad][iTime]) >= 0) {
                        fDigits4dArray[iRow][iPad][iTime].origins.clear();
                        fDigits4dArray[iRow][iPad][iTime].origin = -1;
                        fDigits4dArray[iRow][iPad][iTime].signal = 0.0;
                        fDigits4dArray[iRow][iPad][iTime].isOverlap = kFALSE;
                    }
                }
            }
        }
    } // for (UInt_t iSec = 0; iSec < nSectors;

    tFinish = clock();
    tAll = tAll + (tFinish - tStart);
    delete[] pointsM;
    delete[] hitsM;
    delete virtPoint;
    cout << "MpdTpcFastDigitizer::Exec finished" << endl;
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::Check4Edge(UInt_t iSec, TpcPoint *&prePoint, TpcPoint *virtPoint) {
    // Check for edge-effect for track entering TPC from inside
    // (and correct for it if necessary)

    TVector3 posG, posL;
    prePoint->Position(posG);
    Int_t row0 = MpdTpcSectorGeo::Instance()->Global2Local(posG, posL, iSec % (nSectors / 2));
    row0 = MpdTpcSectorGeo::Instance()->PadRow(row0);
    //cout << " Row: " << row0 << " " << iSec << " " << posL[1] << endl;
    if (row0) return;

    // For padrow == 0:  create virtual point to correct for edge effect
    TVector3 mom, posL1;
    prePoint->Momentum(mom);
    if (mom.Pt() < 0.02) return; // do not adjust for very low-Pt tracks
    if (posL[1] < 0.01) return; // do not adjust - almost at the entrance

    posG += mom;
    MpdTpcSectorGeo::Instance()->Global2Local(posG, posL1, iSec % (nSectors / 2));
    mom = posL1;
    mom -= posL; // momentum in sector frame
    if (mom[1] < 0.02) return; // do not adjust - going inward or parallel to sector lower edge

    Double_t scale = mom[1] / posL[1];
    mom.SetMag(mom.Mag() / scale);
    posL -= mom;
    //cout << posL[0] << " " << posL[1] << " " << posL[2] << endl;
    MpdTpcSectorGeo::Instance()->Local2Global(iSec % (nSectors / 2), posL, posG);
    virtPoint->SetPosition(posG);
    virtPoint->SetTrackID(prePoint->GetTrackID());
    //AZ prePoint->SetEnergyLoss(prePoint->GetEnergyLoss()*1.3); // 29.10.16 - correct for edge-effect
    //AZ-081121 virtPoint->SetEnergyLoss(prePoint->GetEnergyLoss()*1.3); //AZ-090620 - correct for entrance effect
    virtPoint->SetEnergyLoss(prePoint->GetEnergyLoss() * 0.77); //AZ-081121
    prePoint = virtPoint;
}

//---------------------------------------------------------------------------

//AZ Int_t MpdTpcDigitizerAZ::CalcOrigin(const DigOrigArray dig) {
Int_t MpdTpcFastDigitizer::CalcOrigin(DigOrigArray &dig) {

    if (dig.origin >= 0) return dig.origin; // already done before
    if (dig.origins.size() == 0) return -1;

    Float_t max = 0.0;
    Int_t maxOrig = -1;
    if (dig.origins.size() > 1) {
        for (map<Int_t, Float_t>::const_iterator it = dig.origins.begin(); it != dig.origins.end(); ++it) {
            if (it->second > max) {
                maxOrig = it->first;
                max = it->second;
            }
        }
    } else {
        maxOrig = dig.origins.begin()->first;
    }
    dig.origin = maxOrig; //AZ
    return maxOrig;
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::PadResponse(Float_t x, Float_t y, UInt_t timeID, Int_t origin, DigOrigArray ***arr) {

    vector <UInt_t> lightedPads;
    vector <UInt_t> lightedRows;
    vector <Float_t> amps;

    //Float_t avAmp = 0.0;
    Float_t amplSum = 0.0;
    Float_t amplitude = 0.0;

    GetArea(x, y, fSpread * 3, lightedPads, lightedRows);
    Double_t gain = fGain * Polya();
    //fprintf(lunAZ,"%f\n",gain/fGain);
    Int_t nPads = lightedPads.size();

    for (Int_t i = 0; i < nPads; ++i) {
        //AZ amplitude = CalculatePadResponse(lightedPads.at(i), lightedRows.at(i), x, y);
        amplitude = gain * CalculatePadResponse(i, nPads, lightedPads.at(i), lightedRows.at(i), x, y);
        amps.push_back(amplitude);
        amplSum += amplitude;
    }

    /*AZ
    if (amplSum > 0.0) {
      map<Int_t, Float_t>::iterator it;
      avAmp = fGain / amplSum; // Normalize amplitudes
      for (UInt_t i = 0; i < amps.size(); ++i) {
        arr[lightedRows.at(i)][lightedPads.at(i)][timeID].signal += (amps.at(i) * avAmp);
        it = arr[lightedRows.at(i)][lightedPads.at(i)][timeID].origins.find(origin);
        if (it != arr[lightedRows.at(i)][lightedPads.at(i)][timeID].origins.end()) {
      it->second += (amps.at(i) * avAmp);
        } else {
      arr[lightedRows.at(i)][lightedPads.at(i)][timeID].origins.insert(pair<Int_t, Float_t>(origin, amps.at(i) * avAmp));
        }
      }
    }
    */
    if (amplSum > 0.0) {
        map<Int_t, Float_t>::iterator it;
        for (UInt_t i = 0; i < amps.size(); ++i) {
            arr[lightedRows.at(i)][lightedPads.at(i)][timeID].signal += amps.at(i);
            it = arr[lightedRows.at(i)][lightedPads.at(i)][timeID].origins.find(origin);
            if (it != arr[lightedRows.at(i)][lightedPads.at(i)][timeID].origins.end()) {
                it->second += amps.at(i);
            } else {
                arr[lightedRows[i]][lightedPads[i]][timeID].origins.insert(pair<Int_t, Float_t>(origin, amps[i]));
            }
        }
    }
}

//---------------------------------------------------------------------------

Double_t MpdTpcFastDigitizer::Polya() {
    // Gas gain according to Polya-distribution with parameter \theta = 1.5

    static Int_t first = 0;
    static TH1D *hPolya;

    //long rseed;
    Double_t step = 0.01, shift = 0.005, param = 1.5, prob, lambda;

    if (first == 0) {
        hPolya = new TH1D("hPolya", "Polya distribution", 1000, 0, 10);
        Double_t param1 = 1 + param;
        Double_t coef = TMath::Exp(TMath::Log(param1) * param1) / TMath::Gamma(param1);
        for (Int_t i = 0; i < 1000; ++i) {
            lambda = i * step + shift;
            //prob = param / 0.8862 * TMath::Sqrt(param*lambda)*TMath::Exp(-param*lambda); // 0.8862 = Gamma(1.5)
            prob = coef * TMath::Exp(TMath::Log(lambda) * param) * TMath::Exp(-param1 * lambda);
            hPolya->Fill(lambda, prob);
        }
        first = 1;
    }

    return hPolya->GetRandom();
    //return 1; // fixed gain - for debug
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::SignalShaping() {
    // Apply electronics response function

    static Int_t first = 0, nbins = 0, icent = 0, n2 = 0;
    static Double_t *reFilt = nullptr, *imFilt = nullptr;
    static TVirtualFFT *fft[2] = {nullptr, nullptr};
    const Double_t sigma = 190. / 2 / TMath::Sqrt(2 * TMath::Log(2)), sigma2 = sigma * sigma; // FWHM = 190 ns
    const Int_t maxTimeBin = MpdTpcSectorGeo::Instance()->TimeMax() / MpdTpcSectorGeo::Instance()->TimeBin() + 1;

    if (first == 0) {
        first = 1;
        nbins = MpdTpcSectorGeo::Instance()->GetNTimeBins();
        if (nbins % 2 == 0) --nbins;
        n2 = nbins / 2 + 1;
        icent = nbins / 2;
        reFilt = new Double_t[nbins];
        imFilt = new Double_t[nbins];
        for (Int_t i = 0; i < nbins; ++i) {
            Double_t t = (i - icent) * MpdTpcSectorGeo::Instance()->TimeBin();
            Double_t ampl = TMath::Exp(-t * t / 2 / sigma2);
            if (TMath::Abs(t) > 5 * sigma) ampl = 0;
            reFilt[i] = ampl;
        }
        fft[0] = TVirtualFFT::FFT(1, &nbins, "R2C ES K");
        //fft[0] = TVirtualFFT::FFT(1, &nbins, "R2C EX K");
        fft[0]->SetPoints(reFilt);
        fft[0]->Transform();
        fft[0]->GetPointsComplex(reFilt, imFilt);
    }

    Double_t *reSig = new Double_t[nbins];
    Double_t *imSig = new Double_t[nbins];
    Double_t *reTot = new Double_t[nbins];
    Double_t *imTot = new Double_t[nbins];
    map <Int_t, Int_t> cumul; // cumulative active time bin counter
    const Double_t ScaleFactor = 0.083916084; //See ampl for details
    //AZ Int_t nRows = MpdTpcSectorGeo::Instance()->NofRows();
    for (UInt_t iRow = 0; iRow < nRows; ++iRow) {
        for (UInt_t iPad = 0; iPad < (UInt_t) fNumOfPadsInRow[iRow] * 2; ++iPad) {
            memset(reSig, 0, sizeof(Double_t) * nbins);
            UInt_t fired = 0;
            UInt_t ntbins = 0;

            //for (Int_t iTime = 0; iTime < nbins; ++iTime) {
            for (Int_t iTime = 0; iTime < maxTimeBin; ++iTime) {
                //if (fDigits4dArray[iRow][iPad][iTime].signal > 0) {
                if (CalcOrigin(fDigits4dArray[iRow][iPad][iTime]) >= 0) {
                    // Fired channel
                    fired = 1;
                    if (ntbins == 0) cumul.clear();
                    reSig[iTime] = fDigits4dArray[iRow][iPad][iTime].signal;
                    cumul[iTime] = ++ntbins;
                }
            }
            if (!fired) continue;

            // !!! Formally expand each time bin backward by 3 bins (and forward if the next time bin is far enough) !!!
            for (map<Int_t, Int_t>::iterator mit = cumul.begin(); mit != cumul.end(); ++mit) {
                Int_t tbin = mit->first;
                Int_t orig = fDigits4dArray[iRow][iPad][tbin].origin;

                for (Int_t it = 1; it < 4; ++it) {
                    Int_t iTime = tbin - it;
                    if (iTime < 0) break;
                    if (CalcOrigin(fDigits4dArray[iRow][iPad][iTime]) >= 0) break;
                    fDigits4dArray[iRow][iPad][iTime].origin = orig;
                    fDigits4dArray[iRow][iPad][iTime].origins[orig] = 1.0; // 1.0 - some amplitude
                }
                // Expand forward if needed
                map<Int_t, Int_t>::iterator next = cumul.upper_bound(tbin);
                if (next == cumul.end() || next->first - tbin > 4) {
                    Int_t nexp = (next == cumul.end()) ? 4 : next->first - tbin - 3;
                    nexp = TMath::Min(nexp, 4);
                    // Expand
                    for (Int_t it = 1; it < nexp; ++it) {
                        Int_t iTime = tbin + it;
                        //if (iTime >= nbins) break;
                        if (iTime >= maxTimeBin) break;
                        //if (CalcOrigin(fDigits4dArray[iRow][iPad][iTime]) >= 0) break;
                        fDigits4dArray[iRow][iPad][iTime].origin = orig;
                        fDigits4dArray[iRow][iPad][iTime].origins[orig] = 1.0; // 1.0 - some amplitude
                    }
                }
            }

            // Fourier transform
            fft[0]->SetPoints(reSig);
            fft[0]->Transform();
            fft[0]->GetPointsComplex(reSig, imSig);
            // Convolution
            //for (Int_t i = 0; i < nbins; ++i) {
            for (Int_t i = 0; i < n2; ++i) {
                Double_t re = reSig[i] * reFilt[i] - imSig[i] * imFilt[i];
                Double_t im = reSig[i] * imFilt[i] + imSig[i] * reFilt[i];
                reTot[i] = re / nbins;
                imTot[i] = im / nbins;
            }
            // Inverse Fourier transform
            if (!fft[1]) fft[1] = TVirtualFFT::FFT(1, &nbins, "C2R ES K");
            //if (!fft[1]) fft[1] = TVirtualFFT::FFT(1, &nbins, "C2R EX K");
            fft[1]->SetPointsComplex(reTot, imTot);
            fft[1]->Transform();
            fft[1]->GetPoints(reTot);

            //AZ for (Int_t i = 0; i < nbins; ++i) {
            for (Int_t i = 0; i < maxTimeBin; ++i) {
                if (fDigits4dArray[iRow][iPad][i].origin < 0)
                    continue; // !!! do not add extra time bins due to shaping !!!
                Int_t i1 = i;
                if (i1 <= icent) i1 += icent;
                else i1 -= (icent + 1);
                Double_t ampl = reTot[i1];
                //
                // Scale factor to adjust ADC counts
                ampl /= 30.0; //include division in scalefactor?
                //IR 11-MAR-2020 if (ampl > 4095.1) ampl = 4095.1; // dynamic range 12 bits
                //
                // IR03 new scale and rounding
                //Double_t ScaleFactor = 20./(550.*1.25); // 687.5
                // В старой шкале пик от mip для 15mm пэда был 687 канале,
                // тогда среднее (As=1.3MP) от рел-роста 1.01 в канале 550.*1.25*1.3,
                // а в правильной шкале должен быть в 75 канале
                // const Double_t ScaleFactor = 0.083916084; //Move above loop == 75./(550.*1.25*1.3); // 19-MAR-2020 Movchan
                // S.Movchan denies: if( iRow >= 26) ScaleFactor *=(1.2/1.8);
                ampl *= ScaleFactor;
                //AZ if( ampl > 1023.1) ampl = 1023.1;
                if (ampl > fOverflow) ampl = fOverflow;
                //
                fDigits4dArray[iRow][iPad][i].signal = ampl;
            }
        }
    }
    delete[] reSig;
    delete[] imSig;
    delete[] reTot;
    delete[] imTot;
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::GetArea(Float_t xEll, Float_t yEll, Float_t radius, vector <UInt_t> &padIDs,
                                  vector <UInt_t> &rowIDs) {

    //Float_t padW = 0.0, padH = 0.0;
    //Float_t y = 0.0, x = 0.0;
    UInt_t pad = 0, row = 0;
    Float_t yNext; //delta = 0,
    //if (!fResponse) delta = -1000.0; //for test only!!!
    fSecGeo->PadID(xEll, yEll, row, pad, yNext);
    for (Int_t ip = -1; ip < 2; ++ip) {
        Int_t pad1 = pad + ip;
        if (pad1 < 0) continue;
        if (pad1 >= MpdTpcSectorGeo::Instance()->NPadsInRows()[row] * 2) break;
        padIDs.push_back(pad1);
        rowIDs.push_back(row);
    }
    if (fOneRow) return; // for debug - charge only in one padrow
    // Add extra row
    if (TMath::Abs(yNext) < radius) {
        Int_t row1 = row;
        if (yNext < 0) {
            --row1;
            if (row1 < 0) return;
            if (fSecGeo->NPadsInRows()[row1] != fSecGeo->NPadsInRows()[row]) --pad; // different number of pads in rows
        } else if (yNext > 0) {
            ++row1;
            if ((UInt_t) row1 > nRows - 1) return;
            if (fSecGeo->NPadsInRows()[row1] != fSecGeo->NPadsInRows()[row]) ++pad; // different number of pads in rows
        }
        for (Int_t ip = -1; ip < 2; ++ip) {
            Int_t pad1 = pad + ip;
            if (pad1 < 0) continue;
            if (pad1 >= MpdTpcSectorGeo::Instance()->NPadsInRows()[row1] * 2) break;
            padIDs.push_back(pad1);
            rowIDs.push_back(row1);
        }
    }
}

//---------------------------------------------------------------------------

Float_t
MpdTpcFastDigitizer::CalculatePadResponse(Int_t iloop, Int_t nLoop, UInt_t padID, UInt_t rowID, Float_t x, Float_t y) {
    // Calculate pad response using lookup table (assumes the same pad width in both readout plane regions)

    const Int_t npads = 5, nposX = 100, nposX2 = nposX * 2; // 5 pads, 100 positions along padrow (step 25 um)
    const Int_t nposY = 100, nposY2 = nposY * 2;
    static Int_t first = 1, padID0, istep0, idist0;
    static UInt_t rowID0;
    static Double_t chargeX[npads][nposX], stepX = pwIn / nposX2;
    static Double_t chargeY[2][nposY]; // charges on padrow (inner and outer regions)
    static Double_t cy;
    Double_t padW = pwIn, padH = 0, distx, disty, maxX, minX, cx1, cx2, stepY, minY, maxY;

    Double_t sigma = fSpread;

    if (first) {
        // Compute lookup tables
        first = 0;

        // Along X
        for (Int_t i = 0; i < nposX; ++i) {
            distx = i * stepX; // distance to pad center
            maxX = padW / 2 - distx;

            for (Int_t j = 0; j < npads; ++j) {
                if (j == 0) maxX -= 2 * padW;
                minX = maxX - padW;
                if (j == 0) cx2 = TMath::Erf(minX / sigma);
                else cx2 = cx1;
                cx1 = TMath::Erf(maxX / sigma);
                chargeX[j][i] = TMath::Abs(cx1 - cx2) / 2.;
                maxX += padW;
            }
        }

        // Along Y
        // 2 pad heights
        for (Int_t i = 0; i < 2; ++i) {
            padH = (i == 0) ? phIn : phOut;
            stepY = padH / nposY / 2;

            for (Int_t j = 0; j < nposY; ++j) {
                disty = j * stepY; // distance to pad center
                maxY = padH / 2 - disty;
                minY = maxY - padH;
                chargeY[i][j] = TMath::Abs(TMath::Erf(maxY / sigma) - TMath::Erf(minY / sigma)) / 2.;
            }
        }

    }
    //exit(0);

    Int_t izone = 0, idist;

    // Different padrow
    if (iloop == 0 || rowID != rowID0) {
        Double_t padY;
        rowID0 = rowID;
        if (rowID < nInRows) {
            //padW = pwIn;
            padH = phIn;
            padY = padH * ((Double_t) rowID + 0.5); // y-coordinate of pad center
        } else {
            //padW = pwOut;
            padH = phOut;
            padY = fSectInHeight + (((Double_t) rowID - nInRows) + 0.5) * padH; // y-coordinate of pad center
            izone = 1;
        }
        disty = y - padY;
        Int_t istep = TMath::Abs(disty / (padH / nposY2));
        if (istep < nposY) cy = chargeY[izone][istep];
        else {
            istep = nposY2 - istep;
            cy = 1.0 - chargeY[izone][istep];
        }

        Double_t padX = padW * ((Double_t) padID - fNumOfPadsInRow[rowID] + 0.5); // x-coordinate of pad center
        padID0 = padID;
        distx = x - padX;
        idist = TMath::Nint(distx / stepX);
        idist0 = idist;
        istep0 = TMath::Abs(idist);
        istep0 %= nposX2;
        if (istep0 > nposX) istep0 = TMath::Abs(istep0 - nposX2);
        istep0 = TMath::Min(istep0, nposX - 1);
    } else idist = idist0 - (padID - padID0) * nposX2;

    Int_t ipad = TMath::Abs(idist / nposX);
    if (ipad == 0) ipad = npads / 2; // central pad
        //else if (idist > 0) {
    else if (1) {
        if (ipad % 2 == 0) ipad = npads / 2 - ipad / 2;
        else ipad = npads / 2 + (ipad + 1) / 2;
    } else {
        if (ipad % 2 != 0) ipad = npads / 2 + (ipad + 1) / 2;
        else ipad = npads / 2 - ipad / 2;
    }
    cx1 = chargeX[ipad][istep0];

    Double_t ctot = TMath::Abs(cy) * cx1;
    //if (rowID == 1) cout << " Row, pad: " << rowID << " " << padID << " " << x << " " << istep0 << " " << idist << " " << ipad << " " << cx1 << " " << ctot << endl;
    return ctot;
}

//---------------------------------------------------------------------------
TF1 *MpdTpcFastDigitizer::padResponseFunction() {
    if (fPRF)
        return fPRF;

    fPRF = new TF1("Gaus PRF", "gaus", -5, 5);
    fPRF->SetParameter(0, 1.0 / (sqrt(2.0 * TMath::Pi()) * fSpread));
    fPRF->SetParameter(1, 0);
    fPRF->SetParameter(2, fSpread);

    return fPRF;
}

//---------------------------------------------------------------------------
Bool_t MpdTpcFastDigitizer::isSubtrackInInwards(const TpcPoint *p1, const TpcPoint *p2) { //WHAT AM I DOING???
    const Float_t x1 = p1->GetX();
    const Float_t x2 = p2->GetX();
    const Float_t y1 = p1->GetY();
    const Float_t y2 = p2->GetY();
    const Float_t a = (y1 - y2) / (x1 - x2);
    const Float_t b = (y1 * x2 - x1 * y2) / (x2 - x1);
    const Float_t minR = fabs(b) / sqrt(a * a + 1);

    if (minR < r_min) //then check if minimal distance is between our points
    {
        const Float_t x = -a * b / (a * a + 1);
        const Float_t y = b / (a * a + 1);
        if ((x1 - x) * (x2 - x) < 0 && (y1 - y) * (y2 - y) < 0) {
            return kTRUE;
        }
    }
    return kFALSE;
}

//---------------------------------------------------------------------------

void MpdTpcFastDigitizer::TpcProcessing(const TpcPoint *prePoint, const TpcPoint *curPoint,
                                        const UInt_t secID, const UInt_t iPoint, const UInt_t nPoints) {

    Float_t dE = 0.0; //energy loss
    UInt_t qTotal = 0; //sum of clusters charges (=sum of electrons between two TpcPoints)
    UInt_t qCluster = 0; //charge of cluster (= number of electrons)
    TLorentzVector curPointPos; // coordinates for current TpcPoint
    TLorentzVector prePointPos; // coordinates for previous TpcPoint
    TLorentzVector diffPointPos; // steps for clusters creation
    TVector3 diffuse; // vector of diffuse for every coordinates
    TVector3 distort; // vector of distortion for every coordinates
    TLorentzVector electronPos; // coordinates for created electrons
    TLorentzVector clustPos; // coordinates for created clusters
    Float_t driftl = 0.0; // length for drifting
    vector <UInt_t> clustArr; // vector of clusters between two TpcPoints
    Float_t localX = 0.0, localY = 0.0; //local coordinates of electron (sector coordinates)
    MpdTpcSectorGeo *secGeo = MpdTpcSectorGeo::Instance();

    if (fPrintDebugInfo && (iPoint % 1000 == 0))
        cout << UInt_t(iPoint * 1.0 / nPoints * 100.0) << " % of TPC points processed" << endl;
    //        curPoint = (TpcPoint*) fMCPointArray->At(i);

    if (fOnlyPrimary == kTRUE) {
        MpdMCTrack *tr = (MpdMCTrack *) fMCTracksArray->At(curPoint->GetTrackID());
        if (tr->GetMotherId() != -1) return;
    }
    //check if hits are on the same track
    if (curPoint->GetTrackID() == prePoint->GetTrackID() && !isSubtrackInInwards(prePoint, curPoint)) {

        dE = curPoint->GetEnergyLoss() * 1E9; //convert from GeV to eV
        if (dE < 0) {
            Error("MpdTpcDigitizerTask::Exec", "Negative Energy loss!");
            return;
        }

        qTotal = (UInt_t) floor(fabs(dE / fGas->W()));
        if (qTotal == 0) return;

        curPointPos.SetXYZT(curPoint->GetX(), curPoint->GetY(), curPoint->GetZ(), curPoint->GetTime());
        prePointPos.SetXYZT(prePoint->GetX(), prePoint->GetY(), prePoint->GetZ(), prePoint->GetTime());
        if ((curPointPos.T() < 0) || (prePointPos.T() < 0)) {
            Error("MpdTpcDigitizerTask::Exec", "Negative Time!");
            return;
        }

        diffPointPos = curPointPos - prePointPos; //differences between two points by coordinates
        //AZ diffPointPos *= (1 / diffPointPos.Vect().Mag()); //directional cosines //TODO! Do we need this??? Look at line #297

        //while still charge not used-up distribute charge into next cluster

        if (fDistribute) {
            while (qTotal > 0) {
                //roll dice for next cluster
                qCluster = fGas->GetRandomCSUniform();
                if (qCluster > qTotal) qCluster = qTotal;
                qTotal -= qCluster;
                clustArr.push_back(qCluster);
            }// finish loop for cluster creation
        } else {
            clustArr.push_back(qTotal); // JUST FOR TEST. NO CLUSTER DISTRIBUTION!
            //             clustArr.push_back(1); // JUST FOR TEST. NO CLUSTER DISTRIBUTION ONLY ONE ELECTRON IN CLUSTER!
        }

        //AZ diffPointPos *= (diffPointPos.Vect().Mag() / clustArr.size()); //now here are steps between clusters by coordinates TODO: correct distribution
        diffPointPos *= (1. / clustArr.size());
        clustPos = prePointPos;

        for (UInt_t iClust = 0; iClust < clustArr.size(); ++iClust) {
            clustPos += diffPointPos;
            driftl = zCathode - fabs(clustPos.Z());
            if (driftl <= 0) continue;

            for (UInt_t iEll = 0; iEll < clustArr.at(iClust); ++iEll) {

                //attachment
                if (fAttach)
                    if (exp(-driftl * fGas->k()) < gRandom->Uniform()) continue; // FIXME

                //diffusion
                if (fDiffuse) {
                    const Float_t sqrtDrift = sqrt(driftl);
                    const Float_t sigmat = fGas->Dt() * sqrtDrift;
                    const Float_t sigmal = fGas->Dl() * sqrtDrift;
                    diffuse.SetXYZ(gRandom->Gaus(0, sigmat), gRandom->Gaus(0, sigmat), gRandom->Gaus(0, sigmal));
                }

                if (fDistort) {

                    const Float_t dt = 1E-03;                            //time step [s]
                    const Float_t mu = 4.23;                             //electron mobility [m^2 / s / V]
                    const Float_t mu2 = mu * mu;                         //just square of mu

                    const TVector3 E(0.0, 0.0, fGas->E() *
                                               100);         // vector of electric field components (now is constant and parallel to Z axes) // 100 - convert Ez from V/cm to V/m
                    TVector3 B(0.0, 0.0, 0.0);                           // vector of magnetic field components
                    TVector3 v;                                          // vector of current velocity components
                    TVector3 posCur;                                     // vector of current position components
                    TVector3 posPre = clustPos.Vect();                   // vector of previous position components
                    TVector3 EBCross(0.0, 0.0, 0.0);                     // vector product of E and B vectors

                    Bool_t inTpc = kTRUE;
                    while (inTpc) {

                        B.SetXYZ(fMagField->GetBx(posPre.X(), posPre.Y(), posPre.Z()) * 0.1,
                                 fMagField->GetBy(posPre.X(), posPre.Y(), posPre.Z()) * 0.1,
                                 fMagField->GetBz(posPre.X(), posPre.Y(), posPre.Z()) * 0.1);
                        EBCross = E.Cross(B);

                        v = mu / (1 + mu2 * B.Mag2()) * (E - mu * EBCross + mu2 * B * (E * B));
                        posCur = v * dt + posPre;
                        //                            cout << "X = " << posCur.X() << " Y = " << posCur.Y() << " Z = " << posCur.Z() << " Vx = " << v.X() << " Vy = " << v.Y() << " Vz = " << v.Z() << endl;
                        if ((posCur.Perp() > r_min + fSectHeight) || (posCur.Perp() < r_min) ||
                            (Abs(posCur.Z()) > zCathode))
                            inTpc = kFALSE;
                        posPre = posCur;
                    }
                    distort.SetX(posCur.X() - clustPos.Vect().X());
                    distort.SetY(posCur.Y() - clustPos.Vect().Y());
                    distort.SetZ(0.0);  //FIXME
                }

                electronPos.SetVect(clustPos.Vect() + diffuse + distort);
                electronPos.SetT(clustPos.T() + (zCathode - fabs(electronPos.Z())) /
                                                fGas->VDrift()); // Do we need to use clustPos.T() ???

                /*AZ
                const Float_t phiStep = TwoPi() / nSectors * 2;
                const Float_t sectPhi = secID * phiStep;
                localY =  electronPos.X() * Cos(sectPhi) + electronPos.Y() * Sin(sectPhi) - r_min; //converting from global to local coordinates
                localX = -electronPos.X() * Sin(sectPhi) + electronPos.Y() * Cos(sectPhi);        //converting from global to local coordinates
                if ((localY < 0.0) || (Abs(localX) > fSector->GetMaxX()) || (localY > fSectHeight)) continue; //FIXME!!!
                const Float_t timeStep = (zCathode / fNTimeBins) / fGas->VDrift();
                const UInt_t curTimeID = (UInt_t) ((zCathode / fGas->VDrift() - electronPos.T()) / timeStep);
                */

                //AZ
                TVector3 xyzLoc;
                if (secGeo->Global2Local(electronPos.Vect(), xyzLoc, secID % secGeo->NofSectors()) < 0) continue;
                localX = xyzLoc[0];
                localY = xyzLoc[1];
                if (!TMath::Finite(localX) || !TMath::Finite(localY)) {
                    // Debug
                    cout << " !!! Not finite " << secID % secGeo->NofSectors() << endl;
                    electronPos.Vect().Print();
                    diffuse.Print();
                }
                UInt_t curTimeID = UInt_t(secGeo->T2TimeBin(electronPos.T()));
                //cout << localX << " " << xyzLoc[0] << " " << localY << " " << xyzLoc[1] << " " << curTimeID << " " << timeBin << endl;
                //AZ
                if (curTimeID >= fNTimeBins) continue;
                if (fMakeQA) {
                    fHisto->_hX_local->Fill(localX);
                    fHisto->_hY_local->Fill(localY);
                    fHisto->_hXY_local->Fill(localX, localY);
                    fHisto->_hYZ_local->Fill(fabs(electronPos.Z()), localY);
                    fHisto->_hXY_global->Fill(electronPos.X(), electronPos.Y());
                    fHisto->_hRZ_global->Fill(electronPos.Z(), electronPos.Y());
                    fHisto->_h3D_el->Fill(electronPos.X(), electronPos.Y(), electronPos.Z());
                    fHisto->_hZ_local->Fill(fabs(electronPos.Z()));
                    fHisto->_hX_global->Fill(electronPos.X());
                    fHisto->_hY_global->Fill(electronPos.Y());
                    fHisto->_hZ_global->Fill(electronPos.Z());
                    fHisto->_hDiffuseXY->Fill(diffuse.X(), diffuse.Y());
                    fHisto->_hDistortXY->Fill(distort.X(), distort.Y());
                }

                Int_t origin = prePoint->GetTrackID();
                PadResponse(localX, localY, curTimeID, origin, fDigits4dArray);
            } // for (UInt_t iEll = 0; iEll < clustArr.at(iClust);
        } // for (UInt_t iClust = 0; iClust < clustArr.size();
    } // if (curPoint->GetTrackID() == prePoint->GetTrackID()

    clustArr.clear();
    clustArr.resize(0);
}

//---------------------------------------------------------------------------
void MpdTpcFastDigitizer::Finish() {

    cout << "Digitizer work time = " << ((Float_t) tAll) / CLOCKS_PER_SEC << endl;

    if (fMakeQA) {
        toDirectory("QA/TPC");
        Float_t digit = 0.0;
        UInt_t iPad_shifted = 0; //needed for correct drawing of fDigitsArray

        for (UInt_t iRows = 0; iRows < nRows; ++iRows) {
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[iRows] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[iRows];
                for (UInt_t iTime = 0; iTime < fNTimeBins; ++iTime) {
                    digit = fDigits4dArray[iRows][iPads][iTime].signal;
                    fHisto->_hXY_dig->Fill(iPad_shifted, iRows, digit);
//                        fHisto->_hSect_dig->Fill(iSect, digit);
                    fHisto->_hX_dig->Fill(iPad_shifted, digit);
                    fHisto->_hY_dig->Fill(iRows, digit);
                    fHisto->_hZ_dig->Fill(iTime, digit);
                    fHisto->_h3D_dig->Fill(iPad_shifted, iRows, iTime, digit);
                    if (digit > 1000.0) fHisto->_hADC_dig->Fill(digit);
                }
            }
        }


        for (UInt_t iRows = 0; iRows < nRows; ++iRows) {
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[iRows] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[iRows];
                for (UInt_t iTime = 0; iTime < fNTimeBins; ++iTime) {
                    digit = fDigits4dArray[iRows][iPads][iTime].signal;
                    //pad activity
                    //if (digit > 1000.0) {
                    //    fHisto->_hXY_dig->Fill(iPad_shifted, iRows, 1.0);
                    //}
                    //                    fHisto->_hXY_dig->Fill(iPad_shifted, iRows, digit);
                    fHisto->_h3D_dig->Fill(iPad_shifted, iRows, iTime, digit);
                }
            }
        }

        for (UInt_t iTime = 0; iTime < fNTimeBins; ++iTime) {
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[1] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[1];
                digit = fDigits4dArray[1][iPads][iTime].signal;
                fHisto->_hXT_dig_1->Fill(iPad_shifted, iTime, digit);
            }
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[5] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[5];
                digit = fDigits4dArray[5][iPads][iTime].signal;
                fHisto->_hXT_dig_5->Fill(iPad_shifted, iTime, digit);
            }
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[10] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[10];
                digit = fDigits4dArray[10][iPads][iTime].signal;
                fHisto->_hXT_dig_10->Fill(iPad_shifted, iTime, digit);
            }
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[20] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[20];
                digit = fDigits4dArray[20][iPads][iTime].signal;
                fHisto->_hXT_dig_20->Fill(iPad_shifted, iTime, digit);
            }
            for (UInt_t iPads = 0; iPads < (UInt_t) fNumOfPadsInRow[40] * 2; ++iPads) {
                iPad_shifted = iPads + fNumOfPadsInRow[nRows - 1] - fNumOfPadsInRow[40];
                digit = fDigits4dArray[40][iPads][iTime].signal;
                fHisto->_hXT_dig_40->Fill(iPad_shifted, iTime, digit);
            }
        }

        fHisto->Write();
        gFile->cd();
    }
}

//---------------------------------------------------------------------------

//void MpdTpcFastDigitizer::FastDigi(Int_t isec, const TpcPoint* curPoint)
void MpdTpcFastDigitizer::FastDigi(Int_t isec, const MpdTpcHit *curHit) {
    // Interface to fast digitizer

    MpdTpcSectorGeo *secGeo = MpdTpcSectorGeo::Instance();
    TVector3 pin, pout, xyzLoc;
    TpcPoint *curPoint = (TpcPoint *) fMCPointArray->UncheckedAt(curHit->GetRefIndex());
    curPoint->Momentum(modelInputParameters.mom3);
    curHit->Position(pin);
    /*
    curPoint->Position(pin);
    curPoint->PositionOut(pout);
    pin += pout;
    pin *= 0.5;
    */
    Int_t padID = secGeo->Global2Local(pin, xyzLoc, isec % secGeo->NofSectors());
    if (padID < 0) return;
    curHit->LocalPosition(xyzLoc);
    Double_t zHit0 = pin.Z();
    Double_t secAng = secGeo->SectorAngle(isec % secGeo->NofSectors());
    TVector3 sec3(TMath::Cos(secAng), TMath::Sin(secAng), 0.0);
    TVector3 vecxy(pin.X(), pin.Y(), 0.0);
    if (vecxy * modelInputParameters.mom3 < 0) modelInputParameters.mom3 *= -1; // particle goes inward - change direction
    modelInputParameters.dip = (TMath::PiOver2() - modelInputParameters.mom3.Theta()) * TMath::RadToDeg();
    modelInputParameters.mom3.SetZ(0.0);
    //Double_t cross = (modelInputParameters.mom3.Phi() - sec3.Phi()) * TMath::RadToDeg();
    modelInputParameters.cross = TMath::Sign(modelInputParameters.mom3.Angle(sec3), sec3.Cross(modelInputParameters.mom3).Z()) * TMath::RadToDeg();
    modelInputParameters.tbin = secGeo->Z2TimeBin(zHit0) - 0.4935;
    Double_t yHit0 = xyzLoc.X();
    modelInputParameters.row0 = secGeo->PadRow(padID);
    modelInputParameters.pad0 = (yHit0 - 0.5) / secGeo->PadWidth(0) + secGeo->NPadsInRows()[modelInputParameters.row0] + 0.5;

    vector<float> input = prepareModelInput();
    vector<float> output = prepareModelOutput();
    modelWrapper->modelRun(input.data(), output.data(), input.size(), output.size());
    saveModelRunResultToDigitsArray(
       secGeo, curPoint, modelInputParameters.tbin, yHit0, modelInputParameters.row0,
       modelInputParameters.pad0, output);
}


vector<float> MpdTpcFastDigitizer::prepareModelOutput() const
{
   return vector<float>(128);
}


vector<float> MpdTpcFastDigitizer::prepareModelInput() const
{
   return {static_cast<float>(modelInputParameters.cross),
           static_cast<float>(modelInputParameters.dip),
           static_cast<float>(modelInputParameters.tbin),
           static_cast<float>(modelInputParameters.pad0)};
}


void MpdTpcFastDigitizer::saveModelRunResultToDigitsArray(MpdTpcSectorGeo *secGeo, const TpcPoint *curPoint,
                                                          Double_t tbin, Double_t yHit0, Int_t row0, Double_t pad0,
                                                          const vector<float> &output)
{
   Double_t sum = 0.0, scale = 3.10417e+03 / 3.0e-6 * 1.878, coef =
           curPoint->GetEnergyLoss() * scale; // dedx-to-ADC conversion

   for (Int_t ii = 0; ii < 128; ++ii) sum += output[ii];
   coef /= sum;
   if (Int_t(yHit0 / secGeo->PadWidth(0) + secGeo->NPadsInRows()[row0]) < 0) return; // AZ - 301120

   for (int ii_pad = 0; ii_pad < 8; ii_pad++)
       for (int ii_time = 0; ii_time < 16; ii_time++) {
           Int_t i_pad = static_cast<Int_t>(pad0) + ii_pad - 3;
           Int_t i_time = static_cast<Int_t>(tbin) + ii_time - 7;
           if (i_pad >= 0
               && i_pad < secGeo->NPadsInRows()[row0] * 2
               && i_time >= 0
               && i_time < fSecGeo->GetNTimeBins()) {
               Float_t signal = output[ii_pad * 16 + ii_time] * coef;
               if (signal < 0.1) continue;
               //fDigits4dArray[row0][i_pad][i_time].signal += output[ii_pad * 16 + ii_time];
               fDigits4dArray[row0][i_pad][i_time].signal += signal;
               auto origin = curPoint->GetTrackID();
               auto it = fDigits4dArray[row0][i_pad][i_time].origins.find(origin);
               if (it != fDigits4dArray[row0][i_pad][i_time].origins.end()) {
                   //it->second += output[ii_pad * 16 + ii_time];
                   it->second += signal;
               } else {
                   //fDigits4dArray[row0][i_pad][i_time].origins.insert(pair<Int_t, Floa	t_t>(origin, output[ii_pad * 16 + ii_time]));
                   fDigits4dArray[row0][i_pad][i_time].origins.insert(pair<Int_t, Float_t>(origin, signal));
               }
           }
       }
}

//---------------------------------------------------------------------------

ClassImp(MpdTpcFastDigitizer)
