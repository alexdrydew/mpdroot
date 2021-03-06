//-----------------------------------------------------------
//
// Author List:
//      D.Peresunko, KI, 2019
//
//-----------------------------------------------------------

// This Class' Header ------------------
#include "MpdEmcClusterizerKI.h"
#include "MpdEmcClusterKI.h"
#include "MpdEmcDigitKI.h"
#include "MpdEmcGeoUtils.h"
#include "MpdEmcSimParams.h"
#include "MpdEmcCalibParams.h"
#include "TFile.h"
#include "TClonesArray.h"
#include "TMath.h"
#include <iostream>
using namespace std;

//__________________________________________________________________________

MpdEmcClusterizerKI::MpdEmcClusterizerKI()
  : FairTask("EMC Clusterizer"),
    fNumberOfClusters(0),
    fDigitsArray(nullptr),
    fClustersArray(nullptr),
    fSimParams(nullptr),
    fGeom(nullptr),
    fCalibData(nullptr)
{
}
//__________________________________________________________________________
InitStatus MpdEmcClusterizerKI::Init()
{
  // Get ROOT Manager
  FairRootManager* ioman = FairRootManager::Instance();

  if (ioman == 0) {
    LOG(ERROR) << "RootManager not instantiated!";
    return kERROR;
  }

  //Temporary solution: get calibration from file TODO!!!!
  if(!fCalibData){

    TString CalFile = "$VMCWORKDIR/input/MpdEmcCalib.root";

    TFile * f = new TFile(CalFile) ;

    fCalibData =(MpdEmcCalibParams*)f->Get("CalibrationDef") ;
    cout<<"ECAL: Read out EMC calib data"<<endl;
  }


  // Get input collection
  fDigitsArray = (TClonesArray*)ioman->GetObject("EmcDigit");

  if (fDigitsArray == 0) {
    LOG(ERROR) << "Array of digits not found!";
    return kERROR;
  }

  // Create and register output array
  fClustersArray = new TObjArray(); // Clusters may contain different number of digits=>TObjArray
  ioman->Register("EmcCluster", "Emc", fClustersArray, true);

  // Class with list of parameters
  if (!fSimParams) {
    fSimParams = MpdEmcSimParams::GetInstance();
  }
  if (!fGeom) {
    fGeom = MpdEmcGeoUtils::GetInstance();
  }

  return kSUCCESS;
}

//__________________________________________________________________________
void MpdEmcClusterizerKI::Exec(Option_t* opt)
{
  fClustersArray->Delete(); // No possibility to Clear for TObjArray
  fNumberOfClusters = 0;

  // Prepare Digits: Calibration, bad map, cleanup
  PrepareDigits();

  // Collect list of clusters
  MakeClusters();

  // Split clusters with several local maxima if necessary
  MakeUnfoldings();

  // Evaluate cluster position, dispersion etc.
  EvalClusters();
}
//__________________________________________________________________________
void MpdEmcClusterizerKI::PrepareDigits()
{
  // Apply (de)calibration
  // TODO: Implement Energy and time calibration for data and de-calibrations for MC
  // Apply BadMap
  // TODO: Implement bad map

  // Remove digits below clustering threshold
  int n = fDigitsArray->GetEntriesFast();
  for (int i = 0; i < n; i++) {
    MpdEmcDigitKI* digit = static_cast<MpdEmcDigitKI*>(fDigitsArray->UncheckedAt(i));
    if (!digit) { // already removed e.g. by bad map selection
      continue;
    }
    //Calibrate energy
    digit->SetE(digit->GetE()*fCalibData->GetGain(digit->GetDetId()));//V

    //Calibrate Time TODO!!!

    if (digit->GetE() < fSimParams->DigitMinEnergy()) {
      fDigitsArray->RemoveAt(i);
    }
  }
  fDigitsArray->Compress();
  // Alredy sorted by construction, but ROOT requires this to allow search
  fDigitsArray->Sort();

  LOG(INFO) << "Digits used for clusters:" << fDigitsArray->GetEntriesFast() << endl;
}
//__________________________________________________________________________
void MpdEmcClusterizerKI::MakeClusters()
{
  // Combine digits into cluster according to definition of neighbours in
  // MpdEmcGeoUtils::AreNighbours()
  // Note, that only digits in same sector contribute to the cluster

  // Mark all digits as unused yet
  int nDigits = fDigitsArray->GetEntriesFast();
  if (nDigits < 1) { // nothing to do
    return;
  }
  bool* digitsUsed = new bool[nDigits]{ false };

  int iFirst = 0; // first index of digit which potentially can be a part of cluster
                  // e.g. first digit in this sector

  for (int i = 0; i < nDigits; i++) {
    if (digitsUsed[i])
      continue;

    const MpdEmcDigitKI* digitSeed = static_cast<MpdEmcDigitKI*>(fDigitsArray->UncheckedAt(i));

    // is this digit so energetic that start cluster?
    MpdEmcClusterKI* clu = nullptr;
    int iDigitInCluster = 0;
    if (digitSeed->GetE() > fSimParams->ClusteringThreshold()) {
      // start a new EMC RecPoint
      clu = new MpdEmcClusterKI(digitSeed);
      fClustersArray->AddAtAndExpand(clu, fNumberOfClusters);
      fNumberOfClusters++;
      digitsUsed[i] = true;
      iDigitInCluster = 1;
    } else {
      continue;
    }

    // Now scan remaining digits in list to find neigbours of our seed
    int index = 0;
    while (index < iDigitInCluster) { // scan over digits already in cluster
      int digitInCluTowerId = clu->GetDigitTowerId(index);
      index++;
      for (Int_t j = iFirst; j < nDigits;
           j++) { // upper limit not really matters, AreNeighbour stops this loop for too far digits
        if (digitsUsed[j]) {
          continue; // look through remaining digits
        }
        const MpdEmcDigitKI* digitN = static_cast<MpdEmcDigitKI*>(fDigitsArray->UncheckedAt(j));

        // call (digit,digitN) in THAT oder !!!!!
        Int_t ineb = fGeom->AreNeighbours(digitInCluTowerId, digitN->GetDetId());
        switch (ineb) {
          case -1: // too early (e.g. previous sector), do not look before j at subsequent passes
            iFirst = j + 1;
            continue;
          case 0: // not a neighbour
            continue;
          case 1: // are neighbours
            // check if digits have consistent times
            // Be VERY careful with this cut!!! Too strict may strongly modify your spectrum
            if (std::fabs(digitN->GetTime() - clu->GetTime()) < fSimParams->ClusteringTimeGate()) {
              clu->AddDigit(digitN);
              iDigitInCluster++;
              digitsUsed[j] = true;
            }
            continue;
          case 2: // too far from each other, stop loop
          default:
            goto nextDigit;

        } // switch
      }
    nextDigit:;
    } // loop over cluster
  }   // energy theshold
  delete[] digitsUsed;
}
//__________________________________________________________________________
void MpdEmcClusterizerKI::MakeUnfoldings()
{
  //Split cluster if several local maxima are found

  if (!fSimParams->UnfoldClusters()) {
    return;
  }
  int* maxAt = new int[fSimParams->NLMMax()]; // NLMMax:Maximal number of local maxima
  float* maxAtEnergy = new float[fSimParams->NLMMax()];

  float localMaxCut = fSimParams->LocalMaximumCut();
  int numberOfNotUnfolded = fNumberOfClusters;
  for (int index = 0; index < numberOfNotUnfolded; index++) {
    MpdEmcClusterKI* clu = static_cast<MpdEmcClusterKI*>(fClustersArray->At(index));

    Int_t nMultipl = clu->GetMultiplicity();
    int nMax = clu->GetNumberOfLocalMax(maxAt, maxAtEnergy);
    if (nMax > 1) {
      UnfoldOneCluster(clu, nMax, maxAt, maxAtEnergy);

      fClustersArray->Remove(clu);
      fClustersArray->Compress();
      index--;
      fNumberOfClusters--;
      numberOfNotUnfolded--;
    } else {
      clu->SetNLM(1); // Only one local maximum
    }
  }
  delete[] maxAt;
  delete[] maxAtEnergy;
}
//____________________________________________________________________________
void MpdEmcClusterizerKI::UnfoldOneCluster(MpdEmcClusterKI* iniClu, Int_t nMax, int* digitId, float* maxAtEnergy)
{
  // Performs the unfolding of a cluster with nMax overlapping showers
  // Parameters: iniClu cluster to be unfolded
  //             nMax number of local maxima found (this is the number of new clusters)
  //             digitId: index of digits, corresponding to local maxima
  //             maxAtEnergy: energies of digits, corresponding to local maxima

  // Take initial cluster and calculate local coordinates of digits
  // To avoid multiple re-calculation of same parameters
  int mult = iniClu->GetMultiplicity();
  std::vector<double> x(mult);
  std::vector<double> y(mult);
  std::vector<double> z(mult);
  std::vector<double> e(mult);
  std::vector<std::vector<double>> eInClusters(mult, std::vector<double>(nMax));

  for (int idig = 0; idig < mult; idig++) {
    Int_t detID;
    Float_t eDigit;
    ;
    iniClu->GetTransientDigitParams(idig, detID, eDigit);
    e[idig] = eDigit;
    double lx, ly, lz;
    fGeom->DetIdToGlobalPosition(detID, lx, ly, lz);
    x[idig] = lx;
    y[idig] = ly;
    z[idig] = lz;
  }

  // Coordinates of centers of clusters
  std::vector<double> xMax(nMax);
  std::vector<double> yMax(nMax);
  std::vector<double> zMax(nMax);
  std::vector<double> eMax(nMax);

  for (int iclu = 0; iclu < nMax; iclu++) {
    xMax[iclu] = x[digitId[iclu]];
    yMax[iclu] = y[digitId[iclu]];
    zMax[iclu] = z[digitId[iclu]];
    eMax[iclu] = e[digitId[iclu]];
  }

  std::vector<double> prop(nMax); // proportion of clusters in the current digit

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;
  while (insuficientAccuracy && nIterations < fSimParams->NMaxIterations()) {
    // Loop over all digits of parent cluster and split their energies between daughter clusters
    // according to shower shape
    for (int idig = 0; idig < mult; idig++) {
      double eEstimated = 0;
      for (int iclu = 0; iclu < nMax; iclu++) {
        prop[iclu] = eMax[iclu] * ShowerShape(sqrt((x[idig] - xMax[iclu]) * (x[idig] - xMax[iclu]) +
                                                   (y[idig] - yMax[iclu]) * (y[idig] - yMax[iclu])),
                                              z[idig] - zMax[iclu]);
        eEstimated += prop[iclu];
      }
      if (eEstimated == 0.) { // numerical accuracy
        continue;
      }
      // Split energy of digit according to contributions
      for (int iclu = 0; iclu < nMax; iclu++) {
        eInClusters[idig][iclu] = e[idig] * prop[iclu] / eEstimated;
      }
    }
    // Recalculate parameters of clusters and check relative variation of energy and absolute of position
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    for (int iclu = 0; iclu < nMax; iclu++) {
      double oldX = xMax[iclu];
      double oldY = yMax[iclu];
      double oldZ = zMax[iclu];
      double oldE = eMax[iclu];
      // new energy, need for weight
      eMax[iclu] = 0;
      for (int idig = 0; idig < mult; idig++) {
        eMax[iclu] += eInClusters[idig][iclu];
      }
      xMax[iclu] = 0;
      yMax[iclu] = 0;
      zMax[iclu] = 0.;
      double wtot = 0.;
      for (int idig = 0; idig < mult; idig++) {
        double w = std::max(std::log(eInClusters[idig][iclu] / eMax[iclu]) + fSimParams->LogWeight(), 0.);
        xMax[iclu] += x[idig] * w;
        yMax[iclu] += y[idig] * w;
        zMax[iclu] += z[idig] * w;
        wtot += w;
      }
      if (wtot > 0.) {
        xMax[iclu] /= wtot;
        yMax[iclu] /= wtot;
        zMax[iclu] /= wtot;
      }
      // Compare variation of parameters
      insuficientAccuracy += (std::abs(eMax[iclu] - oldE) > fSimParams->UnfogingEAccuracy());
      insuficientAccuracy += (std::abs(xMax[iclu] - oldX) > fSimParams->UnfogingXZAccuracy());
      insuficientAccuracy += (std::abs(yMax[iclu] - oldY) > fSimParams->UnfogingXZAccuracy());
      insuficientAccuracy += (std::abs(zMax[iclu] - oldZ) > fSimParams->UnfogingXZAccuracy());
    }
    nIterations++ ;
  }

  // Iterations finished, add new clusters
  for (int iclu = 0; iclu < nMax; iclu++) {
    MpdEmcClusterKI* clu = new MpdEmcClusterKI();
    fClustersArray->AddAtAndExpand(clu, fNumberOfClusters + iclu);
    clu->SetNLM(nMax);
  }
  for (int idig = 0; idig < mult; idig++) {
    Int_t detID;
    Float_t eDigit;
    ;
    iniClu->GetTransientDigitParams(idig, detID, eDigit);
    MpdEmcDigitKI testdigit(detID, 0, 0, 0);               // test digit
    int jdigit = fDigitsArray->BinarySearch(&testdigit); // Look for digit with same detID
    if (jdigit == -1) {
      LOG(ERROR) << "MpdEmcClusterizerKI::UnfoldOneCluster: Can not find Digit with detID=" << detID;
      continue;
    }
    MpdEmcDigitKI* digit = static_cast<MpdEmcDigitKI*>(fDigitsArray->At(jdigit));
    for (int iclu = 0; iclu < nMax; iclu++) {
      MpdEmcClusterKI* clu = static_cast<MpdEmcClusterKI*>(fClustersArray->UncheckedAt(fNumberOfClusters + iclu));
      clu->AddDigit(digit, eInClusters[idig][iclu]); // Fills geometry and MC infor from Digit,+ correct energy
    }
  }
  fNumberOfClusters += nMax;
}
//__________________________________________________________________________
void MpdEmcClusterizerKI::EvalClusters()
{
  // Calculate cluster properties
  int n = fClustersArray->GetEntriesFast();
  LOG(DEBUG) << "EvalCluProperties: nclu=" << n;

  for (int i = 0; i < n; i++) {
    MpdEmcClusterKI* clu = static_cast<MpdEmcClusterKI*>(fClustersArray->UncheckedAt(i));
    // Remove remaining digits below threshold, e.g. after unfolding
    clu->Purify();
    // Eval all variables: Energy, CoreEnergy, position, Dispersion,...
    clu->EvalAll();
  }
}
//__________________________________________________________________________
double MpdEmcClusterizerKI::ShowerShape(double dx, double dz)
{
  // Shower shape of EM cluster: proportion of energy in cell as a distancs dx,dz (phi,z directions)
  // from the center of gravity of cluster.
  // Due to projective geometry may also depend on Z coordinate. TODO: explore Z-dependence

  // Parameterization from V.Riabov. TODO: verify with beam-test

  double frac = 0;
  double x = std::sqrt(dx * dx + dz * dz);

  if (x < 0.25) return 0.73;

      if (x < 4.5)
	{
	  frac = 7.55666e+000*exp(-2.35773e+000  + 1.23342e-001*x -2.53958e-001*x*x +1.41214e-002*x*x*x );
	}      
      if ( x >= 4.5 && x < 8.33 )
	{
	  frac = 4.26832e-002*exp(1.47109e+000  -2.06712e-001*x -6.60220e-002*x*x +4.88207e-003*x*x*x );
	}
      if ( x >= 8.33 && x < 2.65*4 )
	{
	  frac = 4.46227e-002*exp( 1.56817e+000 -7.41051e-001*x/4 -4.80563e-001*x/4*x/4 );
	}
      if ( x >= 2.65*4 )
	{
	  frac = 1.24899e-002*exp( 3.60075e-001 -8.15748e-001*x/4 -3.74305e-002*x/4*x/4*x/4 );
	}


  if (frac < 1e-24)  frac = 1e-24;

  return frac;
}

/*
//__________________________________________________________________________

void MpdEmcClusterizerKI::GetPhiTheta(Double_t &phi, Double_t &theta)
{
  // Convert COG in units of bins to angles

  static Int_t first = 1, offset = 0;
  static TSpline3 *phiS, *theS;

  if (first) {
    // Get phi and theta angles of the tower centers at their inner face
    first = 0;
    const vector<Double_t> &phis = fEmcGeo->GetPhiRow();
    const vector<Double_t> &thes = fEmcGeo->GetThetaBox();
    const vector<Double_t> &rhos = fEmcGeo->GetRhoCenterBox();
    const vector<Double_t> &zs =   fEmcGeo->GetZCenterBox();

    Int_t nphi = phis.size();
    // Offset due to the fact that the 1'st sector starts at phi = -Phi_sec/2;
    offset = nphi / (fEmcGeo->GetNsec()/2 - 1) / 2;
    Double_t *phia = new Double_t [nphi];
    Double_t *ind = new Double_t [nphi];

    for (Int_t j = 0; j < nphi; ++j) {
      phia[j] = phis[j];
      ind[j] = j;
    }
    phiS = new TSpline3("grs",ind,phia,nphi); // phi vs ind
    delete [] phia;
    delete [] ind;

    Int_t nthe = thes.size();
    Double_t *the = new Double_t [nthe];
    Double_t *ind1 = new Double_t [nthe];
    Double_t height = fEmcGeo->GetLengthBox(); // tower half-height

    for (Int_t j = nthe-1; j >= 0; --j) {
      Double_t rho = rhos[j];
      Double_t z = zs[j];
      Double_t theta1 = thes[j];
      if (j < nthe-1 && thes[j] <= thes[j+1]+0.1) theta1 = 180 - theta1;
      Double_t costhe = TMath::Cos(theta1*TMath::DegToRad());
      Double_t sinthe = TMath::Sin(theta1*TMath::DegToRad());
      rho -= height * sinthe;
      z -= height * costhe;
      the[j] = TMath::ATan2(rho,z) * TMath::RadToDeg();
      ind1[j] = j; // - nthe/2;
    }
    theS = new TSpline3("grs1",ind1,the,nthe); // theta vs ind
    delete [] the;
    delete [] ind1;
  }

  phi = phiS->Eval(phi-offset);
  if (phi > 180) phi -= 360;
  theta = theS->Eval(theta);

}
*/
//__________________________________________________________________________
ClassImp(MpdEmcClusterizerKI)
