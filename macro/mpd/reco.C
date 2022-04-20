#include <Rtypes.h>
#if !defined(__CINT__) && !defined(__CLING__)
// ROOT includes
#include "TString.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TChain.h"

// Fair includes
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairTask.h"
#include "FairField.h"

// MPD includes
#include "MpdTpcHitProducer.h"
#include "MpdTpcClusterFinderTask.h"
#include "MpdTpcDigitizerAZlt.h"
#include "MpdTpcClusterFinderAZ.h"
#include "MpdTpcClusterFinderMlem.h"
#include "MpdKalmanFilter.h"
#include "MpdVertexZfinder.h"
#include "MpdTpcKalmanFilter.h"
#include "MpdKfPrimaryVertexFinder.h"
#include "MpdFfdHitProducer.h"
#include "MpdTofHitProducer.h"
#include "MpdEtofHitProducer.h"
#include "MpdEctTrackFinderTpc.h"
#include "MpdEctTrackFinderTof.h"
#include "MpdEctTrackFinderCpc.h"
#include "MpdTofMatching.h"
#include "MpdZdcDigiProducer.h"
#include "MpdEtofMatching.h"
#include "MpdFillDstTask.h"
#include "MpdGetNumEvents.h"
#include "MpdEmcHitCreation.h"

#include <iostream>
#include <cstdlib>
using namespace std;
#endif

R__ADD_INCLUDE_PATH($VMCWORKDIR)
#include "macro/mpd/mpdloadlibs.C"

#define UseFastDigi  // Choose: UseMlem HitProducer UseFastDigi

// Macro for running reconstruction:
// inFile - input file with MC data, default: evetest.root
// nStartEvent - number (start with zero) of first event to process, default: 0
// nEvents - number of events to process, 0 - all events of given file will be proccessed, default: 1
// outFile - output file with reconstructed data, default: mpddst.root
// run_type - proof execution, default "local". e.g.:
//      "proof" - run on proof-lite with "CPU" count workers,
//      "proof:workers=3" - run on proof-lite with 3 workers
//      "proof:user@proof.server:21001" - to run on the PROOF cluster created with PoD (under user 'MPD', default port - 21001)
//      "proof:user@proof.server:21001:workers=10" - to run on the PROOF cluster created with PoD with 10 workers (under USER, default port - 21001)
//	nc-farm : proof:mpd@nc10.jinr.ru:21001


#ifdef UseFastDigi

#endif

void reco(TString inFile = "$VMCWORKDIR/macro/mpd/evetest.root", TString outFile = "mpddst.root", Int_t nStartEvent = 0, Int_t nEvents = 10, TString run_type = "local") {
    // ========================================================================
    // Verbosity level (0=quiet, 1=event level, 2=track level, 3=debug)
    Int_t iVerbose = 0;

    // -----   Timer   --------------------------------------------------------
    TStopwatch timer;
    timer.Start();

    // -----   Digitization run   -------------------------------------------
    // define parallel or sequential execution
    int ind = run_type.Index(':');
    TString proof_name = "";
    if (ind >= 0) {
        proof_name = run_type(ind + 1, run_type.Length() - ind - 1);
        run_type = run_type(0, ind);
    }

    FairRunAna* fRun;
    if (run_type == "proof")
    {
        fRun = new FairRunAnaProof(proof_name);
        ((FairRunAnaProof*)fRun)->SetProofParName("$VMCWORKDIR/gconfig/libMpdRoot.par");
    }
    else
    {
        if (!CheckFileExist(inFile)) return;
        fRun = new FairRunAna();
    }

    FairSource* fFileSource = new FairFileSource(inFile);
    fRun->SetSource(fFileSource);
    fRun->SetOutputFile(outFile);
    fRun->SetGenerateRunInfo(false);
    fRun->SetUseFairLinks(true);
    // ------------------------------------------------------------------------

    // Parameter file
    TString parFile = inFile;

    // -----  Parameter database   --------------------------------------------
    FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
    FairParRootFileIo* parIo1 = new FairParRootFileIo();
    parIo1->open(parFile.Data());
    rtdb->setFirstInput(parIo1);
    rtdb->setOutput(parIo1);
    rtdb->saveOutput();
    // ------------------------------------------------------------------------

    MpdKalmanFilter *kalman = MpdKalmanFilter::Instance("KF");
    fRun->AddTask(kalman);

#ifndef UseMlem
    MpdTpcHitProducer* hitPr = new MpdTpcHitProducer();
    hitPr->SetModular(0);
    //hitPr->SetPersistance(); //AZ
    fRun->AddTask(hitPr);
#else
    //MpdTpcDigitizerAZ* tpcDigitizer = new MpdTpcDigitizerAZ();
    MpdTpcDigitizerAZlt* tpcDigitizer = new MpdTpcDigitizerAZlt();
    tpcDigitizer->SetPersistence(kFALSE);
    fRun->AddTask(tpcDigitizer);
#endif
#ifdef UseFastDigi
    setenv("MLFLOW_URL", "51.250.106.167", false);
    setenv("MINIO_URL", "51.250.106.167", false);
    setenv("ONNX_MODEL_NAME", "baseline_onnx", false);


    MpdTpcFastDigitizer* tpcDigitizer = new MpdTpcFastDigitizer();
    tpcDigitizer->SetPersistence(kFALSE);
    fRun->AddTask(tpcDigitizer);
#endif

    //  MpdTpcClusterFinderTask *tpcClusterFinder = new MpdTpcClusterFinderTask();
    //  tpcClusterFinder->SetDebug(kFALSE);
    //  tpcClusterFinder->SetMakeQA(kTRUE);
    //  tpcClusterFinder->SetCalcResiduals(kFALSE);
    //  fRun->AddTask(tpcClusterFinder);

#ifndef HitProducer
    MpdTpcClusterFinderMlem *tpcClusAZ = new MpdTpcClusterFinderMlem();
    fRun->AddTask(tpcClusAZ);
#endif

    FairTask* vertZ = new MpdVertexZfinder();
    fRun->AddTask(vertZ);

    MpdTpcKalmanFilter* recoKF = new MpdTpcKalmanFilter("Kalman filter");
#ifndef HitProducer
    recoKF->UseTpcHit(kFALSE); // do not use hits from the hit producer
#endif
    fRun->AddTask(recoKF);

    FairTask* findVtx = new MpdKfPrimaryVertexFinder("Vertex finder");
    fRun->AddTask(findVtx);

    MpdFfdHitProducer* ffdHit = new MpdFfdHitProducer("FFDHitProducer");
    fRun->AddTask(ffdHit);

    // TOF hit producers
    MpdTofHitProducer* tofHit = new MpdTofHitProducer("Hit producer");
    tofHit->SetTimeResolution(0.080);
    fRun->AddTask(tofHit);

/*
    MpdEtofHitProducer* etofHitProd = new MpdEtofHitProducer("ETOF HitProducer");
    fRun->AddTask(etofHitProd);
    
    // Endcap tracking
    FairTask* tpcECT = new MpdEctTrackFinderTpc();
    tpcECT->SetVerbose(iVerbose);
    fRun->AddTask(tpcECT);
    
    MpdEctTrackFinderCpc* tofECT = new MpdEctTrackFinderCpc();
    tofECT->SetVerbose(iVerbose);
    tofECT->SetTpc(kTRUE);
    fRun->AddTask(tofECT);
*/

    // TOF matching
    MpdTofMatching* tofMatch = new MpdTofMatching("TOF matching");
    fRun->AddTask(tofMatch);

    // ETOF matching
    //MpdEtofMatching* etofMatch = new MpdEtofMatching("ETOF matching");
    //fRun->AddTask(etofMatch);

    //FairTask *emcHP = new MpdEmcHitCreation();
    //fRun->AddTask(emcHP);

    FairTask *tdigi = new MpdZdcDigiProducer("MpdZdcDigiProducer");
    fRun->AddTask(tdigi);

    //MpdPidRefitTrackTask* trRefit = new MpdPidRefitTrackTask("Track PID and Refit");
    //fRun->AddTask(trRefit);

    MpdFillDstTask* fillDST = new MpdFillDstTask("MpdDst task");
    fRun->AddTask(fillDST);
    
    MpdMiniDstFillTask* miniDst = new MpdMiniDstFillTask(outFile);
    fRun->AddTask(miniDst);
    
    // -----   Intialise   ----------------------------------------------------
    fRun->Init();
    if (run_type != "proof") cout<<"Field: "<<fRun->GetField()->GetBz(0., 0., 0.)<<endl;
    else {
        TProof* pProof = ((FairRunAnaProof*)fRun)->GetProof();
        pProof->SetParameter("PROOF_PacketizerStrategy", (Int_t) 0);
        ind = proof_name.Index(":workers=");
        if (ind >= 0) {
            TString worker_count = proof_name(ind + 9, proof_name.Length() - ind - 9);
            if (worker_count.IsDigit())
                pProof->SetParallel(worker_count.Atoi());
        }
    }

    // if nEvents is equal 0 then all events of the given file starting with "nStartEvent" should be processed
    if (nEvents == 0)
        nEvents = MpdGetNumEvents::GetNumROOTEvents((char*)inFile.Data()) - nStartEvent;

    // -----   Run   ______________--------------------------------------------
    fRun->Run(nStartEvent, nStartEvent + nEvents);

    // -----   Finish   -------------------------------------------------------
    timer.Stop();
    Double_t rtime = timer.RealTime();
    Double_t ctime = timer.CpuTime();
    cout << endl << endl;
    cout << "Macro finished successfully." << endl;      // marker of successful execution for CDASH
    cout << "Output file is " << outFile << endl;
    cout << "Parameter file is " << parFile << endl;
    cout << "Real time " << rtime << " s, CPU time " << ctime << " s" << endl;
    cout << endl;
}
