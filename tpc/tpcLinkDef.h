// $Id: TofLinkDef.h,v 1.3 2006/03/07 11:51:55 friese Exp $

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class MpdMcDst++;
#pragma link C++ class MpdMcDstReader++;
#pragma link C++ class MpdMcEvent++;
#pragma link C++ class MpdMcParticle++;
#pragma link C++ class MpdMcPIDConverter++;
#pragma link C++ class MpdMcRun++;

#pragma link C++ class TpcDetector+;
#pragma link C++ class TpcPoint+;
#pragma link C++ class TpcGeo+;
#pragma link C++ class TpcGeoPar+;
#pragma link C++ class TpcContFact+;
#pragma link C++ class MpdParticleIdentification+;
#pragma link C++ class MpdTpcPeak+;
#pragma link C++ class MpdTpcFoundHit+;
#pragma link C++ class TpcSector+;
#pragma link C++ class TpcTimeBin+;
#pragma link C++ class MpdTpcClusterFinderTask+;
#pragma link C++ class MpdTpcDigit+;
#pragma link C++ class MpdTpcClusterFinderQAHistograms+;
#pragma link C++ class MpdTpcDigitizerTask+;
#pragma link C++ class MpdTpcDigitizerQAHistograms+;
#pragma link C++ class MpdTpc2dCluster+;
#pragma link C++ class MpdTpcHit+;
#pragma link C++ class MpdTpcHitProducer+;
#pragma link C++ class MpdTpcSectorGeo+;
#pragma link C++ class MpdTpcDigitizerAZ+;
#pragma link C++ class MpdTpcDigitizerAZlt+;
#pragma link C++ class MpdTpcClusterFinderAZ+;
#pragma link C++ class MpdTPCpid+;
#pragma link C++ class MpdTpcClusterFinderMlem+;
#pragma link C++ class MpdTpcEDepParams+;
#pragma link C++ class MpdTpcFastDigitizer+;
#pragma link C++ class TpcFastDigiModelWrapper+;
#pragma link C++ class OnnxruntimeTpcFastDigiModelWrapper;

#endif
