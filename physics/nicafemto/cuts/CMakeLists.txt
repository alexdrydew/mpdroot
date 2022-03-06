# Create a library called "" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
Set(INCLUDE_DIRECTORIES
${CMAKE_SOURCE_FAIRMCSTACK}
${CMAKE_SOURCE_DIR}/tpc
${CMAKE_SOURCE_DIR}/kalman
${CMAKE_SOURCE_DIR}/lhetrack
${CMAKE_SOURCE_DIR}/mpddata
${CMAKE_SOURCE_DIR}/base/event
${CMAKE_SOURCE_DIR}/base/field 
${CMAKE_SOURCE_DIR}/base/steer
${CMAKE_SOURCE_DIR}/base/sim
${CMAKE_SOURCE_DIR}/base/source
${CMAKE_SOURCE_DIR}/fairtools
${NICAFEMTO_FEATURES}
${NICAFEMTO_DATA}
${NICAFEMTO_CUTS}
${NICAFEMTO_MPDFORMAT}
${NICAFEMTO_MPDHELPERS}
${NICAFEMTO_MPDCUTS}
${CMAKE_SOURCE_DIR}/mpdbase
)

Set(SYSTEM_INCLUDE_DIRECTORIES
 ${ROOT_INCLUDE_DIR}
 ${BASE_INCLUDE_DIRECTORIES} 
)

Set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR} 
  ${FAIRROOT_LIBRARY_DIR}
  ${CMAKE_SOURCE_DIR}/onnxruntime/lib
)

Include_Directories(${INCLUDE_DIRECTORIES})
Include_Directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})
 
Link_Directories(${LINK_DIRECTORIES})

# List of source files
set(SRCS
trackcuts/MpdTpcBasicCut.cxx
trackcuts/MpdTrackChargeCut.cxx
trackcuts/MpdTrackFirstPointCut.cxx
trackcuts/MpdNSigmaCut.cxx
trackcuts/MpdPadsEdgeSectorCut.cxx
trackcuts/MpdHitsEdgeSectorCut.cxx
trackcuts/MpdBasicTrackCut.cxx
trackcuts/MpdTrackIsGlobalCut.cxx
#paircuts/MpdTwoTrackSharedQualityCut.cxx



paircuts/common/MpdPairDeltaPhiStarDeltaEtaCut.cxx
paircuts/common/MpdSplittedPairsCut.cxx
paircuts/common/MpdPairDeltaDCA.cxx
paircuts/common/tpc_cylinder/MpdPairTpcSimpleCut.cxx
paircuts/common/tpc_cylinder/MpdPairTpcSeparationSimpleCut.cxx
paircuts/common/tpc_cylinder/MpdPairTpcEntrySimpleCut.cxx
paircuts/common/tpc_cylinder/MpdCylinderTpcPairCut.cxx
paircuts/common/MpdPairDeltaPhistarMinCut.cxx

paircuts/full/tpc_modular/MpdModularTpcPairCut.cxx
paircuts/full/tpc_modular/MpdPairSharedPadsCut.cxx
paircuts/full/tpc_modular/MpdPairTpcEntranceCut.cxx
paircuts/full/tpc_modular/MpdTpcPairSeparationCut.cxx
paircuts/full/MpdDeltaEtaDeltaPhiStarAdvancedCut.cxx
paircuts/full/MpdAlicePairQuality.cxx
paircuts/full/MpdDeltaEtaDeltaPhiStarAdvancedCut.cxx
paircuts/full/MpdHelixSep.cxx
paircuts/full/MpdNominalTpcPairPadsDistanceCut.cxx
paircuts/full/MpdPairSharedHitsCut.cxx
paircuts/full/MpdStarPairQualityCut.cxx
paircuts/full/MpdTpcPadsPairCut.cxx

paircuts/minidst/MpdFemtoPairCut.cxx
paircuts/minidst/MpdSharedPadsCut.cxx
paircuts/minidst/MpdPairTpcEntranceNomininalCut.cxx

cutmonitors/MpdKinMonitor.cxx
cutmonitors/MpdTofMonitor.cxx
cutmonitors/MpdTpcMonitor.cxx
cutmonitors/MpdDcaMonitor.cxx
)

Set(HEADERS )
Set(LINKDEF MpdNicaCutsLinkDef.h)
Set(LIBRARY_NAME NicaMpdCuts)
Set(DEPENDENCIES NicaCut NicaMpdFormat NicaMpdHelper onnxruntime curl)

GENERATE_LIBRARY()


