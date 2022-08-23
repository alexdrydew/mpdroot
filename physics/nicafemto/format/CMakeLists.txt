# Create a library called "" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
Set(INCLUDE_DIRECTORIES
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_DIR}/tpc
${CMAKE_SOURCE_DIR}/mpddata
${CMAKE_SOURCE_DIR}/mpdbase
${CMAKE_SOURCE_DIR}/mpdpid
${CMAKE_SOURCE_DIR}/kalman
${CMAKE_SOURCE_DIR}/mcstack
${CMAKE_SOURCE_DIR}/lhetrack
${CMAKE_SOURCE_DIR}/base/event
${CMAKE_SOURCE_DIR}/base/field 
${CMAKE_SOURCE_DIR}/base/steer
${CMAKE_SOURCE_DIR}/base/sim
${CMAKE_SOURCE_DIR}/base/source
${CMAKE_SOURCE_DIR}/fairtools
${CMAKE_SOURCE_DIR}/mpdbase
${CMAKE_SOURCE_DIR}/mpddst
${CMAKE_SOURCE_DIR}/mpddst/mcDst
${CMAKE_SOURCE_DIR}/mpddst/MpdMiniEvent
${CMAKE_SOURCE_DIR}/generators/unigen_format
${CMAKE_SOURCE_DIR}/physics/nicafemto/format
${NICAFEMTO_FEATURES}
${NICAFEMTO_MPDHELPERS}
${CMAKE_SOURCE_DIR}/external/nicafemto/nicagenerators
${CMAKE_SOURCE_DIR}/external/nicafemto/nicagenerators/readers
${CMAKE_SOURCE_DIR}/external/nicafemto/nicagenerators/writers
${CMAKE_SOURCE_DIR}/external/nicafemto/analysis/v0s
${CMAKE_SOURCE_DIR}/external/nicafemto/analysis/qa
${NICAFEMTO_DATA}
${NICAFEMTO_MPDFORMAT}
)

Set(SYSTEM_INCLUDE_DIRECTORIES
 ${ROOT_INCLUDE_DIR} 
  ${CLHEP_INCLUDE_DIR}
 ${Boost_INCLUDE_DIRS}
)

Set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
  ${CMAKE_SOURCE_DIR}/external/onnxruntime/lib
  ${CMAKE_SOURCE_DIR}/tpc/fastdigimodel
)
message(WARNING,"LINK DIRS ${LINK_DIRECTORIES}")

Include_Directories(${INCLUDE_DIRECTORIES})
Include_Directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})
 
Link_Directories(${LINK_DIRECTORIES})

# List of source files
set(SRCS
combined/NicaMpdDstMCEvent.cxx
mpdevent/NicaMpdEventInterface.cxx
combined/NicaMpdMiniDstFullEvent.cxx
combined/NicaMpdDstMCEventTpcPads.cxx
minidst/NicaMpdMiniDstEvent.cxx
minidst/NicaMpdMiniDstEventInterface.cxx
minidst/NicaMpdMiniDstMcEvent.cxx
minidst/NicaMpdMiniDstMcEventInterface.cxx
minidst/NicaMpdMiniDstMcTrack.cxx
minidst/NicaMpdMiniDstTrack.cxx
minidst/NicaMiniDstSource.cxx
mpdevent/NicaMpdEvent.cxx
mpdevent/NicaMpdEventTpcPads.cxx
mpdevent/NicaMpdTrack.cxx
mpdevent/NicaMpdTrackInterface.cxx
mpdevent/NicaMpdTrackTpcPads.cxx
mpdevent/NicaTrackTpcPads.cxx
mpdmc/NicaMpdMcEvent.cxx
mpdmc/NicaMpdMcEventInterface.cxx
mpdmc/NicaMpdMcTrack.cxx
mpdmc/NicaMpdMcTrackInterface.cxx
unigen/NicaGeneratorReadUnigen.cxx
unigen/NicaGeneratorWriteUnigen.cxx
unigen/NicaUnigenEvent.cxx
unigen/NicaUnigenEventInterface.cxx
unigen/NicaUnigenSource.cxx
unigen/NicaUnigenTrack.cxx
unigen/NicaUnigenTrackInterface.cxx
hbtformat/NicaMpdHbtComplexEvent.cxx
hbtformat/NicaMpdHbtEvent.cxx
hbtformat/NicaMpdHbtTrack.cxx
minidstv0/NicaMiniDstFullV0Event.cxx
minidstv0/NicaMpdMiniDstEventV0.cxx
minidstv0/NicaMpdMiniDstEventV0Interface.cxx
MpdQACoreManager.cxx
)

Set(HEADERS )
Set(LINKDEF NicaMpdInterfaceLinkDef.h)
Set(LIBRARY_NAME NicaMpdFormat)
Set(DEPENDENCIES NicaFeatures NicaGen NicaDataFormat MpdDst MpdBase tpc MpdMCStack NicaMpdHelper UniGenFormat NicaQA onnxruntime)

GENERATE_LIBRARY()


