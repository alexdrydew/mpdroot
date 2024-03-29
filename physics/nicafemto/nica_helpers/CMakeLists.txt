# Create a library called "" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
Set(INCLUDE_DIRECTORIES
${BASE_INCLUDE_DIRECTORIES}
${CMAKE_SOURCE_FAIRMCSTACK}
${CMAKE_SOURCE_DIR}/tpc
${CMAKE_SOURCE_DIR}/mpddata
${CMAKE_SOURCE_DIR}/mpdbase
${CMAKE_SOURCE_DIR}/mpdpid
${CMAKE_SOURCE_DIR}/kalman
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
${CMAKE_SOURCE_DIR}/physics/nicafemto/nica_helpers
${NICAFEMTO_FEATURES}
${NICAFEMTO_DATA}
${NICAFEMTO_CUTS}
${NICAFEMTO_ANA}
 ${CLHEP_INCLUDE_DIR}
 ${Boost_INCLUDE_DIRS}
)


Set(SYSTEM_INCLUDE_DIRECTORIES
 ${ROOT_INCLUDE_DIR} 
)

Set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
  ${CMAKE_SOURCE_DIR}/external/onnxruntime/lib
)

Include_Directories(${INCLUDE_DIRECTORIES})
Include_Directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})
 
Link_Directories(${LINK_DIRECTORIES})

# List of source files
set(SRCS
MpdPIDOnTheFly.cxx
NicaPairAnaDebugger.cxx
MpdSharedHitMap.cxx
MpdHitsOnTheFly.cxx
NicaMpdConst.cxx
NicaTpcSectorGeo.cxx
)

Set(HEADERS )
Set(LINKDEF NicaMpdHelpersLinkDef.h)
Set(LIBRARY_NAME NicaMpdHelper)
Set(DEPENDENCIES NicaFeatures NicaDataFormat MpdDst tpc Base NicaCut NicaAna onnxruntime)

GENERATE_LIBRARY()


