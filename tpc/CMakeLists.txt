# Create a library called "libtpc" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/mpdfield
  ${CMAKE_SOURCE_DIR}/mcstack
  ${CMAKE_SOURCE_DIR}/tpc
  ${CMAKE_SOURCE_DIR}/tof
  ${CMAKE_SOURCE_DIR}/kalman
  ${CMAKE_SOURCE_DIR}/generators
  ${CMAKE_SOURCE_DIR}/onnxruntime/include
)

Set(SYSTEM_INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
)

include_directories(${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
  ${CMAKE_SOURCE_DIR}/tpc/fastdigimodel
  ${CMAKE_SOURCE_DIR}/onnxruntime/lib
)
 
link_directories(${LINK_DIRECTORIES})

# List of source files
set(SRCS
TaskHelpers.cxx
TpcContFact.cxx
TpcDetector.cxx
TpcGas.cxx
TpcGeo.cxx
TpcGeoPar.cxx
TpcPoint.cxx
MpdTpcSector.cxx
TpcTimeBin.cxx
MpdTpcClusterFinderTask.cxx
MpdTpcDigit.cxx
MpdTpcPeak.cxx
MpdTpcFoundHit.cxx
MpdTpcClusterFinderQAHistograms.cxx
MpdParticleIdentification.cxx
MpdTpcDigitizerTask.cxx
MpdTpcDigitizerQAHistograms.cxx
MpdTpc2dCluster.cxx
MpdTpcHit.cxx
MpdTpcHitProducer.cxx
MpdTpcSectorGeo.cxx
MpdTpcDigitizerAZ.cxx
MpdTpcDigitizerAZlt.cxx
MpdTpcClusterFinderAZ.cxx
MpdTPCpid.cxx
MpdTpcClusterFinderMlem.cxx
MpdTpcEDepParams.cxx
MpdTpcFastDigitizer.cxx
fastdigimodel/TpcFastDigiModelWrapper.cxx
)

Set(HEADERS)
Set(LINKDEF tpcLinkDef.h)
Set(LIBRARY_NAME tpc)
Set(DEPENDENCIES Base Tof Gdml onnxruntime)

GENERATE_LIBRARY()

#add_custom_target(
#        run ALL
#        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/onnxruntime/build.sh
#        COMMENT "Compiling onnxruntime"
#)
#
#add_dependencies(tpc run)
