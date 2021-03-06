# Create a library called "libEtof" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
 ${BASE_INCLUDE_DIRECTORIES}
 ${CMAKE_SOURCE_DIR}/mcstack 
 ${CMAKE_SOURCE_DIR}/tof
 ${CMAKE_SOURCE_DIR}/kalman
 ${CMAKE_SOURCE_DIR}/lhetrack
 ${CMAKE_SOURCE_DIR}/etof
)

Set(SYSTEM_INCLUDE_DIRECTORIES
 ${ROOT_INCLUDE_DIR}
)

include_directories(${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
 ${ROOT_LIBRARY_DIR}
 ${FAIRROOT_LIBRARY_DIR}
)
 
link_directories(${LINK_DIRECTORIES})

# List of source files
set(SRCS
MpdEtofGeo.cxx
MpdEtofGeoPar.cxx
MpdEtof.cxx
MpdEtofContFact.cxx
MpdEtofHitProducer.cxx
MpdEtofHitProducerIdeal.cxx
MpdEtofGeoUtils.cxx
MpdEtofMatching.cxx
)

Set(HEADERS)
Set(LINKDEF MpdEtofLinkDef.h)
Set(LIBRARY_NAME Etof)
Set(DEPENDENCIES)

GENERATE_LIBRARY()
