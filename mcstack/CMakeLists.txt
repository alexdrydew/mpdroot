# Create a library called "libMCStack" which includes the source files given in
# the array.
# The extension is already found.  Any number of sources could be listed here.

set(INCLUDE_DIRECTORIES
  ${BASE_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/mcstack
  ${CMAKE_SOURCE_DIR}/generators 
)

include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
)
 
link_directories(${LINK_DIRECTORIES})

set(SRCS
MpdStack.cxx 
MpdMCTrack.cxx
)

set(HEADERS
MpdStack.h 
MpdMCTrack.h
MpdDetectorList.h
)

Set(LINKDEF MpdMCStackLinkDef.h)
Set(LIBRARY_NAME MpdMCStack)
Set(DEPENDENCIES)

GENERATE_LIBRARY()
