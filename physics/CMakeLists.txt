add_subdirectory(femto/fsiTools)
add_subdirectory(femto/MpdFemtoMaker)
add_subdirectory(femto/MpdFemtoMakerUser)
add_subdirectory(photons)

# Create a library called "MpdPhysics" which includes the source files given in the array.
# The extension is already found. Any number of sources could be listed here.
set(INCLUDE_DIRECTORIES
 ${BASE_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/mpdbase
  ${CMAKE_SOURCE_DIR}/kalman
  ${CMAKE_SOURCE_DIR}/physics
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
 MpdAnalysisEvent.cxx
 MpdAnalysisTask.cxx
 MpdAnalysisManager.cxx
)

#Set(HEADERS MpdAnalysisEvent.h MpdAnalysisTask.h MpdAnalysisManager.h)
Set(HEADERS)
Set(LINKDEF MpdPhysicsLinkDef.h)
Set(LIBRARY_NAME MpdPhysics)
Set(DEPENDENCIES Base MpdBase)

GENERATE_LIBRARY()
