function(GENERATE_MPDCONFIG [NICA])
	file( WRITE ${CMAKE_BINARY_DIR}/mpd-config "#!/bin/bash\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "mpdlibs=\"\\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-L/${CMAKE_BINARY_DIR}/lib -lbmd -lCluster -lCpc\\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lEmc -lEtof -lEventDisplay -lFfd -lHADGEN \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lKalman -lLHETrack -lMCStack -lMpdBase -lMpdData \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lMpdFemto -lMpdField \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lMpdGeneratorGenerator \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lMpdGen -lMpdPid -lPassive -lStrawECT \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lStrawendcap -lSts -lTHadgen \\\n" )
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lTof -ltpc -lTShieldGenerator \\\n" )
	if ( NICA )
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lTShield -lZdc\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicaFeatures -lNicaDataFormat\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicaCut -lNicaAna\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicaFemto -lNicaFlow\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicaGen -lNicaFluct\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicapdCuts -lNicaMpdFormat\\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lNicaMpdHelper -lNicaSpectra\"\n")
	else()
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-lTShield -lZdc\"\n")
	endif()
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "mpdheads=\"\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/bbc \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/bmd \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/clustering \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/dch \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/emc \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/etof \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/eventdisplay \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/ffd \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/fsa \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/generators \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/kalman \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/lhetrack \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/mcstack \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/mpdbase \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/mpddst \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/mpdfield \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/mpdpid \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/passive \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/physics \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/physics/femto \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/sft \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/shield_pack \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/strawECT \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/strawendcap \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/sts \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/tof \\\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/tpc \\\n")
	if( NICA )
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/tgem \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/base \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/base/chains \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/base/on_the_fly \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/ana \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/base \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/corrfit \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/corrfit/fittingfunctions \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/corrfit/mapgenerators \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/imaging \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/femto/weights \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/flow \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/fluctuations \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/spectra \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/analysis/v0s \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/cutmonitors \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/eventcuts/ \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/trackcuts \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/trackcuts/detector \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/trackcuts/kinematics \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/trackcuts/mc \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/trackcuts/resolution \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/twotrackcuts/detector \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/twotrackcuts/kinematics \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/cuts/twotrackcuts/mc \\\n")

		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/compound \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/detector \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/formats/fair \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/formats/fairextended \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/formats/unigen \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/features \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/nicagenerators/ \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/nicagenerators/readers \\\n")
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/nicaroot/dataformat/nicagenerators/writers \"\n")
	else()
		file( APPEND ${CMAKE_BINARY_DIR}/mpd-config "-I${CMAKE_SOURCE_DIR}/tgem\"\n")	
	endif()

	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "if test $# -eq 0; then\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "exit 1\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "fi\n")

	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "case $1 in\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  " --libs)\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "echo $mpdlibs\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  ";;\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "--cflags)\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "echo $mpdheads\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  ";;\n")
	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config  "esac\n")


	file( COPY ${CMAKE_BINARY_DIR}/mpd-config DESTINATION 
	${CMAKE_BINARY_DIR}/bin/
	FILE_PERMISSIONS GROUP_EXECUTE OWNER_EXECUTE WORLD_EXECUTE WORLD_READ GROUP_READ OWNER_READ)
#	file( APPEND ${CMAKE_BINARY_DIR}/bin/mpd-config
#	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config
#	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config
#	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config
#	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config
#	file( APPEND ${CMAKE_BINARY_DIR}/mpd-config
#				file( APPEND ${CMAKE_BINARY_DIR}/config.sh "export URQMD=${URQMD_PATH}\n")
#	endif()
endfunction()