!! SHIELD sub is not used (A.Timofeev)
!      SUBROUTINE SHIELD
!C The SHIELD Hadron Transport Code.
!C
!      COMMON /RANDOM/ IX,IXINIT,NUMTRE,IXFIRS,LASTIX
!      COMMON /OLN/ OLN
!          COMMON /HIPRO0/ APRO0,ZPRO0    ! For the input only!       !HION3
!          COMMON /HIPROJ/ APROJ,ZPROJ    ! For AGT, CALMHI           !HION3
!      COMMON /ANTLAB/ lanti          ! Lab-AntiLab flag. LANTI=0 only!
!      COMMON /DEBUG/ LSTAR,LCASC     ! LSTAR=0 only!  LCASC=0/1
!      COMMON /T0JPR0/ TMAX0,JPART0   ! to give T0 and JPART0 into INSPAR
!      COMMON /ISOSET/ WISOSET(2000),FERTAZ(2000,50),  ! Fertman ! Truba06
!     *                ISOSET(2,2000),TDEC(2000),NIS200,NISTRU   ! userset
!      CHARACTER*10 TDEC   ! userset
!C
!C SET DEFAULT FLAGS:
!      lanti=0
!      LSTAR=0
!      LCASC=0
!C                 Input and scratch files
!      OPEN(16,FILE='PASIN.DAT',STATUS='OLD')                         ! GEMCA
!!      OPEN(17,FILE='FOR017',STATUS='UNKNOWN')                        ! GEMCA
!      OPEN(18,FILE='ATAB.DAT',STATUS='OLD')
!      OPEN(19,FILE='TABNUC.DAT',STATUS='OLD')
!      OPEN(22,FILE='FOR022.DAT',STATUS='OLD')
!      OPEN(23,FILE='FOR023.DAT',STATUS='OLD')
!      OPEN(45,FILE='ISOTIN.DAT',STATUS='OLD')     ! userset
!C                 Output files
!      OPEN(24,FILE='FOR024',STATUS='NEW')
!!      OPEN(25,FILE='FOR025',STATUS='NEW')
!      OPEN(26,FILE='FOR026')
!      OPEN(27,FILE='FOR027')
!      OPEN(44,FILE='nmtc_yield',STATUS='NEW')    ! DCHAIN
!      OPEN(46,FILE='ISOTOUT',STATUS='NEW')       ! userset 
!C
!C MAIN INPUT
!      READ(23,1)IXFIRS,LUXCNT,  ! Initial random generator state  ! RANLUX
!     *          JPART0,  ! Type of incident particle
!     *          TMAX0,   ! Incident energy
!     *          NSTAT,NSAVE,  ! Monte Carlo statistics and saving    ! Nsave
!     *          OLN,     ! Cutoff energy for LOENT
!     *          LCASC    ! 0 - AGT only; 1- CASCAD at E<1 GeV
!C
!    1 FORMAT(/20X,2I12,36X/                                       ! RANLUX
!     *        20X,I12,48X/
!     *        20X,F12.3,48X/
!     *        20X,2I12,36X/                                          ! Nsave
!     *        20X,E12.4,48X/
!     *        20X,I12,48X)
!C
!      IF(JPART0.eq.25)THEN                                           !HION3
!          READ(23,3)APRO0,   ! A of Nucleus-projectile               !HION3
!     *              ZPRO0    ! Z of Nucleus-projectile               !HION3
!    3     FORMAT(/20X,F12.3,48X/                                     !HION3
!     *            20X,F12.3,48X/)                                    !HION3
!      END IF                                                         !HION3
!      CLOSE(23)                                                      ! Nsave
!C
!      TMAX=TMAX0                        ! Top limit for histogramms
!*      IF(OLN.LT.0.215E-06)OLN=0.215E-06 ! Thermal energy
!      if(LCASC.ne.1)LCASC=0             ! 0 or 1 only!
!      LCASC=0  ! KarSjh: to exclude accidental run of CASEVP instead of AGT.
!C
!      CONTINUE!WRITE(24,11)
!      WRITE(44,11)   ! DCHAIN
!      WRITE(46,11)   ! userset
!   11 FORMAT(/10X,'INPUT DATA FOR SHIELD TRANSPORT CALCULATION.')
!      CONTINUE!WRITE(24,12)JPART0,TMAX0,NSTAT,OLN
!      WRITE(44,12)JPART0,TMAX0,NSTAT,OLN   ! DCHAIN
!      WRITE(46,12)JPART0,TMAX0,NSTAT,OLN   ! userset
!   12 FORMAT(14X,'TYPE of incident particle/nucleus:',I6/            !HION3
!     *       14X,'Incident ENERGY     (MeV)/(MeV/A):',F10.1/         !HION3
!     *       14X,'Monte Carlo STATISTICS:',I9/
!     *       14X,'CUTOFF energy for neutron transport (MeV):',E11.4)
!      IF(JPART0.eq.25)CONTINUE!WRITE(24,14)JPART0,INT(APRO0),INT(ZPRO0)       !HION3
!      IF(JPART0.eq.25)WRITE(44,14)JPART0,INT(APRO0),INT(ZPRO0)   !HION3 ! DCHAIN
!      IF(JPART0.eq.25)WRITE(46,14)JPART0,INT(APRO0),INT(ZPRO0)   !HION3 ! userset
!   14 FORMAT(14X,7X,'Nucleus-projectile TYPE',I3,': (',I3,',',I2,')')!HION3
!      IF(LCASC.eq.1)CONTINUE!WRITE(24,13)
!   13 FORMAT(14X,'(Note: Using of CASCAD below 1 GeV)')
!C
!C FIXING OF ARRAYS DIMENSIONS
!      CALL DIMAR
!C Set parameters for AGT, PRECO, DEEX, and CASCAD
!      CALL INLEVR
!C
!C//////////////////// userset /////////////////////
!C Input of isotope set (Zi,Ai)
!      NISTRU=1  ! Counter of isotope number i
!      READ(45,51)
!   51 FORMAT()  ! To skip title line
!   52 READ(45,53,END=6)ISOSET(1,NISTRU),ISOSET(2,NISTRU),TDEC(NISTRU)
!   53 FORMAT(2I4,A10)
!      NISTRU=NISTRU+1
!      GOTO 52
!    6 NISTRU=NISTRU-1
!      IF(NISTRU.gt.NIS200)THEN
!        CONTINUE!WRITE(25,1000)NISTRU,NIS200
! 1000   FORMAT(' SHIELD Diagnostic: NISTRU=',I5,' > NIS200=',I5)
!        STOP
!      END IF
!cC         Debug printout
!c      CONTINUE!WRITE(25,7)NISTRU
!c    7 format(/'Input isotope set, NISTRU=',I4)
!c      do j=1,NISTRU
!c        CONTINUE!WRITE(25,8)j,ISOSET(1,j),ISOSET(2,j),TDEC(j)
!c    8   format(I5,I7,I4,A10)
!c      end do
!      CLOSE(45)
!C//////////////////// end userset /////////////////
!C
!C INPUT OF MEDIA PARAMETERS AND PRINTOUT OF ITS DEFAULT VALUES
!      CALL INMACR
!      CLOSE(22)                                                      ! Nsave
!      CALL BDPRIN
!C
!C CALCULATION OF MACROSCOPICAL CROSS SECTIONS,
!C RANGE-ENERGY DEPENDENCES, OPTICAL DEPTHS.
!      CALL CALMAC
!      CALL ROPTPR                                                    !HION8
!        IF(JPART0.eq.25)THEN                                  !HION3
!          APROJ=APRO0                                         !HION3
!          ZPROJ=ZPRO0                                         !HION3
!          CALL CALMHI(1)                                      !HION3
!          CALL ROPTHI(1)                                             !HION8
!        END IF                                                !HION3
!      CLOSE(26)                                                      ! Nsave
!      CALL LNMACR
!      CLOSE(27)                                                      ! Nsave
!C
!C CLEANING OF ARRAYS FOR HISTOGRAMMS
!      CALL CLETOT
!      CALL CLERLN
!C
!C Geometry INITIALIZATION                                            ! GEMCA
!	CALL GEOINI                                                  ! GEMCA
!C             Writing of PASIN.DAT (unit 16) into FOR024 (unit 24)   ! GEMCA
!	      CLOSE(16)                                              ! GEMCA
!	      OPEN(16,FILE='PASIN.DAT',STATUS='OLD')                 ! GEMCA
!	      CALL PUTFIL(16,24)                                     ! GEMCA
!	      CLOSE(16)                              ! userset       ! GEMCA
!	      OPEN(16,FILE='PASIN.DAT',STATUS='OLD') ! userset       ! GEMCA
!	      CALL PUTFIL(16,46)                     ! userset       ! GEMCA
!              CLOSE(16)                                   ! GEMCA    ! Nsave
!              CLOSE(17,STATUS='DELETE')                   ! GEMCA    ! Nsave
!C
!C ------------------- KI05Fragm ----------------------
!C Writing of Fragments.f into For024 for identificatin
!      CONTINUE!WRITE(24,21)
!   21 FORMAT(/'C ',25('-'),' The file Fragments.f ',27('-'))
!	OPEN(61,FILE='Fragments.f',STATUS='OLD')
!	CALL PUTFIL(61,24)
!      CLOSE(61)
!C ------------------- end KI05Fragm ------------------
!C
!C Set cutoff energies for charged particles transport
!      CALL CUTOFF
!C Initialization of ATAB for Amelin generator
!      CALL INITAM
!C Initialization of TABNUC for sampling A from natural isotope mixture
!      CALL INITIS
!      CLOSE(19)                                                      ! Nsave
!C Initialization of the random number generator RANLUX
!      CALL INIT_LUX(IXFIRS,LUXCNT,24)                           ! RANLUX
!C Writing of label "END OF TITLES" into FILE 24                      ! Nsave
!      CONTINUE!WRITE(24,33)                                                    ! Nsave
!      WRITE(44,33)               ! DCHAIN                             ! Nsave
!      WRITE(46,33)               ! userset                             ! Nsave
!   33 FORMAT('END OF TITLES')                                        ! Nsave
!      CLOSE(24)                                                      ! Nsave
!      CLOSE(44)                  ! DCHAIN                            ! Nsave
!      CLOSE(46)                  ! userset                            ! Nsave
!C
!C STATISTICS: MAIN LOOP
!      IX=IXFIRS
!cC ============Forced initialization of RANLUX for debug=============
!c        LUXX=3
!c        ILT=19736501
!cc        K11=50915606
!c        K11=385059918
!c        K22=1
!c        CALL RLUXGO(LUXX,ILT,K11,K22)
!cC===================================================================
!      DO J=1,NSTAT
!        CALL RLUXAT(LUX,IMT,K1,K2)                              ! RANLUX
!        PRINT 2,J,LUX,IMT,K1,K2                                 ! RANLUX
!    2   FORMAT(1X,'Event of SHIELD  No.',I8,';  LUX=',I4,3I12)  ! RANLUX
!        CALL GENTRE(J)                                               ! GEMCA
!	CALL OUTTOT(0,TMAX)
!	CALL NEUTLO                                                  ! GEMCA
!C
!          IF(NSAVE.le.0 .OR. NSAVE.gt.NSTAT)NSAVE=NSTAT              ! Nsave
!          J1=(J-1)/NSAVE                                             ! Nsave
!          J2=J/NSAVE                                                 ! Nsave
!          if(J2.gt.J1)then                                           ! Nsave
!            CALL SAVES(TMAX,J,NSTAT)                                         ! Nsave
!          end if                                                     ! Nsave
!C
!      END DO
!C
!      CLOSE(18)
!      CLOSE(25)
!C
!      RETURN
!      END


!! SAVES not used (A.Timofeev)
!      SUBROUTINE SAVES(TMAX,JSAVE,NSTAT)                             ! Nsave
!C Saving of files FOR024 and FOR025 after every group of NSAVE events.
!      CHARACTER*13 LABEL1,LABEL2
!      DATA LABEL1 /'END OF TITLES'/
!C
!      CHARACTER*12 LABEL3,LABEL4              ! KarSjh
!      DATA LABEL3 /'END OF DIAGN'/			! KarSjh
!C
!      COMMON /TLEPA/ TLEPA(122,20,1000),NG73,LTLEPA  ! TLEST
!      REAL*8 TLEPA                                   ! TLEST
!C
!C Closing and opening of FILE 25. Rewinding to the end of file.
!C======================= KarSjh ===========================================
!      CONTINUE!WRITE(25,1)
!    1 FORMAT('END OF DIAGN')              ! Label 'END OF DIAGN' is added
!      CLOSE(25)                           ! to the end of the file FOR025
!      OPEN(25,FILE='FOR025',STATUS='OLD') ! for correct working under the
!    2 READ(25,3)LABEL4                    ! MS Fortran Power Station 4.0.
!    3 FORMAT(A12)
!      IF(LABEL4.ne.LABEL3)THEN            ! For NDP-Fortran and g77-Fortran
!        GO TO 2                           ! under LINUX this label not need.
!      ELSE                                !                     21.04.2001
!        BACKSPACE 25
!        GOTO 8        ! TLEST
!      END IF
!C===================== end KarSjh =========================================
!
!C===================== TLEST ====================
!    8	CONTINUE
!      OPEN(29,FILE='FOR029',FORM='UNFORMATTED')
!      REWIND 29
!      WRITE(29) (((TLEPA(I,J,K), I=1,NG73), J=1,20), K=1,LTLEPA)
!      CLOSE(29)
!      GOTO 10	      
!C===================== end TLEST ================
!C
!C Opening of FILE 24. Rewinding to label "END OF TITLES".
!   10 CONTINUE
!      OPEN(24,FILE='FOR024',STATUS='OLD')
!   20 READ(24,30)LABEL2
!   30 FORMAT(A13)
!      IF(LABEL2.ne.LABEL1)GO TO 20
!C
!      OPEN(44,FILE='nmtc_yield',STATUS='OLD')     ! DCHAIN
!   21 READ(44,31)LABEL2                           ! DCHAIN
!   31 FORMAT(A13)                                 ! DCHAIN
!      IF(LABEL2.ne.LABEL1)GO TO 21                ! DCHAIN
!C
!C
!      OPEN(46,FILE='ISOTOUT',STATUS='OLD')     ! userset
!   22 READ(46,32)LABEL2                        ! userset
!   32 FORMAT(A13)                              ! userset
!      IF(LABEL2.ne.LABEL1)GO TO 22             ! userset
!C
!C PRINTING OF THE LAST TREE AND SHIELD AND LOENT TOTAL OUTPUT.
!      CALL PRINTR
!      CALL PRNTOT(TMAX)
!      CALL PRINLN
!      CONTINUE!WRITE(24,40)JSAVE,NSTAT
!      WRITE(46,40)JSAVE,NSTAT   ! userset
!   40 FORMAT(' Saving after J=',I7,' events from NSTAT=',I7,' events.')
!      CALL OUT_LUX(24)                                      ! RANLUX
!      CLOSE(24)
!      CLOSE(44)                                   ! DCHAIN
!      CLOSE(46)                                   ! userset
!C
!      RETURN
!      END



      SUBROUTINE SREACT(INOVER)
C
      COMMON /INREAC/ COST,SINF,COSF,TINT,WINT,JPART,NUCLID,KSTATE
      COMMON /INRE00/ COST0,SINF0,COSF0,TINT0,WINT0,JPAR0,NUCL0,KSTA0 ! HYDROG
      COMMON /RANDOM/ IX,IXINIT,NUMTRE,IXFIRS,LASTIX
      COMMON /DEBUG/ LSTAR,LCASC        ! 3.1.97
        COMMON /HIPROJ/ APROJ,ZPROJ                            !HION9
      COMMON /SECHAR/ SPT(6,5000),SNU(10,101),LS6,LS100,LS10,LS11  ! GSITR
      COMMON /ANTIL/ lantil                                         ! KarSjh
      COMMON /ISLERR/ ISLERR   ! HADES1
      ISLERR=0                 ! HADES1
C
C ******************* !HION9 ******************
      IF(JPART.eq.21)THEN
        APROJ=2.
        ZPROJ=1.
      END IF
      IF(JPART.eq.22)THEN
        APROJ=3.
        ZPROJ=1.
      END IF
      IF(JPART.eq.23)THEN
        APROJ=3.
        ZPROJ=2.
      END IF
      IF(JPART.eq.24)THEN
        APROJ=4.
        ZPROJ=2.
      END IF
C *************** end of !HION9 ******************
C
C ++++++++++++++++++++++ ! HYDROG ++++++++++++++++++++++++++++
c      CONTINUE!WRITE(25,56)    ! Debug printout of /INREAC/
c      write( *,56)
c   56 format('        COST        SINF        COSF        TINT',
c     *       '        WINT JPR NCL KST')
c      CONTINUE!WRITE(25,57)COST,SINF,COSF,TINT,WINT,JPART,NUCLID,KSTATE
c      write( *,57)COST,SINF,COSF,TINT,WINT,JPART,NUCLID,KSTATE
c   57 format(5E12.6,3I4)
C
C       Storing of initial /INREAC/
        COST0=COST
        SINF0=SINF
        COSF0=COSF
        TINT0=TINT
        WINT0=WINT
        JPAR0=JPART
        NUCL0=NUCLID
        KSTA0=KSTATE
C
C Interaction of HI with hydrogen-target.
      IF(NUCLID.eq.1 .AND. JPART.ge.21)THEN
C       Transformation of /INREAC/ to HI-rest system
        COST=-COST   !
        SINF=-SINF   !  Inverse direction of fly
        COSF=-COSF   !
          TINT=TINT
          WINT=WINT
        IF(JPART.eq.21)NUCLID=101    ! D
        IF(JPART.eq.22)NUCLID=102    ! T
        IF(JPART.eq.23)NUCLID=104    ! He3
        IF(JPART.eq.24)NUCLID=2      ! He4
        IF(JPART.eq.25)NUCLID=INT(ZPROJ)
C                        ! WARNING: It is necessary to use as A of the NUCLID
C                        ! not A=AMIXIS(NUCLID), but APROJ.
        lantil=1 ! KarSjh: To give into GENAGT right A of NUCLID, namely APROJ.
        JPART=2  ! KarSjh: There was error. There must be JPART=2, not JPART=1
          KSTATE=KSTATE
      ELSE	           ! KarSjh
        lantil=0         ! KarSjh
      END IF
C ++++++++++++++++++++++++++ end HYDROG ++++++++++++++++++++++++++
C
C GENERATOR BRANCHING
      IF(TINT.le.1000. .AND. (JPART.ge.1 .and. JPART.le.4)
     *                 .AND. NUCLID.gt.8
     *                 .AND. LCASC.eq.1)THEN
	  CALL CASEVP(INOVER)  ! DUBNA-70 cascade model
      ELSE
	  CALL GENAGT(INOVER)  ! AGT generator
        if(ISLERR.gt.0)return       ! HADES1
      END IF
C
C ++++++++++++++++++++++ ! HYDROG ++++++++++++++++++++++++++++
C Lorentz transformation and restoring of initial /INREAC/
C to hydrogen-rest system.
      IF(NUCL0.eq.1 .AND. JPAR0.ge.21)THEN
c          CALL KARSJH(36) ! debug printout
c          CALL STAR(1)    ! debug printout
C
        CALL SLABANT  ! Transform of SPT,SNU to hydrogen-rest system
C
C       Restoring of initial /INREAC/
        COST=COST0
        SINF=SINF0
        COSF=COSF0
        TINT=TINT0
        WINT=WINT0
        JPART=JPAR0
        NUCLID=NUCL0
        KSTATE=KSTA0
C============================ KarSjh ===============================
C Don`t forget, that Nucleus-target is H(1,1), but not (APROJ,ZPROJ).
        SNU(1,1)=1.0
        SNU(2,1)=1.0
C=========================== end KarSjh ============================
c          CALL KARSJH(36) ! debug printout
c          CALL STAR(1)    ! debug printout
      END IF
C ++++++++++++++++++++++++++ end HYDROG ++++++++++++++++++++++++++
C
      RETURN
      END



      SUBROUTINE DIMAR
C FIXING OF MAIN ARRAYS DIMENSIONS.
      COMMON /RANDOM/ IX,IXINIT,NUMTRE,IXFIRS,LASTIX
      COMMON /TREE/ TREE(15,400000),ITREE(7,400000),LTR13,LTR7,LTR300 ! GSITR
C
      COMMON /SOURN/ SOURN(13,100000),ISOURN(7,100000),LSN13,LSN7,LSN100   ! GSITR
      COMMON /SOURG/ SOURG(13,20000),ISOURG(7,20000),LSG13,LSG7,LSG100  ! GSITR
      COMMON /SEMIN/ SEMIN(13,100),ISEMIN(7,100),L1313,L137,L13100
      COMMON /SEPLS/ SEPLS(13,100),ISEPLS(7,100),L1413,L147,L14100
      COMMON /SNUEL/ SNUEL(13,100),ISNUEL(7,100),L1713,L177,L17100
      COMMON /SANUE/ SANUE(13,100),ISANUE(7,100),L1813,L187,L18100
      COMMON /SNUMU/ SNUMU(13,100),ISNUMU(7,100),L1913,L197,L19100
      COMMON /SANUM/ SANUM(13,100),ISANUM(7,100),L2013,L207,L20100
C
      COMMON /SECHAR/ SPT(6,5000),SNU(10,101),LS6,LS100,LS10,LS11  ! GSITR
      COMMON /STERAN/ STEPR(9,500),LST9,LST50
      COMMON /TRAN/ TRAN(5,600000),ITRAN(6,600000),LSR5,LSR6,LSR500  ! GSITR
      COMMON /TMINPR/ TMINUS(25),TMINDF(25),LPR30                    !HION4
      COMMON /TREVS/ TREVS(15,400),ITREVS(7,400),LVS13,LVS7,LVS100   !hion6
      COMMON /TOTOUT/ SPNHIG(28),SPROT(28),SPIM(28),SPIP(28),
     *                SANTIN(28),SANTIP(28),SKMIN(28),SKPLS(28),
     *                SK0L(28),SK0S(28),SMUM(28),SMUP(28),
     *                SOURCE(-1:28),SOGAM(28),SOEL(28),SOPOS(28),    ! BNAB28
     *                SONUEL(28),SOANUE(28),SONUMU(28),SOANUM(28),
     *                TARGIN(19,1000),LIZ50         ! 21.02.97
      COMMON /AZSPD/ AZSPD(280,100,0:1000),LZS50        ! DCHAIN
C            COMMONS OF LOENT
      COMMON /REMLN/ REMLN(11,5000),NREMLN,LRL50   ! Koshka_ADS      ! GEMCA
      COMMON /STEPLN/ STEPLN(10,500),ITEPLN(6,500),ISTEP,LNN50
      COMMON /LNHIST/ ESTIM(13,1000),SOURLN(-1:28),SPLOUT(-1:28),
     *                                      RESTRE(-1:28),LES50   ! BNAB28
      REAL*8 ESTIM   ! Loent28_DP
      COMMON /CRSSEC/ CROSEC(51,3,20,10),NUPART
      COMMON /SIGMAC/ ENET(60),SIGMAC(60,15,48),NUPAR0     ! CORNEL   !HION
      COMMON /RANMAC/ TR(100),RANMAC(100,13,48),OPTMAC(100,13,48),NUPARZ  ! CORNEL !HION
C
C-------------------- TLEST -----------------------------------------
C WARNING!
C Dimensions of the energy variable in the COMMONs /EGRIDL/,/TLEPA/
C MUST OBEY to the following conditions: NG73=NG70+3
      COMMON /TLEPA/ TLEPA(122,20,1000),NG73,LTLEPA
      REAL*8 TLEPA
      COMMON /EGRIDL/ EGRIDL(119),NG70
C-------------------- end TLEST -------------------------------------
C
      COMMON /ISOSET/ WISOSET(2000),FERTAZ(2000,50),  ! Fertman ! Truba06
     *                ISOSET(2,2000),TDEC(2000),NIS200,NISTRU   ! userset
      CHARACTER*10 TDEC   ! userset
C
      LTR13=15                                                        !HION1
      LTR7=7
      LTR300=400000   ! GSITR
	  LSN13=13
	  LSN7=7
          LSN100=100000   ! GSITR
	      LSG13=13
	      LSG7=7
              LSG100=20000  ! GSITR
		  LS6=6
		  LS100=5000   !GSITR
		  LS10=10
          LS11=101       ! GSITR
		      LST9=9
                      LST50=500
			  LSR5=5
			  LSR6=6
                          LSR500=600000  ! GSITR
                              LPR30=25                               !HION4
                                  LVS13=15                           !hion6
				  LVS7=7
                                  LVS100=400                         !hion6
                                      LIZ50=1000
					  LRL50=5000                                ! Koshka_ADS
                                              LNN50=500
                                                  LES50=1000
                                                  LZS50=1000        ! DCHAIN
      L1313=13
      L137=7
      L13100=100
	  L1413=13
	  L147=7
	  L14100=100
	      L1713=13
	      L177=7
	      L17100=100
		  L1813=13
		  L187=7
		  L18100=100
		      L1913=13
		      L197=7
		      L19100=100
			  L2013=13
			  L207=7
			  L20100=100
      NUPART=10
      NUPAR0=15                                                    !HION
      NUPARZ=13                                                    !HION
      NIS200=2000   ! userset   ! Truba06
C
C---------------- TLEST ----------
      NG73=122
      LTLEPA=1000
      NG70=119
C---------------- end TLEST ------
C
      RETURN
      END



      SUBROUTINE CUTOFF
C APPROPRTATION OF CUT OFF ENERGIES FOR EXTRANUCLEAR PARTICLES
      COMMON /TMINPR/ TMINUS(25),TMINDF(25),LPR30                    !HION4
C
      DO 10 J=1,LPR30
      TMINUS(J)=0.
      TMINDF(J)=0.
   10 CONTINUE
C
C Default CUTOFF energies
      TMINDF( 2)=3.
      TMINDF( 3)=3.
      TMINDF( 4)=3.
      TMINDF( 7)=3.
      TMINDF( 8)=3.
      TMINDF( 9)=3.
      TMINDF(10)=3.
      TMINDF(11)=3.
      TMINDF(15)=3.
      TMINDF(16)=3.
      TMINDF(21)=3.                                                  !HION4
      TMINDF(22)=3.                                                  !HION4
      TMINDF(23)=3.                                                  !HION4
      TMINDF(24)=3.                                                  !HION4
      TMINDF(25)=3.                                                  !HION4
C
C User's CUTOFF energies
      TMINUS( 2)=0.
      TMINUS( 3)=0.
      TMINUS( 4)=0.
      TMINUS( 7)=0.
      TMINUS( 8)=0.
      TMINUS( 9)=0.
      TMINUS(10)=0.
      TMINUS(11)=0.
      TMINUS(15)=0.
      TMINUS(16)=0.
      TMINUS(21)=0.                                                  !HION4
      TMINUS(22)=0.                                                  !HION4
      TMINUS(23)=0.                                                  !HION4
      TMINUS(24)=0.                                                  !HION4
      TMINUS(25)=0.                                                  !HION4
C
      DO 40 J=1,LPR30
c      IF(TMINUS(J) .LT. TMINDF(J)) TMINUS(J)=1.01*TMINDF(J)
      IF(TMINUS(J) .LT. TMINDF(J)) TMINUS(J)=TMINDF(J)
   40 CONTINUE
C
      CONTINUE!WRITE(24,50)
   50 FORMAT(/10X,'CUTOFF ENERGIES for charged particles transport.')
      CONTINUE!WRITE(24,60)
   60 FORMAT(10X,'       P     PI-     PI+  Anti P      K-      K+',
     *           '     MU-     MU+')
      CONTINUE!WRITE(24,70)TMINDF( 2),TMINDF( 3),TMINDF( 4),TMINDF( 7),
!     *            TMINDF( 8),TMINDF( 9),TMINDF(15),TMINDF(16)
   70 FORMAT('     DFLT:',8F8.1)
      CONTINUE!WRITE(24,80)TMINUS( 2),TMINUS( 3),TMINUS( 4),TMINUS( 7),
!     *            TMINUS( 8),TMINUS( 9),TMINUS(15),TMINUS(16)
   80 FORMAT('     USER:',8F8.1)
C
C ******************** !HION4 ***********************
      CONTINUE!WRITE(24,61)
   61 FORMAT(10X,'   DEUTR  TRITON     He3   Alpha HEV*ION')
      CONTINUE!WRITE(24,71)TMINDF(21),TMINDF(22),TMINDF(23),TMINDF(24),TMINDF(25)
   71 FORMAT('     DFLT:',5F8.1)
      CONTINUE!WRITE(24,81)TMINUS(21),TMINUS(22),TMINUS(23),TMINUS(24),TMINUS(25)
   81 FORMAT('     USER:',5F8.1)
C ******************** end of !HION4 *****************
C
C     RETURN
      END



      SUBROUTINE SLABANT
C Transformation of SPT, SNU  after hadron-nucleus reaction
C (nucleus-target is at rest in Lab. system) into Antilab. system.
C
C       SECONDARY PARTICLES CHARACTERISTICS IN ARRAY SPT
C           SPT(1,J) - COS OF POLAR ANGLE OF SECONDARY PARTICLE
C           SPT(2,J) -  SIN OF AZIMUTAL ANGLE
C           SPT(3,J) -  COS OF AZIMUTAL ANGLE
C           SPT(4,J) - KINETIC ENERGY OF SECONDARY PARTICLE (MEV)
C           SPT(5,J) - SECONDARY PARTICLE TYPE
C           SPT(6,J) - SECONDARY PARTICLE WEIGHT
C       NUCLEUS CHARACTERISTICS
C         Nucleus-target
C           SNU(1,1) - NUCLEUS-TARGET ATOMIC WEIGHT
C           SNU(2,1) - NUCLEUS-TARGET ATOMIC NUMBER
C         Nuclei-products after deexitation, J=2-11
C           SNU(1,J) - Aprod
C           SNU(2,J)   Zprod
C           SNU(3,J) - Excitation energy (MeV)
C           SNU(5,J)
C           SNU(6,J)   -  Nucleus momentum (MeV/C)
C           SNU(7,J)
C-------------------------------------------------------------------------
      COMMON /SECHAR/ SPT(6,5000),SNU(10,101),LS6,LS100,LS10,LS11  ! GSITR
      COMMON /INREAC/ COST,SINF,COSF,TINT,WINT,JPART,NUCLID,KSTATE
C
      DIMENSION HADMAS(24)  ! Particle's mass (MeV)
      DATA HADMAS /940., 940., 140., 140., 140., 940., 940.,
     *             495., 495., 495., 495.,   0., 0.511, 0.511,
     *             106., 106., 4*0., 1880., 2820., 2820., 3760./
C
C 4-momentum of incident hadron in Lab. system:
      RM0=HADMAS(JPART) ! hadron mass
      E0=TINT+RM0       ! total energy
c       P0X=0.                         ! KarSjh: These formulas for momentum
c       P0Y=0.                         ! are valid only for HADGEN
c       P0Z=SQRT(TINT*(TINT+2.0*RM0))  ! (when projectile moves along Z-axis).
C================ KarSjh ============
      P0MOD=SQRT(TINT*(TINT+2.0*RM0))
	IF(ABS(COST).ge.1.0)THEN
        SINT=0.
      ELSE
        SINT=SQRT(1.0-COST**2)
      END IF
      P0X=P0MOD*SINT*COSF
      P0Y=P0MOD*SINT*SINF
      P0Z=P0MOD*COST
C============== end KarSjh ==========
C
C -------------TRANSFORMATION LOOP for SPT----------------
      DO J=1,LS100
	  IPART=IFIX(SPT(5,J))     ! Type of secondary particle.
	  IF(IPART.EQ.0)GO TO 10   ! No more particles in SPT.
C
C     4-momentum of secondary particle No.J  in Lab. system:
	  CT=SPT(1,J)
	  SF=SPT(2,J)
	  CF=SPT(3,J)
	  T1=SPT(4,J)
	  RM1=HADMAS(IPART)
      CALL MOMSFE(CT,SF,CF,T1,RM1, E1,P1LAB,P1X,P1Y,P1Z)
C
C     4-momentum of secondary particle No.J in AntiLab. system:
      CALL LORENZ(E0,P0X,P0Y,P0Z,  E1,P1X,P1Y,P1Z,
     *            E1al,P1Xal,P1Yal,P1Zal)
c      WRITE(36,178)J,E0,P0X,P0Y,P0Z,
c     *               E1,P1X,P1Y,P1Z,
c     *               E1al,P1Xal,P1Yal,P1Zal
c  178 format(/' SPT:',I3,4E11.4/8X,4E11.4/8X,4E11.4)
c      CALL PRILOR(E0,P0X,P0Y,P0Z)
c      CALL PRILOR(E1,P1X,P1Y,P1Z)
c      CALL PRILOR(E1al,P1Xal,P1Yal,P1Zal)
C
C     Inversion of AntiLab system (to fly in positive direction of Z axis)
C      P1Xal=-P1Xal  ! KarSjh  ! This is excessive inversion, 02.05.01
C      P1Yal=-P1Yal  ! KarSjh
C      P1Zal=-P1Zal  ! KarSjh
C
C     Spherical angles of secondary particle No.J in AntiLab. system:
      CALL DECARS(P1Xal,P1Yal,P1Zal,P1ALAB,COST1,SINF1,COSF1)
C
C     Appropriations into SPT in Antilab system
	  SPT(1,J)=COST1
	  SPT(2,J)=SINF1
	  SPT(3,J)=COSF1
	  SPT(4,J)=E1al-RM1
C
      END DO
C
C -------------TRANSFORMATION LOOP for SNU----------------
   10 CONTINUE
C
      DO 20 J=2,LS11
	  IF(SNU(1,J).EQ.0)GO TO 20
C
C 4-momentum of nucleus No.J in Lab. system
	  RMN1=940.0*SNU(1,J)
	  EN1=SQRT((SNU(5,J)**2+SNU(6,J)**2+SNU(7,J)**2)+RMN1**2)
	  PN1X=SNU(5,J)
	  PN1Y=SNU(6,J)
	  PN1Z=SNU(7,J)
C
C     4-momentum of nucleus No.J in AntiLab. system:
      CALL LORENZ(E0,P0X,P0Y,P0Z,  EN1,PN1X,PN1Y,PN1Z,
     *            EN1al,PN1Xal,PN1Yal,PN1Zal)
c      WRITE(36,179)J,E0,P0X,P0Y,P0Z,
c     *               EN1,PN1X,PN1Y,PN1Z,
c     *               EN1al,PN1Xal,PN1Yal,PN1Zal
c  179 format(/' SNU:',I3,4E11.4/8X,4E11.4/8X,4E11.4)
c      CALL PRILOR(E0,P0X,P0Y,P0Z)
c      CALL PRILOR(EN1,PN1X,PN1Y,PN1Z)
c      CALL PRILOR(EN1al,PN1Xal,PN1Yal,PN1Zal)
C
C     Inversion of AntiLab system (to fly in positive direction of Z axis)
C      PN1Xal=-PN1Xal  ! KarSjh  ! This is excessive inversion, 02.05.01
C      PN1Yal=-PN1Yal  ! KarSjh
C      PN1Zal=-PN1Zal  ! KarSjh
C
C     Appropriations into SNU in Antilab system
	  SNU(5,J)=PN1Xal
	  SNU(6,J)=PN1Yal
	  SNU(7,J)=PN1Zal
C
   20 CONTINUE
C
      RETURN
      END



      SUBROUTINE MOMSFE(COST,SINF,COSF,TKIN,RMASS,ETOT,PMOD,PX,PY,PZ)
C Calculation of 4-momentum of particle from its kinetic energy and
C spheric angular direction.
C
C INPUT: COST - Cos(Tetha)
C        SINF - Sin(Phi)
C        COSF - Cos(Phi)
C        TKIN - Kinetic energy (MeV)
C        RMASS - Mass (MeV)
C
C OUTPUT: ETOT - Total energy (MeV)
C         PMOD - Modulus of 3-momentum (MeV/c)
C         PX, PY, PZ - Cartesius components of 3-momentum (MeV/c)
C
      REAL*8 DCT,DSF,DCF,DST,DTKIN,DRMASS,DETOT,DPMOD,DPX,DPY,DPZ
C
      DCT=DBLE(COST)
      DSF=DBLE(SINF)
      DCF=DBLE(COSF)
      DST=DSQRT(1.0-DCT*DCT)     ! Sin(Tetha)
	  DTKIN=DBLE(TKIN)
	  DRMASS=DBLE(RMASS)
	  DETOT=DTKIN+DRMASS                      ! Etot
	  DPMOD=DSQRT(DTKIN*(DTKIN+2.0*DRMASS))   ! Pmod
	      DPX=DPMOD*DST*DCF                   ! Px
	      DPY=DPMOD*DST*DSF                   ! Py
	      DPZ=DPMOD*DCT                       ! Pz
C
      ETOT=SNGL(DETOT)
      PMOD=SNGL(DPMOD)
      PX=SNGL(DPX)
      PY=SNGL(DPY)
      PZ=SNGL(DPZ)
C
      RETURN
      END



      SUBROUTINE LORENZ(E0,P0X,P0Y,P0Z,  E1,P1X,P1Y,P1Z,
     *                                   E1str,P1Xstr,P1Ystr,P1Zstr)
C Lorenz transformation.
C Let in some reference system (Lab. system) particles 0 and 1
C have 4-momenta  (E0,P0X,P0Y,P0Z)  and  (E1,P1X,P1Y,P1Z).
C Then in reference system where particle 0 is in rest (Antilab. syst.),
C particle 1 has 4-momentum E1str,P1Xstr,P1Ystr,P1Zstr.
C      [see. Kopylov, page 23, eq. (26), (27)]
C
C INPUT: E0,P0X,P0Y,P0Z - 4-momentum of incident particle in Lab.system
C        E1,P1X,P1Y,P1Z - 4-momentum of secondary particle in Lab.system
C
C OUTPUT: E1str,P1Xstr,P1Ystr,P1Zstr - 4-momentum of secondary particle
C                                      in AntiLab.system
C
      REAL*8 DE0,DP0X,DP0Y,DP0Z,DE1,DP1X,DP1Y,DP1Z,DRM0,DTMP,
     *       DE1s,DP1Xs,DP1Ys,DP1Zs
C
      DE0 =DBLE(E0)
      DP0X=DBLE(P0X)
      DP0Y=DBLE(P0Y)
      DP0Z=DBLE(P0Z)
      DE1 =DBLE(E1)
      DP1X=DBLE(P1X)
      DP1Y=DBLE(P1Y)
      DP1Z=DBLE(P1Z)
C
      DRM0=DSQRT(DE0*DE0-(P0X*P0X+P0Y*P0Y+P0Z*P0Z)) ! RM0 mass of inc. part.
      DE1s=(DE0*DE1-(P0X*P1X+P0Y*P1Y+P0Z*P1Z))/DRM0 ! E1star
	  DTMP=(DE1+DE1s)/(DE0+DRM0)
      DP1Xs=DP1X-DP0X*DTMP                          ! P1Xstar
      DP1Ys=DP1Y-DP0Y*DTMP                          ! P1Ystar
      DP1Zs=DP1Z-DP0Z*DTMP                          ! P1Zstar
C
      E1str =SNGL(DE1s)
      P1Xstr=SNGL(DP1Xs)
      P1Ystr=SNGL(DP1Ys)
      P1Zstr=SNGL(DP1Zs)
C
      RETURN
      END



      subroutine PRILOR(E,PX,PY,PZ)
C Debug printout of 4-momentum
      CONTINUE!WRITE(36,1)E,PX,PY,PZ
    1 format(' E, Px, Py, Pz: ',4E12.5)
C
      P2=PX**2+PY**2+PZ**2
      P=SQRT(P2)
      RM=SQRT(E**2-P2)
      T=E-RM
C
      CONTINUE!WRITE(36,2)P2,P,RM,T
    2 format(' P2, P, RM, T : ',4E12.5)
C
      RETURN
      END



      SUBROUTINE INLEVR
C This program provides an initialization of values of some parameters
C for the MSDM generator. By varying of these values (in COMMON /LEVERS/)
C one can affect the results of simulation.
C
C -------------------PARAMETERS of the AGT generator-----------------
C COMMONs of AGT-generator which contain the parameters (all are REAL*8):
C      COMMON /PIDABS/ PIDABS
C      COMMON /TLIMIT/ TLIMIT
C      COMMON /MMATUR/ MMES,MBAR
C      COMMON /HCASC/ ANUCL1,ANUCL2,ZNUCL1,ZNUCL2,T0,EPS1,EPS2,VPI,A1,A2,
C     *               C1,C2,D1,D2,R0N1,R0N2,TF01,TF02,RM1,RM2
C
C STATEMENTs in GENAGT which set the parameter values
C General AGT parameters
C      PIDABS=0.01      ! absorbtion via pi-D !
C      TLIMIT=25.       ! total interaction time in fm/c !
C      MMES=0.2         ! for T0<1 GeV one may use MMES=999.9 (?)!
C      MBAR=999.999
C
C Parameters of nuclear density distribution (A>10)
C     EPS2=0.007      ! average (for product far from target) !
C     VPI=0.025
C     D2=0.05
C     C2=0.545    ! for A>10 ! ! for A<10 don't used !
C     R0N2=1.07   ! for A>10 !
C
C Parameter R0N for light nuclei (A<=10)
C     IF(LA.EQ.2 .AND. LZ.EQ.1)R0N2=2.31
C     IF(LA.EQ.3 .AND. LZ.EQ.1)R0N2=2.4
C     IF(LA.EQ.3 .AND. LZ.EQ.2)R0N2=2.4
C     IF(LA.EQ.4 .AND. LZ.EQ.2)R0N2=1.60
C     IF(LA.EQ.6 .AND. LZ.EQ.3)R0N2=2.4
C     IF(LA.EQ.7 .AND. LZ.EQ.3)R0N2=2.4
C     IF(LA.EQ.9 .AND. LZ.EQ.4)R0N2=2.48
C     IF(LA.EQ.10 .AND. LZ.EQ.5)R0N2=2.48
C              Absent in GENAGT!  For enhancement in future.
C     IF(LA.EQ.11 .AND. LZ.EQ.5)R0N2=2.48 !
C     IF((LA.EQ.12 .OR. LA.EQ.13) .AND. LZ.EQ.6)R0N2=2.48 !
C     IF((LA.EQ.14 .OR. LA.EQ.15) .AND. LZ.EQ.7)R0N2=2.48 !
C     IF((LA.EQ.16 .OR. LA.EQ.17 .OR. LA.EQ.18) .AND. LZ.EQ.8)R0N2=2.48 !
C
C -------------------PARAMETERS of the PRECOmpound emission----------
C COMMON of PRECO which contain the parameter:
C     COMMON /PRECOM/ PRECOM  ! (0<PRECOM<1)
C STATEMENT in PRECO which set the parameter value
C     PRECOM=1.
C     NSP=NINT(SQRT((1.19*AM*A*U+0.5)*PRECOM))
C
C -------------------PARAMETERS of DEEXcitation----------------------
* Parameters of DEEX.F which are defined by BLOCK DATA.
* Parameters which are labelled by [*] will be overwritten by PARLEV
* -----------------------------------------------------------------
* BLOCK DATA DEFALT
* COMMON /BLMULF/ IMULF 
* COMMON /BLIMA/  IMA [*]
* COMMON /BLPARA/ EPSIL0 [*], FKAPPA, FKACOL [*]
* COMMON /BPLACE/ RFI
* COMMON /ADDNUC/ IZPN,IAPN
* COMMON /RESNU0/ EMIN,EMAX,YMIN,YMAX - not used anywhere 
*  DATA IMULF  /1/   - IMULF is changed inside Subroutine DEEX during simulation 
*  DATA IMA    /3/   - IMA is overwritten by PARLEV [*]
*  DATA EPSIL0 /16./ - EPSIL0 is overwritten by PARLEV [*]
*  DATA FKAPPA /1./  - FKAPPA is used in ZVEZD1
*  DATA FKACOL /2./  - FKACOL is overwritten by PARLEV [*]
*   DATA IZPN   /0/  - IZPN is used in ZVEZD1
*   DATA IAPN   /0/  - IAPN is used in ZVEZD1
*   DATA RFI    /0./ - RFI is used in CULIMP
*   DATA EMIN   /0./ - not used anywhere
*   DATA EMAX   /0./ - not used anywhere
*   DATA YMIN   /0./ - not used anywhere
*   DATA YMAX   /0./ - not used anywhere
* 
* BLOCK DATA DMAS02
* COMMON /BL0999/ RNCL [*]
* COMMON /BL1100/ AMS [*], AMFS [*]
* COMMON /BLCAN/  ICAN [*]
* COMMON /BLSEC/  ISEC
* COMMON ......
* DATA RNCL /0.0/   - RNCL is overwritten by PARLEV [*]
* DATA AMS  /0.125/ - AMS is overwritten by PARLEV [*]
* DATA AMFS /0.125/ - AMFS is overwritten by PARLEV [*]
* DATA ICAN /1/     - ICAN is overwritten by PARLEV [*]
* DATA ISEC /1/     - ISEC is used in GAMMAC, TKIN
* DATA ........
* 
* 
* Parameters of DEEX.F which are taken from COMMON /LEVERS/ PARLEV(40)
* and overwrite [*] the BLOCK DATA DEFALT 
* -------------------------------------------------------------------
*                   ! No.  COMMON    Parameter     DEFAULT in DFLT(40)
* PARLEV(23)=0.     ! 23  /BLCAN/    ICAN [*]              0
* PARLEV(24)=3.     ! 24  /BLIMA/    IMA [*]               3
* PARLEV(25)=16.    ! 25  /BLPARA/   EPSIL0 [*]           16.
* PARLEV(26)=2.     ! 26  /BLPARA/   FKACOL [*]            2.
* PARLEV(27)=1.3    ! 27  /BL0999/   RNCL [*]              0.
* PARLEV(28)=0.1    ! 28  /BL1100/   AMS [*]           0.125
* PARLEV(29)=0.1    ! 29  /BL1100/   AMFS [*]          0.125
* PARLEV(30)=16.    ! 30             Amax for Fermi       16.
* PARLEV(31)=0.002  ! 31             Min Umulf(GeV/A)  0.002
* PARLEV(32)=2.     ! 32  /ILEVRA/   ILEVRA                2   ! DKFZg
* PARLEV(33)=1.     ! 33  /FKAP/     FKAP1 - Coulomb       1.  ! DKFZg
* PARLEV(34)=1.     ! 34  /FKAP/     FKAP2 - Volume        1.  ! DKFZg
C===================================================================
C
      COMMON /LEVERS/ PARLEV(40)  ! Values of the parameters
      DIMENSION DFLT(40)          ! Default values of the parameters
C
      DATA DFLT /
     *0.01,     25.,      0.2,      999.999,  0.007, !  5
     *0.025,    0.545,    0.05,     1.07,     2.31,  ! 10
     *2.4,      2.4,      1.60,     2.4,      2.4,   ! 15
     *2.48,     2.48,     2.48,     2.48,     2.48,  ! 20
     *2.48,     1.,       0.,       3.,       16.,   ! 25
     *2.,       0.,       0.125,    0.125,    16.,   ! 30
     *0.002,    2.,       1.0,      1.0,      0.,    ! 35   ! DKFZg
     *0.,       0.,       0.,       0.75,     1.0/   ! 40   ! DKFZf
C
      DO J=1,40
	PARLEV(J)=DFLT(J)
      END DO
C
C Appropriation of the parameter values
C                        ! No.  COMMON    Parameter         DEFAULT
      PARLEV( 1)=0.01    !  1  /PIDABS/   PIDABS            0.01
      PARLEV( 2)=25.     !  2  /TLIMIT/   TLIMIT            25.
      PARLEV( 3)=999.999 !  3  /MMATUR/   MMES              0.2
      PARLEV( 4)=999.999 !  4  /MMATUR/   MBAR              999.999
      PARLEV( 5)=0.007   !  5  /HCASC/    EPS1,EPS2         0.007
      PARLEV( 6)=0.025   !  6  /HCASC/    VPI               0.025
      PARLEV( 7)=0.545   !  7  /HCASC/    C1,C2             0.545
      PARLEV( 8)=0.05    !  8  /HCASC/    D1,D2             0.05
      PARLEV( 9)=1.07    !  9  /HCASC/    R0N1,R0N2         1.07
      PARLEV(10)=2.31    ! 10             R0N for H(2,1)    2.31
      PARLEV(11)=2.4     ! 11             R0N for H(3,1)    2.4
      PARLEV(12)=2.4     ! 12             R0N for He(3,2)   2.4
      PARLEV(13)=1.60    ! 13             R0N for He(4,2)   1.60
      PARLEV(14)=2.4     ! 14             R0N for Li(6,3)   2.4
      PARLEV(15)=2.4     ! 15             R0N for Li(7,3)   2.4
      PARLEV(16)=2.48    ! 16             R0N for Be(4,9)   2.48
      PARLEV(17)=2.48    ! 17             R0N for B(10,5)   2.48
c      PARLEV(18)=2.48   ! 18             R0N for B(11,5)   2.48 ! Absent
c      PARLEV(19)=2.48   ! 19             R0N for C(Nat,6)  2.48 ! in GENAGT
c      PARLEV(20)=2.48   ! 20             R0N for N(Nat,7)  2.48 !
c      PARLEV(21)=2.48   ! 21             R0N for O(Nat,8)  2.48 !
C
      PARLEV(22)=1.      ! 22  /PRECOM/   PRECOM            1.
C
      PARLEV(23)=0.      ! 23  /BLCAN/    ICAN              0
      PARLEV(24)=3.      ! 24  /BLIMA/    IMA               3
      PARLEV(25)=16.     ! 25  /BLPARA/   EPSIL0            16.
      PARLEV(26)=2.      ! 26  /BLPARA/   FKACOL            2.
      PARLEV(27)=1.3     ! 27  /BL0999/   RNCL              0.
      PARLEV(28)=0.1     ! 28  /BL1100/   AMS               0.125
      PARLEV(29)=0.1     ! 29  /BL1100/   AMFS              0.125
      PARLEV(30)=16.     ! 30             Amax for Fermi    16.
      PARLEV(31)=0.002   ! 31             Min of Umulf/A    0.002 GeV/A
      PARLEV(32)=2.      ! 32  /ILEVRA/   ILEVRA            2     ! DKFZg
      PARLEV(33)=1.      ! 33  /FKAP/     FKAP1 - Coulomb   1.    ! DKFZg
      PARLEV(34)=1.      ! 34  /FKAP/     FKAP2 - Volume    1.    ! DKFZg
C
      PARLEV(39)=1.0     ! 39             Renorm AA-SIGION  0.75  ! DKFZf
      PARLEV(40)=1.0     ! 40             Renorm hA-MICROD  1.0   ! DKFZf
C
C Printout of the parameter values (if differ from default value only!)
c      CONTINUE!WRITE(24,100)
c  100 FORMAT(/10X,'PARAMETERS of the SHIELD hA- and AA-GENERATOR'/
c     *        10X,'HAVE BEEN ESTABLISHED. The following values'/
c     *        10X,'differ from default ones:')
C
c      IF(PARLEV( 1) .ne. DFLT( 1)) CONTINUE!WRITE(24, 1) 1,PARLEV( 1)
c      IF(PARLEV( 2) .ne. DFLT( 2)) CONTINUE!WRITE(24, 2) 2,PARLEV( 2)
c      IF(PARLEV( 3) .ne. DFLT( 3)) CONTINUE!WRITE(24, 3) 3,PARLEV( 3)
c      IF(PARLEV( 4) .ne. DFLT( 4)) CONTINUE!WRITE(24, 4) 4,PARLEV( 4)
c      IF(PARLEV( 5) .ne. DFLT( 5)) CONTINUE!WRITE(24, 5) 5,PARLEV( 5)
c      IF(PARLEV( 6) .ne. DFLT( 6)) CONTINUE!WRITE(24, 6) 6,PARLEV( 6)
c      IF(PARLEV( 7) .ne. DFLT( 7)) CONTINUE!WRITE(24, 7) 7,PARLEV( 7)
c      IF(PARLEV( 8) .ne. DFLT( 8)) CONTINUE!WRITE(24, 8) 8,PARLEV( 8)
c      IF(PARLEV( 9) .ne. DFLT( 9)) CONTINUE!WRITE(24, 9) 9,PARLEV( 9)
c      IF(PARLEV(10) .ne. DFLT(10)) CONTINUE!WRITE(24,10)10,PARLEV(10)
c      IF(PARLEV(11) .ne. DFLT(11)) CONTINUE!WRITE(24,11)11,PARLEV(11)
c      IF(PARLEV(12) .ne. DFLT(12)) CONTINUE!WRITE(24,12)12,PARLEV(12)
c      IF(PARLEV(13) .ne. DFLT(13)) CONTINUE!WRITE(24,13)13,PARLEV(13)
c      IF(PARLEV(14) .ne. DFLT(14)) CONTINUE!WRITE(24,14)14,PARLEV(14)
c      IF(PARLEV(15) .ne. DFLT(15)) CONTINUE!WRITE(24,15)15,PARLEV(15)
c      IF(PARLEV(16) .ne. DFLT(16)) CONTINUE!WRITE(24,16)16,PARLEV(16)
c      IF(PARLEV(17) .ne. DFLT(17)) CONTINUE!WRITE(24,17)17,PARLEV(17)
c      IF(PARLEV(18) .ne. DFLT(18)) CONTINUE!WRITE(24,18)18,PARLEV(18)
c      IF(PARLEV(19) .ne. DFLT(19)) CONTINUE!WRITE(24,19)19,PARLEV(19)
c      IF(PARLEV(20) .ne. DFLT(20)) CONTINUE!WRITE(24,20)20,PARLEV(20)
c      IF(PARLEV(21) .ne. DFLT(21)) CONTINUE!WRITE(24,21)21,PARLEV(21)
C
c      IF(PARLEV(22) .ne. DFLT(22)) CONTINUE!WRITE(24,22)22,PARLEV(22)
C
c      IF(PARLEV(23) .ne. DFLT(23)) CONTINUE!WRITE(24,23)23,INT(PARLEV(23))
c      IF(PARLEV(24) .ne. DFLT(24)) CONTINUE!WRITE(24,24)24,INT(PARLEV(24))
c      IF(PARLEV(25) .ne. DFLT(25)) CONTINUE!WRITE(24,25)25,PARLEV(25)
c      IF(PARLEV(26) .ne. DFLT(26)) CONTINUE!WRITE(24,26)26,PARLEV(26)
c      IF(PARLEV(27) .ne. DFLT(27)) CONTINUE!WRITE(24,27)27,PARLEV(27)
c      IF(PARLEV(28) .ne. DFLT(28)) CONTINUE!WRITE(24,28)28,PARLEV(28)
c      IF(PARLEV(29) .ne. DFLT(29)) CONTINUE!WRITE(24,29)29,PARLEV(29)
c      IF(PARLEV(30) .ne. DFLT(30)) CONTINUE!WRITE(24,30)30,PARLEV(30)
c      IF(PARLEV(31) .ne. DFLT(31)) CONTINUE!WRITE(24,31)31,PARLEV(31)
c      IF(PARLEV(32) .ne. DFLT(32)) CONTINUE!WRITE(24,32)32,INT(PARLEV(32))   ! DKFZg
c      IF(PARLEV(33) .ne. DFLT(33)) CONTINUE!WRITE(24,33)33,PARLEV(33)   ! DKFZg
c      IF(PARLEV(34) .ne. DFLT(34)) CONTINUE!WRITE(24,34)34,PARLEV(34)   ! DKFZg
Cc
c      IF(PARLEV(39) .ne. DFLT(39)) CONTINUE!WRITE(24,39)39,PARLEV(39)   ! DKFZf
c      IF(PARLEV(40) .ne. DFLT(40)) CONTINUE!WRITE(24,40)40,PARLEV(40)   ! DKFZf
Cc
c    1 FORMAT(6X,I2,'. ','AGT parameter PIDABS:          ',F10.5)
c    2 FORMAT(6X,I2,'. ','AGT parameter TLIMIT:          ',F10.5)
c    3 FORMAT(6X,I2,'. ','AGT parameter MMES:            ',F10.5)
c    4 FORMAT(6X,I2,'. ','AGT parameter MBAR:            ',F10.5)
c    5 FORMAT(6X,I2,'. ','AGT parameter EPS1,EPS2:       ',F10.5)
c    6 FORMAT(6X,I2,'. ','AGT parameter VPI:             ',F10.5)
c    7 FORMAT(6X,I2,'. ','AGT parameter C1,C2:           ',F10.5)
cn    8 FORMAT(6X,I2,'. ','AGT parameter D1,D2:           ',F10.5)
c    9 FORMAT(6X,I2,'. ','AGT parameter R0N1,R0N2:       ',F10.5)
c   10 FORMAT(6X,I2,'. ','AGT parameter R0N for H(2,1):  ',F10.5)
c   11 FORMAT(6X,I2,'. ','AGT parameter R0N for H(3,1):  ',F10.5)
c   12 FORMAT(6X,I2,'. ','AGT parameter R0N for He(3,2): ',F10.5)
c   13 FORMAT(6X,I2,'. ','AGT parameter R0N for He(4,2): ',F10.5)
c   14 FORMAT(6X,I2,'. ','AGT parameter R0N for Li(6,3): ',F10.5)
c   15 FORMAT(6X,I2,'. ','AGT parameter R0N for Li(7,3): ',F10.5)
c   16 FORMAT(6X,I2,'. ','AGT parameter R0N for Be(4,9): ',F10.5)
c   17 FORMAT(6X,I2,'. ','AGT parameter R0N for B(10,5): ',F10.5)
c   18 FORMAT(6X,I2,'. ','AGT parameter R0N for B(11,5): ',F10.5)
c   19 FORMAT(6X,I2,'. ','AGT parameter R0N for C(Nat,6):',F10.5)
c   20 FORMAT(6X,I2,'. ','AGT parameter R0N for N(Nat,7):',F10.5)
c   21 FORMAT(6X,I2,'. ','AGT parameter R0N for O(Nat,8):',F10.5)
C
c   22 FORMAT(6X,I2,'. ','PRECO parameter PRECOM:        ',F10.5)
cC
c   23 FORMAT(6X,I2,'. ','DEEX parameter ICAN:           ',I10)
c   24 FORMAT(6X,I2,'. ','DEEX parameter IMA:            ',I10)
c   25 FORMAT(6X,I2,'. ','DEEX parameter EPSIL0:         ',F10.5)
c   26 FORMAT(6X,I2,'. ','DEEX parameter FKACOL:         ',F10.5)
c   27 FORMAT(6X,I2,'. ','DEEX parameter RNCL:           ',F10.5)
c   28 FORMAT(6X,I2,'. ','DEEX parameter AMS:            ',F10.5)
c   29 FORMAT(6X,I2,'. ','DEEX parameter AMFS:           ',F10.5)
c   30 FORMAT(6X,I2,'. ','DEEX parameter Amax for Fermi: ',F10.5)
c   31 FORMAT(6X,I2,'. ','DEEX parameter Min(Umulf/A):   ',F10.5)
c   32 FORMAT(6X,I2,'. ','DEEX parameter ILEVRA:         ',I10)     ! DKFZg
c   33 FORMAT(6X,I2,'. ','DEEX parameter FKAP1-Coulomb:  ',F10.5)   ! DKFZg
c   34 FORMAT(6X,I2,'. ','DEEX parameter FKAP2-Volume:   ',F10.5)   ! DKFZg
C
c   39 FORMAT(6X,I2,'. ','SIGION AA-Renorm parameter:    ',F10.5)   ! DKFZf
c   40 FORMAT(6X,I2,'. ','MICROD hA-Renorm parameter:    ',F10.5)   ! DKFZf
C
      RETURN
      END
