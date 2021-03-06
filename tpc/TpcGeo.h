//-----------------------------------------------------------
// File and Version Information:
// $Id$
//
// Description:
//      Tpc geometry module
//
//
// Environment:
//      Software developed for the PANDA Detector at FAIR.
//
// Author List:
//      Sebastian Neubert    TUM            (original author)
//
//
//-----------------------------------------------------------

#ifndef TPCGEO_HH
#define TPCGEO_HH

// Base Class Headers ----------------
#include "FairGeoSet.h"

// Collaborating Class Headers -------

// Collaborating Class Declarations --



class TpcGeo : public FairGeoSet {
public:

  // Constructors/Destructors ---------
  TpcGeo();
  ~TpcGeo(){}

  // Operators
  

  // Accessors -----------------------
  const char* getModuleName(Int_t);
  const char* getEleName(Int_t);
  //inline Int_t getModNumInMod(const TString&);


  // Modifiers -----------------------


  // Operations ----------------------
protected:
  char modName[22];  // name of module
  char eleName[20];  // substring for elements in module
private:

  // Private Data Members ------------


  // Private Methods -----------------

public:
  ClassDef(TpcGeo,1)

};

#endif

//--------------------------------------------------------------
// $Log$
//--------------------------------------------------------------
