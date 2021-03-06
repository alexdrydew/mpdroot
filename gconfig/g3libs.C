/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#include <iostream>

Bool_t isLibrary(const char *libName)
{
  return gSystem->DynamicPathName(libName, kTRUE) != nullptr;
}

void g3libs()
{
   cout << "Loading Geant3 libraries ..." << endl;

   if (isLibrary("libdummies.so")) gSystem->Load("libdummies.so");
   // libdummies.so needed from geant3_+vmc version 0.5

   if (isLibrary("libpythia6.so"))
      gSystem->Load("libpythia6.so");
   else if (isLibrary("libPythia6.so")) // Old FairSoft
      gSystem->Load("libPythia6.so");

   gSystem->Load("libEGPythia6.so");
   gSystem->Load("libgeant321.so");

   cout << "Loading Geant3 libraries ... finished" << endl;
}
