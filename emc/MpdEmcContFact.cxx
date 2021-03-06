#include "MpdEmcContFact.h"

#include "MpdEmcGeoPar.h"

#include "FairRuntimeDb.h"

#include <iostream>

ClassImp(MpdEmcContFact)

static MpdEmcContFact gMpdEmcContFact;

MpdEmcContFact::MpdEmcContFact() {
  /** Constructor (called when the library is loaded) */
  fName="MpdEmcContFact";
  fTitle="Factory for parameter containers in libTutorial1";
  setAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void MpdEmcContFact::setAllContainers() {
  /** Creates the Container objects with all accepted 
      contexts and adds them to
      the list of containers for the Tutorial1 library.
  */

  FairContainer* p= new FairContainer("MpdEmcGeoPar",
				    "MpdEmc Geometry Parameters",
				    "TestDefaultContext");
  p->addContext("TestNonDefaultContext");
  
  containers->Add(p);
}

FairParSet* MpdEmcContFact::createContainer(FairContainer* c) {
  /** Calls the constructor of the corresponding parameter container.
      For an actual context, which is not an empty string and not 
      the default context
      of this container, the name is concatinated with the context. 
  */
  const char* name=c->GetName();
  FairParSet* p=NULL;
  if (strcmp(name,"MpdEmcGeoPar")==0) {
    p=new MpdEmcGeoPar(c->getConcatName().Data(),
			       c->GetTitle(),c->getContext());
  }
  return p;
}
