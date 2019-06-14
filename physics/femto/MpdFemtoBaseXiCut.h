/**
 * \class MpdFemtoBaseXiCut
 * \brief Base class for Xi cuts
 *
 * The pure virtual base class for the Xi cut. All Xi cuts must
 * inherit from this one.
 */

#ifndef MpdFemtoBaseXiCut_h
#define MpdFemtoBaseXiCut_h

// MpdFemtoMaker headers
// Infrastructure
#include "MpdFemtoTypes.h"
#include "MpdFemtoXi.h"
// Base
#include "MpdFemtoBaseParticleCut.h"

//_________________
class MpdFemtoBaseXiCut : public MpdFemtoBaseParticleCut {

 public:
  /// Default constructor
  MpdFemtoBaseXiCut() { /* empty */}
  /// Copy constructor
  MpdFemtoBaseXiCut(const MpdFemtoBaseXiCut& copy);
  /// Assignment operator
  MpdFemtoBaseXiCut& operator=(const MpdFemtoBaseXiCut& copy);
  /// Default destructor
  virtual ~MpdFemtoBaseXiCut() { /* empty */ }

  /// Returns true if cut has been passed and false if not
  virtual bool pass(const MpdFemtoXi* ) = 0;

  /// Return Xi type
  virtual MpdFemtoParticleType type() { return hbtXi; }

  /// Clone Xi cut
  virtual MpdFemtoBaseXiCut* clone() { return nullptr; }

#ifdef __ROOT__
  ClassDef(MpdFemtoBaseXiCut, 0)
#endif
};

//_________________
inline MpdFemtoBaseXiCut::MpdFemtoBaseXiCut(const MpdFemtoBaseXiCut& c) : MpdFemtoBaseParticleCut(c) { /* no-op */ }

//_________________
inline MpdFemtoBaseXiCut& MpdFemtoBaseXiCut::operator=(const MpdFemtoBaseXiCut& c) {
  if (this != &c ) {
    MpdFemtoBaseParticleCut::operator=(c);
  }
  return *this;
}

#endif // #define MpdFemtoXiCut_h