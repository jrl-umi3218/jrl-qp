/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <vector>

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

namespace jrlqp::internal
{
  class JRLQP_DLLAPI ActiveSet
  {
  public:
    ActiveSet();
    ActiveSet(int nCstr, int nBnd = 0);
    ActiveSet(const std::vector<ActivationStatus>& as, int nBnd);

    void resize(int nCstr, int nBnd = 0);
    void reset();

    bool isActive(int cstrIdx) const;
    bool isActiveBnd(int bndIdx) const;
    ActivationStatus activationStatus(int cstrIdx) const;
    ActivationStatus activationStatusBnd(int bndIdx) const;
    const std::vector<ActivationStatus>& activationStatus() const;

    int operator[](int activeIdx) const { return activeSet_[activeIdx]; }

    void activate(int cstrIdx, ActivationStatus status);
    void deactivate(int activeIdx);

    int nbCstr()                  const { return nbCstr_; }
    int nbBnd()                   const { return nbBnd_; }
    int nbAll()                   const { return nbCstr_ + nbBnd_; }

    int nbActiveCstr()            const { return me_+mi_+mb_; }
    int nbActiveEquality()        const { return me_; }
    int nbActiveInequality()      const { return mi_; }
    int nbActiveLowerInequality() const { return ml_; }
    int nbActiveUpperInequality() const { return mu_; }
    int nbActiveBound()           const { return mb_; }
    int nbActiveLowerBound()      const { return mbl_; }
    int nbActiveUpperBound()      const { return mbu_; }

  private:
    std::vector<ActivationStatus> status_;
    std::vector<int> activeSet_;

    int nbCstr_;// number of constraints
    int nbBnd_; // number of bounds
    int me_;    // number of constraints active as equality
    int mi_;    // number of constraints active as general inequality
    int ml_;    // number of inequality constraints active as their lower bound
    int mu_;    // number of inequality constraints active as their upper bound
    int mb_;    // number of constraints active as bound
    int mbl_;   // number of bound constraints active as their lower bound
    int mbu_;   // number of bound constraints active as their upper bound
  };
}