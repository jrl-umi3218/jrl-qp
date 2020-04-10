/* Copyright 2020 CNRS-AIST JRL
 */

#include <assert.h>

#include <jrl-qp/internal/ActiveSet.h>

namespace jrlqp::internal
{
  ActiveSet::ActiveSet()
    : ActiveSet(0)
  {
  }

  ActiveSet::ActiveSet(int nCstr, int nBnd)
  {
    resize(nCstr, nBnd);
  }

  void ActiveSet::resize(int nCstr, int nBnd)
  {
    assert(nCstr >= 0);
    assert(nBnd >= 0);
    size_t nTot = static_cast<size_t>(nCstr) + static_cast<size_t>(nBnd);
    status_.resize(nTot);
    activeSet_.reserve(nTot);

    nbCstr_ = nCstr;
    nbBnd_ = nBnd;

    reset();
  }

  void ActiveSet::reset()
  {
    std::fill(status_.begin(), status_.end(), ActivationStatus::INACTIVE);
    activeSet_.clear();

    me_  = 0;
    mi_  = 0;
    ml_  = 0;
    mu_  = 0;
    mb_  = 0;
    mbl_ = 0;
    mbu_ = 0;
  }

  bool ActiveSet::isActive(int cstrIdx) const
  {
    assert(cstrIdx < nbCstr_ + nbBnd_);
    return status_[cstrIdx] != ActivationStatus::INACTIVE;
  }
  bool ActiveSet::isActiveBnd(int bndIdx) const
  {
    assert(bndIdx < nbBnd_);
    return status_[static_cast<size_t>(nbCstr_) + bndIdx] != ActivationStatus::INACTIVE;;
  }
  ActivationStatus ActiveSet::activationStatus(int cstrIdx) const
  {
    assert(cstrIdx < nbCstr_ + nbBnd_);
    return status_[cstrIdx];
  }
  ActivationStatus ActiveSet::activationStatusBnd(int bndIdx) const
  {
    assert(bndIdx < nbBnd_);
    return status_[static_cast<size_t>(nbCstr_) + bndIdx];
  }
  void ActiveSet::activate(int cstrIdx, ActivationStatus status)
  {
    assert(cstrIdx < nbCstr_ + nbBnd_);
    assert(status_[cstrIdx] == ActivationStatus::INACTIVE && "Specified constraint is already active");
    assert(status != ActivationStatus::INACTIVE && "You need to specify an non-inactive status");
    assert((cstrIdx < nbCstr_ && (status == ActivationStatus::EQUALITY || status == ActivationStatus::LOWER || status == ActivationStatus::UPPER))
        || (cstrIdx >= nbCstr_ && (status == ActivationStatus::LOWER_BOUND || status == ActivationStatus::UPPER_BOUND))
        && "The given status is not compatible with the constraint index");

    activeSet_.push_back(cstrIdx);
    status_[cstrIdx] = status;

    switch (status)
    {
    case ActivationStatus::LOWER:       ++mi_; ++ml_;   break;
    case ActivationStatus::UPPER:       ++mi_; ++mu_;   break;
    case ActivationStatus::EQUALITY:    ++me_;          break;
    case ActivationStatus::LOWER_BOUND: ++mb_; ++mbl_;  break;
    case ActivationStatus::UPPER_BOUND: ++mb_; ++mbu_;  break;
    default: assert(false);
    }
  }
  void ActiveSet::deactivate(int activeIdx)
  {
    int cstrIdx = activeSet_[activeIdx];
    auto status = status_[cstrIdx];

    activeSet_.erase(activeSet_.begin() + activeIdx);
    status_[cstrIdx] = ActivationStatus::INACTIVE;

    switch (status)
    {
    case ActivationStatus::LOWER:       --mi_; --ml_;   break;
    case ActivationStatus::UPPER:       --mi_; --mu_;   break;
    case ActivationStatus::EQUALITY:    --me_;          break;
    case ActivationStatus::LOWER_BOUND: --mb_; --mbl_;  break;
    case ActivationStatus::UPPER_BOUND: --mb_; --mbu_;  break;
    default: assert(false);
    }
  }
}