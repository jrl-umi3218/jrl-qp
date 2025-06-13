/* Copyright 2020 CNRS-AIST JRL */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <assert.h>

#include <jrl-qp/internal/ActiveSet.h>

using namespace jrl::qp;

const static ActivationStatus I = ActivationStatus::INACTIVE;
const static ActivationStatus E = ActivationStatus::EQUALITY;
const static ActivationStatus L = ActivationStatus::LOWER;
const static ActivationStatus U = ActivationStatus::UPPER;
const static ActivationStatus LB = ActivationStatus::LOWER_BOUND;
const static ActivationStatus UB = ActivationStatus::UPPER_BOUND;

const static bool F = false;
const static bool T = true;

void checkActivation(const internal::ActiveSet & as, const std::vector<bool> & act)
{
  assert(as.nbAll() == static_cast<int>(act.size()));
  for(int i = 0; i < as.nbAll(); ++i) FAST_CHECK_EQ(as.isActive(i), act[i]);
}

void checkStatus(const internal::ActiveSet & as, const std::vector<ActivationStatus> & status)
{
  for(int i = 0; i < as.nbAll(); ++i) FAST_CHECK_EQ(as.activationStatus(i), status[i]);
}

void checkActiveIdx(const internal::ActiveSet & as, const std::vector<int> & actIdx)
{
  FAST_CHECK_EQ(as.nbActiveCstr(), actIdx.size());
  for(int i = 0; i < as.nbActiveCstr(); ++i) FAST_CHECK_EQ(as[i], actIdx[i]);
}

void checkNb(const internal::ActiveSet & as,
             int nbCstr,
             int nbEq,
             int nbIneq,
             int nbLIneq,
             int nbUIneq,
             int nbBnd,
             int nbLBnd,
             int nbUBnd)
{
  FAST_CHECK_EQ(as.nbActiveCstr(), nbCstr);
  FAST_CHECK_EQ(as.nbActiveEquality(), nbEq);
  FAST_CHECK_EQ(as.nbActiveInequality(), nbIneq);
  FAST_CHECK_EQ(as.nbActiveLowerInequality(), nbLIneq);
  FAST_CHECK_EQ(as.nbActiveUpperInequality(), nbUIneq);
  FAST_CHECK_EQ(as.nbActiveBound(), nbBnd);
  FAST_CHECK_EQ(as.nbActiveLowerBound(), nbLBnd);
  FAST_CHECK_EQ(as.nbActiveUpperBound(), nbUBnd);
}

TEST_CASE("Test ActiveSet Ctor")
{
  internal::ActiveSet as(5, 3);

  checkActivation(as, {F, F, F, F, F, F, F, F});
  checkStatus(as, {I, I, I, I, I, I, I, I});
  checkActiveIdx(as, {});
  checkNb(as, 0, 0, 0, 0, 0, 0, 0, 0);
}

TEST_CASE("Activation")
{
  internal::ActiveSet as(5, 3);

  as.activate(3, ActivationStatus::EQUALITY);
  checkActivation(as, {F, F, F, T, F, F, F, F});
  checkStatus(as, {I, I, I, E, I, I, I, I});
  checkActiveIdx(as, {3});
  checkNb(as, 1, 1, 0, 0, 0, 0, 0, 0);

  as.activate(6, ActivationStatus::UPPER_BOUND);
  checkActivation(as, {F, F, F, T, F, F, T, F});
  checkStatus(as, {I, I, I, E, I, I, UB, I});
  checkActiveIdx(as, {3, 6});
  checkNb(as, 2, 1, 0, 0, 0, 1, 0, 1);

  as.activate(2, ActivationStatus::LOWER);
  checkActivation(as, {F, F, T, T, F, F, T, F});
  checkStatus(as, {I, I, L, E, I, I, UB, I});
  checkActiveIdx(as, {3, 6, 2});
  checkNb(as, 3, 1, 1, 1, 0, 1, 0, 1);

  as.activate(4, ActivationStatus::UPPER);
  checkActivation(as, {F, F, T, T, T, F, T, F});
  checkStatus(as, {I, I, L, E, U, I, UB, I});
  checkActiveIdx(as, {3, 6, 2, 4});
  checkNb(as, 4, 1, 2, 1, 1, 1, 0, 1);

  as.deactivate(1);
  checkActivation(as, {F, F, T, T, T, F, F, F});
  checkStatus(as, {I, I, L, E, U, I, I, I});
  checkActiveIdx(as, {3, 2, 4});
  checkNb(as, 3, 1, 2, 1, 1, 0, 0, 0);

  as.activate(7, ActivationStatus::LOWER_BOUND);
  checkActivation(as, {F, F, T, T, T, F, F, T});
  checkStatus(as, {I, I, L, E, U, I, I, LB});
  checkActiveIdx(as, {3, 2, 4, 7});
  checkNb(as, 4, 1, 2, 1, 1, 1, 1, 0);

  as.deactivate(2);
  checkActivation(as, {F, F, T, T, F, F, F, T});
  checkStatus(as, {I, I, L, E, I, I, I, LB});
  checkActiveIdx(as, {3, 2, 7});
  checkNb(as, 3, 1, 1, 1, 0, 1, 1, 0);

  as.deactivate(2);
  checkActivation(as, {F, F, T, T, F, F, F, F});
  checkStatus(as, {I, I, L, E, I, I, I, I});
  checkActiveIdx(as, {3, 2});
  checkNb(as, 2, 1, 1, 1, 0, 0, 0, 0);

  as.deactivate(0);
  checkActivation(as, {F, F, T, F, F, F, F, F});
  checkStatus(as, {I, I, L, I, I, I, I, I});
  checkActiveIdx(as, {2});
  checkNb(as, 1, 0, 1, 1, 0, 0, 0, 0);

  as.deactivate(0);
  checkActivation(as, {F, F, F, F, F, F, F, F});
  checkStatus(as, {I, I, I, I, I, I, I, I});
  checkActiveIdx(as, {});
  checkNb(as, 0, 0, 0, 0, 0, 0, 0, 0);
}
