/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <iosfwd>
#include <vector>

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

namespace jrl::qp::internal
{
/** A class to track the activation status of a set of constraints and bounds.
 *
 * This class uses two types of representations and accompanying indices:
 *  - on one hand, it maintains a vector with the status of all constraints
 *    and bounds, active or not. This is the status vector, that is organized
 *    with the (general) constraints first and the bounds after.
 *  - on the other hand it acts as a vector of active constraints, where
 *    element \p i of the vector is the index of an active constraint with
 *    respect to the status vector.
 */
class JRLQP_DLLAPI ActiveSet
{
public:
  /** Default constructor */
  ActiveSet();
  /** Constructor declaring a predefined number of constraints and bounds.
   *
   * \param nCstr Number of constraints.
   * \param nBnd Number of bounds.
   */
  ActiveSet(int nCstr, int nBnd = 0);
  /** Constructor for a vector of activation status, such as can be returned by
   * jrl::qp::ActiveSet::activationStatus().
   *
   * \param as Vector of activation status with status of general constraints
   * first, followed by the status of bounds. as[i] contains the activation
   * status of the i-th constraint/bound, wether it is active or not.
   * \param nBnd Number of bounds.
   */
  ActiveSet(const std::vector<ActivationStatus> & as, int nBnd);

  /** Constructor declaring a predefined number of constraints and bounds.
   *
   * \param nCstr Number of constraints.
   * \param nBnd Number of bounds.
   */
  void resize(int nCstr, int nBnd = 0);

  /** Set all constraints to inactive.*/
  void reset();

  /** Check if constraint with index \p cstrIdx is active.
   * If \p cstrIdx >= \p nbCstr(), bounds are accessed.
   */
  bool isActive(int cstrIdx) const;
  /** Check if the bound with index \p bndIdx is active, where 0 corresponds to
   * the first bound.
   */
  bool isActiveBnd(int bndIdx) const;
  /** Get the activation status of the constraint with index \p cstrIdx.
   * If \p cstrIdx >= \p nbCstr(), bounds are accessed.
   */
  ActivationStatus activationStatus(int cstrIdx) const;
  /** Get the activation status of the bound with index \p bndIdx, where 0
   * corresponds to the first bound.
   */
  ActivationStatus activationStatusBnd(int bndIdx) const;
  /** Return the vector of activation status with general constraints first and
   * bounds after.
   */
  const std::vector<ActivationStatus> & activationStatus() const;

  /** Return the index of the \p activeIdx-th constraint.*/
  int operator[](int activeIdx) const
  {
    return activeSet_[activeIdx];
  }

  /** Activate a constraint with a given status
   *
   * \param cstrIdx Index of the constraint to activate. Indices 0 to nbCstr()-1
   * indicate general constraints. Indices from nbCstr() to nbCstr()+nbBnd()-1
   * indicate bounds.
   * \param status Status of the activated constraint. The status must be
   * compatible with the type (general, bound) of the constraint being activated.
   */
  void activate(int cstrIdx, ActivationStatus status);
  /** Deactivate a constraint.
   *
   * \param activeIdx Index of the constraint to deactivate, with respect to
   * the set of activated constraints.
   */
  void deactivate(int activeIdx);

  /** Number of (general) constraints. */
  int nbCstr() const
  {
    return nbCstr_;
  }
  /** Number of bounds. */
  int nbBnd() const
  {
    return nbBnd_;
  }
  /** Number of general constraints and bounds (i.e. the size of the status vector.*/
  int nbAll() const
  {
    return nbCstr_ + nbBnd_;
  }

  int nbActiveCstr() const
  {
    return me_ + mi_ + mb_;
  }
  int nbActiveEquality() const
  {
    return me_;
  }
  int nbActiveInequality() const
  {
    return mi_;
  }
  int nbActiveLowerInequality() const
  {
    return ml_;
  }
  int nbActiveUpperInequality() const
  {
    return mu_;
  }
  int nbActiveBound() const
  {
    return mb_;
  }
  int nbActiveLowerBound() const
  {
    return mbl_;
  }
  int nbActiveUpperBound() const
  {
    return mbu_;
  }
  int nbFixedVariable() const
  {
    return mbe_;
  }

private:
  /** Activation status of ALL the constraints. */
  std::vector<ActivationStatus> status_;
  /** Index of the active constraints in an order determined by the order of
   * activations and deactivation. Indices 0 to nbCstr_-1 indicate general
   * constraints. Indices from nbCstr_ to nbCstr_+nbBnd_-1 indicate bounds.
   */
  std::vector<int> activeSet_;

  int nbCstr_; // number of constraints
  int nbBnd_; // number of bounds
  int me_; // number of constraints active as equality
  int mi_; // number of constraints active as general inequality
  int ml_; // number of inequality constraints active as their lower bound
  int mu_; // number of inequality constraints active as their upper bound
  int mb_; // number of constraints active as bound
  int mbl_; // number of bound constraints active as their lower bound
  int mbu_; // number of bound constraints active as their upper bound
  int mbe_; // number of fixed variable
};

/** Printing an active set to a stream. */
std::ostream & operator<<(std::ostream & os, const ActiveSet & a);
} // namespace jrl::qp::internal