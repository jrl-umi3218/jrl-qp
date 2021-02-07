/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/enums.h>

namespace jrl::qp::internal
{
/** Helper class to represent the selection of a violated constraint
 *
 * This is simply an (index, status) pair.
 */
class SelectedConstraint
{
public:
  /** Default constructor.*/
  SelectedConstraint() : p_(-1), status_(ActivationStatus::INACTIVE) {}
  /** Usual constructor
   *
   * \param p Index of the selected constraint
   * \param status Activation status of the selected constraint
   */
  SelectedConstraint(int p, ActivationStatus status) : p_(p), status_(status) {}

  /** Underlying index of the constraint.*/
  int index() const
  {
    return p_;
  }
  /** Activation status of the corresponding constraint.*/
  ActivationStatus status() const
  {
    return status_;
  }

  friend std::ostream & operator<<(std::ostream & os, const SelectedConstraint & sc)
  {
    os << "{" << sc.p_ << ", " << static_cast<int>(sc.status_) << "}";
    return os;
  }

private:
  int p_;
  ActivationStatus status_;
};
} // namespace jrl::qp::internal