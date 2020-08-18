/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include<Eigen/Core>

#include<jrl-qp/api.h>

namespace jrl::qp::internal
{
  /** Check if dynamic allocation is allowed in Eigen operations. */
  void JRLQP_DLLAPI check_that_malloc_is_allowed();

  /** Allow or disallow dynamic allocation in Eigen operations. */
  bool JRLQP_DLLAPI set_is_malloc_allowed(bool allow);
}