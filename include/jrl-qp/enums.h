/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/defs.h>

namespace jrl::qp
{
/** Activation status of a constraint.
 *
 * \internal Order matters: for example we compare values to LOWER_BOUND to
 * decide if we have a general constraint or a bound.
 */
enum class ActivationStatus
{
  INACTIVE, // Constraint is inactive
  LOWER, // Constraint is active at its lower bound
  UPPER, // Constraint is active at its upper bound
  EQUALITY, // Constraint is an equality constraint
  LOWER_BOUND, // Constraint is a bound on variable constraint, active at its lower bound
  UPPER_BOUND, // Constraint is a bound on variable constraint, active at its upper bound
  FIXED // Constraint fixes the corresponding variable
};

/** Information on the termination reason for the solver. */
enum class TerminationStatus
{
  SUCCESS, // Operation was successful
  INCONSISTENT_INPUT, // Inputs are not consistent with one another (e.g. mismatch of size, activation status, ...)
  NON_POS_HESSIAN, // Quadratic matrix of the problem is expected to be positive definite, but is not.
  INFEASIBLE, // Problem is infeasible
  MAX_ITER_REACHED, // Maximum number of iteration was reached. You can increase this number by using the solver options.
  LINEAR_DEPENDENCY_DETECTED, // Some active constraints are linearly dependent and the solver doesn't know how to
                              // handle this case.
  OVERCONSTRAINED_PROBLEM, // Too many equality constraints and fixed variables
  UNKNOWN, // Unknown status
};

/** Flags for the log and debug outputs.*/
enum class LogFlags : std::uint32_t
{
  NONE = 0,
  INPUT = 1 << 0,
  TERMINATION = 1 << 1,
  ITERATION_BASIC_DETAILS = 1 << 2,
  ITERATION_ADVANCE_DETAILS = 1 << 3,
  ACTIVE_SET = 1 << 4,
  ACTIVE_SET_DETAILS = 1 << 5,
  INIT = 1 << 6,
  MISC = 1 << 30,
  NO_ITER = constant::noIterationFlag
};

inline std::uint32_t operator|(LogFlags a, LogFlags b)
{
  return static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b);
}

inline std::uint32_t operator|(std::uint32_t a, LogFlags b)
{
  return a | static_cast<std::uint32_t>(b);
}

inline std::uint32_t operator|(LogFlags a, std::uint32_t b)
{
  return static_cast<std::uint32_t>(a) | b;
}
} // namespace jrl::qp