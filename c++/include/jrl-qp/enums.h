/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <jrl-qp/defs.h>

namespace jrlqp
{
  /** Activation status of a constraint. */
  enum class ActivationStatus
  {
    INACTIVE,     // Constraint is inactive
    LOWER,        // Constraint is active at its lower bound
    UPPER,        // Constraint is active at its upper bound
    EQUALITY,     // Constraint is and equality constraint
    LOWER_BOUND,  // Constraint is a bound on variable constraint, active at its lower bound
    UPPER_BOUND   // Constraint is a bound on variable constraint, active at its upper bound
  };

  /** Information on the termination reason for the solver. */
  enum class TerminationStatus
  {
    SUCCESS,
    INCONSISTENT_INPUT,
    INFEASIBLE,
    MAX_ITER_REACHED,
    LINEAR_DEPENDENCY_DETECTED
  };

  enum class LogFlags: std::uint32_t
  {
    NONE = 0,
    INPUT = 1 << 0,
    TERMINATION = 1 << 1,
    ITERATION_BASIC_DETAILS = 1 << 2,
    ITERATION_ADVANCE_DETAILS = 1 << 3,
    ACTIVE_SET = 1 << 4,
    ACTIVE_SET_DETAILS = 1 << 5,
    MISC = 1 << 30,
    NO_ITER = constant::noIterationFlag
  };

  inline std::uint32_t operator| (LogFlags a, LogFlags b) 
  { 
    return static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b); 
  }

  inline std::uint32_t operator| (std::uint32_t a, LogFlags b)
  {
    return a | static_cast<std::uint32_t>(b);
  }

  inline std::uint32_t operator| (LogFlags a, std::uint32_t b)
  {
    return static_cast<std::uint32_t>(a) | b;
  }
}