/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

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

  enum class LogFlags: uint32_t
  {
    TERMINATION = 1
  };
}