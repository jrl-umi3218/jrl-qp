/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>
#include <string>

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

namespace jrlqp
{
  /** Options for the solvers*/
  struct JRLQP_DLLAPI SolverOptions
  {
    int maxIter_ = 500;
    double bigBnd_ = 1e100;
    std::uint32_t logFlags_;
    std::ostream* logStream_ = &defaultStream_;

    static std::ostream& defaultStream_;

    SolverOptions& logFlags(LogFlags f) { logFlags_ = static_cast<std::uint32_t>(f); return *this; };
    SolverOptions& logFlags(std::uint32_t f) { logFlags_ = f; return *this; };
    SolverOptions& addLogFlag(LogFlags f) { logFlags_ |= static_cast<std::uint32_t>(f); return *this; }
    SolverOptions& removeLogFlag(LogFlags f) { logFlags_ &= ~static_cast<std::uint32_t>(f); return *this; }
  };
}