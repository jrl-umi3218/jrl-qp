/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <iosfwd>
#include <string>

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

namespace jrl::qp
{
/** Factorization of G */
enum class GFactorization
{
  /** No factorization */
  NONE,
  /** G is given as the lower triangular matrix L such that G = L L^T */
  L,
  /** G is given as the lower triangular matrix invL such that G^-1 = invL^T invL */
  L_INV,
  /** G is given as the lower triangular matrix invL such that G^-1 = invL invL^T */
  L_TINV,
  /** G is given as a matrix J = L^-T Q, where G = L L^T and Q is the orthonormal
  matrix appearing in the QR decomposition of the activated constraints (see other
  options for details). */
  L_TINV_Q
};

/** Options for the solvers*/
struct JRLQP_DLLAPI SolverOptions
{
  int maxIter_ = 500;
  double bigBnd_ = 1e100;
  bool warmStart_ = false;
  bool equalityFirst_ = false; // True if all equality constraints are given first in the constraint matrix
  bool RIsGiven_ = false; // True when the R factor in the decomposition of some initially active constraints is given.
                          // If equalityFirst is true, these are the equality constraints. Ignored if gFactorization !=
                          // L_TINV_Q or equalityFirst_ = false.
  GFactorization gFactorization_ = GFactorization::NONE;
  std::uint32_t logFlags_ = 0;
  std::ostream * logStream_ = &defaultStream_;

  static std::ostream & defaultStream_;

  std::uint32_t logFlags() const
  {
    return logFlags_;
  }
  SolverOptions & logFlags(LogFlags f)
  {
    logFlags_ = static_cast<std::uint32_t>(f);
    return *this;
  };
  SolverOptions & logFlags(std::uint32_t f)
  {
    logFlags_ = f;
    return *this;
  };
  SolverOptions & addLogFlag(LogFlags f)
  {
    logFlags_ |= static_cast<std::uint32_t>(f);
    return *this;
  }
  SolverOptions & removeLogFlag(LogFlags f)
  {
    logFlags_ &= ~static_cast<std::uint32_t>(f);
    return *this;
  }

  int maxIter() const
  {
    return maxIter_;
  }
  SolverOptions & maxIter(int max)
  {
    maxIter_ = max;
    return *this;
  }

  double bigBnd() const
  {
    return bigBnd_;
  }
  SolverOptions & bigBnd(double big)
  {
    bigBnd_ = big;
    return *this;
  }

  bool warmStart() const
  {
    return warmStart_;
  }
  SolverOptions & warmStart(bool warm)
  {
    warmStart_ = warm;
    return *this;
  }

  bool equalityFirst() const
  {
    return equalityFirst_;
  }
  SolverOptions & equalityFirst(bool eqFirst)
  {
    equalityFirst_ = eqFirst;
    return *this;
  }

  bool RIsGiven() const
  {
    return RIsGiven_;
  }
  SolverOptions & RIsGiven(bool Rgiven)
  {
    RIsGiven_ = Rgiven;
    return *this;
  }

  GFactorization gFactorization() const
  {
    return gFactorization_;
  }
  SolverOptions & gFactorization(GFactorization fact)
  {
    gFactorization_ = fact;
    return *this;
  }

  std::ostream & logStream() const
  {
    return *logStream_;
  }
  SolverOptions & logStream(std::ostream & os)
  {
    logStream_ = &os;
    return *this;
  }
};
} // namespace jrl::qp