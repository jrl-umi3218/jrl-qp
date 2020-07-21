/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <jrl-qp/GoldfarbIdnaniSolver.h>

namespace jrlqp::experimental
{
  class JRLQP_DLLAPI BoxAndSingleConstraintSolver : public GoldfarbIdnaniSolver
  {
  public:
    BoxAndSingleConstraintSolver();
    BoxAndSingleConstraintSolver(int nbVar);

    TerminationStatus solve(const VectorConstRef& x0, const VectorConstRef& c, double bl,
      const VectorConstRef& xl, const VectorConstRef& xu);

    virtual internal::InitTermination init_() override;

  private:
    Eigen::Matrix<double, 1, 1> bl_;
    Eigen::Matrix<double, 1, 1> bu_;
  };
}