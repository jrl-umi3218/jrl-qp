/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/GoldfarbIdnaniSolver.h>

namespace jrl::qp::experimental
{
/** A specialized solver for problems of the form
 * min. 0.5 ||x - x0||^2
 * s.t. c'x >= bl
 *      xl <= x <= xu
 * where c is a vector.
 */
class JRLQP_DLLAPI BoxAndSingleConstraintSolver : public GoldfarbIdnaniSolver
{
public:
  BoxAndSingleConstraintSolver();
  BoxAndSingleConstraintSolver(int nbVar);

  TerminationStatus solve(const VectorConstRef & x0,
                          const VectorConstRef & c,
                          double bl,
                          const VectorConstRef & xl,
                          const VectorConstRef & xu);

  virtual internal::InitTermination init_() override;

private:
  Eigen::Matrix<double, 1, 1> bl_;
  Eigen::Matrix<double, 1, 1> bu_;
};
} // namespace jrl::qp::experimental

#include <jrl-qp/test/problems.h>
namespace jrl::qp::test
{
/** Generate a problem
 * min. || x- x0 ||
 * s.t. c'x >= bl   (1)
 *      xl<=x<=xu
 * with nbVar variables and the constraint c'x >= bl active if act is true
 * Noting xb the solution without the constraint (1) and su the vertex of the box
 * [xl, xu] the furthest away in the direction given by c, actLevel indicates
 * where the constraint plan intersects the segement [xb, su].
 */
LeastSquareProblem<> JRLQP_DLLAPI generateBoxAndSingleConstraintProblem(int nbVar, bool act, double actLevel = 0.5);

} // namespace jrl::qp::test
