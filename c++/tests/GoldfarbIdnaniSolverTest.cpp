/* Copyright 2020 CNRS-AIST JRL
 */

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include "utils.h"

using namespace Eigen;
using namespace jrlqp;

TEST_CASE("Simple problem")
{
  MatrixXd G = MatrixXd::Identity(3, 3);
  VectorXd a = VectorXd::Zero(3);
  MatrixXd C = MatrixXd::Random(3, 5);
  VectorXd bl = -VectorXd::Ones(5);
  VectorXd bu = VectorXd::Ones(5);
  VectorXd xl(0);
  VectorXd xu(0);

  GoldfarbIdnaniSolver qp(3, 5, false);

  // unconstraint solution
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));

  // At least one constraint activated
  bl[1] = -2; bu[1] = -1;
  G.setIdentity();
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}

TEST_CASE("Simple problem paper")
{
  // example from the Goldfarb-Idnani paper
  MatrixXd G(2, 2); G << 4, -2, -2, 4;
  VectorXd a(2); a << 6, 0;
  MatrixXd C(2, 1); C << 1, 1;
  VectorXd bl(1); bl << 2;
  VectorXd bu(1); bu << 10;
  VectorXd xl = VectorXd::Zero(2);
  VectorXd xu = VectorXd::Constant(2, 10);

  GoldfarbIdnaniSolver qp(2, 1, true);
  qp.options(SolverOptions().logFlags(0xFFFF));

  // unconstraint solution
  qp.solve(G, a, C, bl, bu, xl, xu);

  G << 4, -2, -2, 4;
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}