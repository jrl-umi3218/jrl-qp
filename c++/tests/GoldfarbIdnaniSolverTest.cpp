/* Copyright 2020 CNRS-AIST JRL
 */

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/GoldfarbIdnaniSolver.h>

using namespace Eigen;

TEST_CASE("Simple problem")
{
  MatrixXd G = MatrixXd::Identity(3, 3);
  VectorXd a = VectorXd::Zero(3);
  MatrixXd C = MatrixXd::Random(3, 5);
  VectorXd bl = -VectorXd::Ones(5);
  VectorXd bu = VectorXd::Ones(5);
  VectorXd xl(0);
  VectorXd xu(0);

  jrlqp::GoldfarbIdnaniSolver qp(3, 5, false);
  qp.solve(G, a, C, bl, bu, xl, xu);

  std::cout << qp.solution().transpose() << std::endl;
  std::cout << qp.multipliers() << std::endl;

  bl[1] = -2; bu[1] = -1;
  G.setIdentity();
  qp.solve(G, a, C, bl, bu, xl, xu);

  std::cout << qp.solution().transpose() << std::endl;
  std::cout << qp.multipliers() << std::endl;
}