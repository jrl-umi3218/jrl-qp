/* Copyright 2024 CNRS-AIST JRL, Inria */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/test/randomProblems.h>

using namespace jrl::qp::test;

TEST_CASE("Test FactorizedG")
{
  jrl::qp::GoldfarbIdnaniSolver solver(7, 11, false);
  jrl::qp::SolverOptions options;

  for(int i = 0; i < 10; ++i)
  {
    auto pb = QPProblem(randomProblem(ProblemCharacteristics(7, 7, 3, 8)));
    pb.C.transposeInPlace();

    auto llt = pb.G.llt();
    Eigen::MatrixXd L = llt.matrixL();

    options.factorizedG(true);
    solver.options(options);
    solver.solve(L, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x1 = solver.solution();

    options.factorizedG(false);
    solver.options(options);
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x2 = solver.solution();

    FAST_CHECK_UNARY(x1.isApprox(x2));
  }
}