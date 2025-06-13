/* Copyright 2020 CNRS-AIST JRL */

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/BoxAndSingleConstraintSolver.h>

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

TEST_CASE("Compare")
{
  const int N = 10;
  int nbVar = 10;

  GoldfarbIdnaniSolver GIsolver(nbVar, 0, true);
  experimental::BoxAndSingleConstraintSolver solver(nbVar);

  // Check with inactive constraint
  for(int i = 0; i <= N; ++i)
  {
    auto pb = generateBoxAndSingleConstraintProblem(nbVar, false);
    test::QPProblem qpp(pb);
    GIsolver.solve(qpp.G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu);
    solver.solve(pb.b, pb.C, pb.l[0], pb.xl, pb.xu);

    FAST_CHECK_UNARY(GIsolver.solution().isApprox(solver.solution()));
    FAST_CHECK_EQ(GIsolver.objectiveValue() + 0.5 * pb.b.squaredNorm(), doctest::Approx(solver.objectiveValue()));
  }

  // Check with active constraint
  for(int i = 0; i <= N; ++i)
  {
    auto pb = generateBoxAndSingleConstraintProblem(nbVar, true);
    test::QPProblem qpp(pb);
    GIsolver.solve(qpp.G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu);
    solver.solve(pb.b, pb.C, pb.l[0], pb.xl, pb.xu);

    FAST_CHECK_UNARY(GIsolver.solution().isApprox(solver.solution()));
    FAST_CHECK_EQ(GIsolver.objectiveValue() + 0.5 * pb.b.squaredNorm(), doctest::Approx(solver.objectiveValue()));
  }
}
