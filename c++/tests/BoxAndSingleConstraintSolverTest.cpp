/* Copyright 2020 CNRS-AIST JRL
 */

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/BoxAndSingleConstraintSolver.h>
#include <jrl-qp/test/randomProblems.h>

using namespace Eigen;
using namespace jrlqp;

TEST_CASE("Compare")
{
  const int N = 10;
  bool act = true;

  MatrixXd I = MatrixXd::Identity(N, N);
  VectorXd x0 = VectorXd::Random(N);
  VectorXd r1 = VectorXd::Random(N);
  VectorXd r2 = VectorXd::Random(N);
  VectorXd xl(N);
  VectorXd xu(N);
  VectorXd xb(N);
  for (int i = 0; i < N; ++i)
  {
    if (r1[i] < r2[i])
    {
      xl[i] = r1[i];
      xu[i] = r2[i];
    }
    else
    {
      xl[i] = r2[i];
      xu[i] = r1[i];
    }
    if (x0[i] < xl[i])
      xb[i] = xl[i];
    else if (x0[i] > xu[i])
      xb[i] = xu[i];
    else
      xb[i] = x0[i];
  }

  VectorXd c = VectorXd::Random(N);
  // Points of the box the closer and further away in the direction of c
  VectorXd sl(N);
  VectorXd su(N);
  for (int i = 0; i < N; ++i)
  {
    if (c[i] > 0)
    {
      sl[i] = xl[i];
      su[i] = xu[i];
    }
    else
    {
      sl[i] = xu[i];
      su[i] = xl[i];
    }
  }


  //VectorXd u(N);
  //double min = std::numeric_limits<double>::infinity();
  //double max = -std::numeric_limits<double>::infinity();
  //int imin, imax;
  //for (int i = 0; i < 1024; ++i)
  //{
  //  for (int j = 0; j < N; ++j)
  //  {
  //    if ((i >> j) % 2)
  //      u[j] = xl[j];
  //    else
  //      u[j] = xu[j];
  //  }
  //  double d = c.dot(u);
  //  if (d < min)
  //  {
  //    min = d;
  //    imin = i;
  //  }
  //  if (d > max)
  //  {
  //    max = d;
  //    imax = i;
  //  }
  //}

  //std::cout << "min = " << min << ", c.sl = " << c.dot(sl) << std::endl;
  //std::cout << "max = " << max << ", c.su = " << c.dot(su) << std::endl;

  double b;
  if (act)
  {
    // If we want the constraint c'x >= b active we chose b so that d2 <= c'x <= d1 
    double d1 = c.dot(xb); // minimum b so that the constraint is active
    double d2 = c.dot(su); // maximum b so that the problem is feasible
    b = 0.5 * (d1 + d2);
  }
  else
  {
    b = c.dot(sl); //the constraint doesn't intersect the box
  }

  VectorXd bl(1); bl[0] = b;
  VectorXd bu(1); bu[0] = std::numeric_limits<double>::infinity();

  GoldfarbIdnaniSolver GIsolver(N, 0, true);
  GIsolver.solve(I, -x0, c, bl, bu, xl, xu);

  SolverOptions opt;
  opt.logFlags(LogFlags::ITERATION_BASIC_DETAILS
    | LogFlags::ACTIVE_SET
    | LogFlags::ACTIVE_SET_DETAILS
    | LogFlags::ITERATION_ADVANCE_DETAILS
  );

  experimental::BoxAndSingleConstraintSolver solver(N);
  solver.options(opt);
  solver.solve(x0, c, b, xl, xu);

  std::cout << GIsolver.solution().transpose() << std::endl;
  std::cout << solver.solution().transpose() << std::endl;
}