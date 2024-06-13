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
    Eigen::MatrixXd invL = L.inverse();
    Eigen::MatrixXd invLT = invL.transpose();

    options.gFactorization(jrl::qp::GFactorization::NONE);
    solver.options(options);
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x0 = solver.solution();

    options.gFactorization(jrl::qp::GFactorization::L);
    solver.options(options);
    solver.solve(L, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x1 = solver.solution();

    options.gFactorization(jrl::qp::GFactorization::L_INV);
    solver.options(options);
    solver.solve(invL, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x2 = solver.solution();

    options.gFactorization(jrl::qp::GFactorization::L_TINV);
    solver.options(options);
    solver.solve(invLT, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x3 = solver.solution();

    FAST_CHECK_UNARY(x1.isApprox(x0));
    FAST_CHECK_UNARY(x2.isApprox(x0));
    FAST_CHECK_UNARY(x3.isApprox(x0));
  }
}

TEST_CASE("Test EqualityFirst")
{
  jrl::qp::GoldfarbIdnaniSolver solver(7, 11, false);
  jrl::qp::SolverOptions options;

  for(int i = 0; i < 10; ++i)
  {
    const int neq = 3;
    auto pb = QPProblem(randomProblem(ProblemCharacteristics(7, 7, neq, 8)));
    pb.C.transposeInPlace();

    for (int i = 0; i < neq; ++i)
    {
      REQUIRE_EQ(pb.l[i], pb.u[i]);
    }

    auto llt = pb.G.llt();
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::MatrixXd invL = L.inverse();
    Eigen::MatrixXd invLT = invL.transpose();

    options.gFactorization(jrl::qp::GFactorization::NONE);
    solver.options(options);
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x0 = solver.solution();

    options.equalityFirst(true);
    solver.options(options);
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x1 = solver.solution();

    options.gFactorization(jrl::qp::GFactorization::L);
    solver.options(options);
    solver.solve(L, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x2 = solver.solution();

    //options.gFactorization(jrl::qp::GFactorization::L_INV);
    //solver.options(options);
    //solver.solve(invL, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    //Eigen::VectorXd x3 = solver.solution();

    //options.gFactorization(jrl::qp::GFactorization::L_TINV);
    //solver.options(options);
    //solver.solve(invLT, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    //Eigen::VectorXd x4 = solver.solution();

    FAST_CHECK_UNARY(x1.isApprox(x0));
    FAST_CHECK_UNARY(x2.isApprox(x0));
    //FAST_CHECK_UNARY(x3.isApprox(x0));
    //FAST_CHECK_UNARY(x4.isApprox(x0));
  }
}

TEST_CASE("Precomputed R")
{
  const int nbVar = 7;
  const int neq = 4;
  const int nIneq = 8;
  jrl::qp::GoldfarbIdnaniSolver solver(nbVar, neq+nIneq, false);
  jrl::qp::SolverOptions options;

  for(int i = 0; i < 10; ++i)
  {
    auto pb = QPProblem(randomProblem(ProblemCharacteristics(nbVar, nbVar, neq, nIneq)));
    pb.C.transposeInPlace();

    for(int i = 0; i < neq; ++i)
    {
      REQUIRE_EQ(pb.l[i], pb.u[i]);
    }

    auto llt = pb.G.llt();

    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(nbVar, nbVar);
    Eigen::VectorXd tmp(nbVar);
    llt.matrixL().transpose().solveInPlace(J);
    Eigen::MatrixXd B = J.template triangularView<Eigen::Lower>().transpose() * pb.C.leftCols(neq);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(B);
    qr.householderQ().applyThisOnTheRight(J, tmp);

    options.gFactorization(jrl::qp::GFactorization::NONE);
    solver.options(options);
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x0 = solver.solution();

    options.gFactorization(jrl::qp::GFactorization::L_TINV_Q);
    options.RIsGiven(true);
    solver.options(options);
    solver.setPrecomputedR(Eigen::MatrixXd(qr.matrixQR().triangularView<Eigen::Upper>()));
    solver.solve(J, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
    Eigen::VectorXd x1 = solver.solution();

    FAST_CHECK_UNARY(x1.isApprox(x0));
  }
}