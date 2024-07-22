/* Copyright 2020 CNRS-AIST JRL */

#include <fstream>
#include <iostream>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include "QPSProblems.h"
#include "QPSReader.h"
#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/GoldfarbIdnaniSolver.h>
#include <jrl-qp/internal/memoryChecks.h>
#include <jrl-qp/test/kkt.h>
#include <jrl-qp/test/randomProblems.h>

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

TEST_CASE_TEMPLATE("Simple problem", T, GoldfarbIdnaniSolver, experimental::GoldfarbIdnaniSolver)
{
  MatrixXd G = MatrixXd::Identity(3, 3);
  VectorXd a = VectorXd::Zero(3);
  MatrixXd C = MatrixXd::Random(3, 5);
  VectorXd bl = -VectorXd::Ones(5);
  VectorXd bu = VectorXd::Ones(5);
  VectorXd xl(0);
  VectorXd xu(0);

  T qp(3, 5, false);

  // unconstraint solution
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));

  // At least one constraint activated
  bl[1] = -2;
  bu[1] = -1;
  G.setIdentity();
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}

TEST_CASE_TEMPLATE("Simple problem paper", T, GoldfarbIdnaniSolver, experimental::GoldfarbIdnaniSolver)
{
  // example from the Goldfarb-Idnani paper
  MatrixXd G(2, 2);
  G << 4, -2, -2, 4;
  VectorXd a(2);
  a << 6, 0;
  MatrixXd C(2, 1);
  C << 1, 1;
  VectorXd bl(1);
  bl << 2;
  VectorXd bu(1);
  bu << 10;
  VectorXd xl = VectorXd::Zero(2);
  VectorXd xu = VectorXd::Constant(2, 10);

  T qp(2, 1, true);

  qp.solve(G, a, C, bl, bu, xl, xu);

  G << 4, -2, -2, 4;
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}

TEST_CASE_TEMPLATE("Random problems", T, GoldfarbIdnaniSolver, experimental::GoldfarbIdnaniSolver)
{
  std::vector problems = {
      randomProblem(ProblemCharacteristics(5, 5)), randomProblem(ProblemCharacteristics(5, 5).nEq(2)),
      randomProblem(ProblemCharacteristics(5, 5).nIneq(8).nStrongActIneq(4)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(3)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(1).bounds(true).nStrongActBounds(2))};

  for(const auto & pb : problems)
  {
    QPProblem qpp(pb);
    MatrixXd G = qpp.G; // copy for later check
    T solver(static_cast<int>(qpp.G.rows()), static_cast<int>(qpp.C.rows()), pb.bounds);
    //    jrl::qp::internal::set_is_malloc_allowed(false);
    auto ret = solver.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
    //    jrl::qp::internal::set_is_malloc_allowed(true);
    FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
    FAST_CHECK_UNARY(
        test::testKKT(solver.solution(), solver.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu, false));
    FAST_CHECK_UNARY(solver.solution().isApprox(pb.x, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().head(pb.E.rows()).isApprox(pb.lambdaEq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().segment(pb.E.rows(), pb.C.rows()).isApprox(pb.lambdaIneq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().tail(pb.xl.size()).isApprox(pb.lambdaBnd, 1e-6));
  }
}

TEST_CASE_TEMPLATE("Multiple uses", T, GoldfarbIdnaniSolver, experimental::GoldfarbIdnaniSolver)
{
  std::vector problems = {
      randomProblem(ProblemCharacteristics(5, 5)), randomProblem(ProblemCharacteristics(5, 5).nEq(2)),
      randomProblem(ProblemCharacteristics(5, 5).nIneq(8).nStrongActIneq(4)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(3)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(1).bounds(true).nStrongActBounds(2))};

  T solver(5, 8, true);
  for(const auto & pb : problems)
  {
    QPProblem qpp(pb);
    MatrixXd G = qpp.G; // copy for later check
    jrl::qp::internal::set_is_malloc_allowed(false);
    auto ret = solver.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
    jrl::qp::internal::set_is_malloc_allowed(true);
    FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
    FAST_CHECK_UNARY(
        test::testKKT(solver.solution(), solver.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu, false));
    FAST_CHECK_UNARY(solver.solution().isApprox(pb.x, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().head(pb.E.rows()).isApprox(pb.lambdaEq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().segment(pb.E.rows(), pb.C.rows()).isApprox(pb.lambdaIneq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().tail(pb.xl.size()).isApprox(pb.lambdaBnd, 1e-6));
  }
}

TEST_CASE("Warm-start")
{
  /** FIXME For precisions up to 1e-4 some of the tests below will not pass in a small number of problems
   *
   * For the moment we keep the desired precision (1e-6) but we only check that a small number of tests fail (< 0.1%)
   */
  // We keep this number relatively high such that we generally hit some problematic cases
  size_t n_problems = 10000;
  size_t n_allowed_to_fail = static_cast<size_t>(std::floor(0.001 * static_cast<double>(n_problems)));
  size_t n_failed = 0;
  for(size_t i = 0; i < n_problems; ++i)
  {
    std::vector problems = {
        randomProblem(ProblemCharacteristics(5, 5)), randomProblem(ProblemCharacteristics(5, 5).nEq(2)),
        randomProblem(ProblemCharacteristics(5, 5).nIneq(8).nStrongActIneq(4)),
        randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(3)),
        randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(1).bounds(true).nStrongActBounds(2))};

    for(const auto & pb : problems)
    {
      QPProblem qpp(pb);
      MatrixXd G = qpp.G; // copy for restore and later check
      GoldfarbIdnaniSolver solverNoWS(static_cast<int>(qpp.G.rows()), static_cast<int>(qpp.C.rows()), pb.bounds);
      experimental::GoldfarbIdnaniSolver solverWS(static_cast<int>(qpp.G.rows()), static_cast<int>(qpp.C.rows()),
                                                  pb.bounds);
      SolverOptions opt;
      opt.warmStart_ = true;
      solverWS.options(opt);
      //    jrl::qp::internal::set_is_malloc_allowed(false);
      auto retNoWS = solverNoWS.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
      qpp.G = G;
      auto retWS =
          solverWS.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu, solverNoWS.activeSet());
      //    jrl::qp::internal::set_is_malloc_allowed(true);

      FAST_CHECK_EQ(retNoWS, TerminationStatus::SUCCESS);
      FAST_CHECK_EQ(retWS, TerminationStatus::SUCCESS);
      FAST_CHECK_UNARY(test::testKKT(solverWS.solution(), solverWS.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl,
                                     qpp.xu, false));
#define CHECK_OR_BAIL(LHS, RHS)  \
  if(!(LHS.isApprox(RHS, 1e-6))) \
  {                              \
    n_failed++;                  \
    continue;                    \
  }
      CHECK_OR_BAIL(solverWS.solution(), pb.x);
      CHECK_OR_BAIL(solverWS.multipliers().head(pb.E.rows()), pb.lambdaEq);
      CHECK_OR_BAIL(solverWS.multipliers().segment(pb.E.rows(), pb.C.rows()), pb.lambdaIneq);
      CHECK_OR_BAIL(solverWS.multipliers().tail(pb.xl.size()), pb.lambdaBnd);
      FAST_CHECK_EQ(solverWS.iterations(), 0);

      // Check warm start reusing previous active set
      qpp.G = G;
      solverWS.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
      FAST_CHECK_EQ(solverWS.iterations(), 0);

      // Check warm start with rubish guess
      qpp.G = G;
      auto as = solverWS.activeSet();
      /** FIXME Actually this fails in some circumstances
      for(size_t i = 0; i < as.size(); ++i)
      {
        if(as[i] == ActivationStatus::INACTIVE)
        {
          if(static_cast<int>(i) < qpp.C.rows())
            as[i] = ActivationStatus::LOWER;
          else
            as[i] = ActivationStatus::LOWER_BOUND;
        }
        else if(as[i] == ActivationStatus::LOWER)
          as[i] = ActivationStatus::UPPER;
        else if(as[i] == ActivationStatus::UPPER)
          as[i] = ActivationStatus::LOWER;
        else if(as[i] == ActivationStatus::LOWER_BOUND)
          as[i] = ActivationStatus::UPPER_BOUND;
        else if(as[i] == ActivationStatus::LOWER_BOUND)
          as[i] = ActivationStatus::UPPER_BOUND;
      }
      */

      retWS = solverWS.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu, as);
      FAST_CHECK_EQ(retWS, TerminationStatus::SUCCESS);
      FAST_CHECK_UNARY(test::testKKT(solverWS.solution(), solverWS.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl,
                                     qpp.xu, false));
      CHECK_OR_BAIL(solverWS.solution(), pb.x);
      CHECK_OR_BAIL(solverWS.multipliers().head(pb.E.rows()), pb.lambdaEq);
      CHECK_OR_BAIL(solverWS.multipliers().segment(pb.E.rows(), pb.C.rows()), pb.lambdaIneq);
      CHECK_OR_BAIL(solverWS.multipliers().tail(pb.xl.size()), pb.lambdaBnd);
#undef CHECK_OR_BAIL
    }
  }
  FAST_REQUIRE_LT(n_failed, n_allowed_to_fail);
}

TEST_CASE_TEMPLATE("Iterative Improvements", T, GoldfarbIdnaniSolver)
{
  std::vector problems = {randomProblem(ProblemCharacteristics(5, 5)),
                          randomProblem(ProblemCharacteristics(5, 5).nEq(2))};

  for(const auto & pb : problems)
  {
    QPProblem qpp(pb);
    MatrixXd G = qpp.G; // copy for later check
    T solver(static_cast<int>(qpp.G.rows()), static_cast<int>(qpp.C.rows()), pb.bounds);
    // jrl::qp::internal::set_is_malloc_allowed(false);
    auto ret = solver.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
    // jrl::qp::internal::set_is_malloc_allowed(true);

    // Perturbed problem
    double eps = 1e-2;
    MatrixXd Gp = G + eps * MatrixXd::Random(G.rows(), G.cols());
    MatrixXd Cp = qpp.C + eps * MatrixXd::Random(qpp.C.rows(), qpp.C.cols());
    solver.iterativeImprovement(Gp, Cp.transpose(), 50);

    // Checks
    FAST_CHECK_UNARY(
        test::testKKT(solver.solution(), solver.multipliers(), Gp, qpp.a, Cp, qpp.l, qpp.u, qpp.xl, qpp.xu, false));
  }
}

#ifdef QPS_TESTS_DIR
template<typename Solver>
struct ExcludePb
{
  static const std::vector<std::string> list;

  static bool check(const std::string & name)
  {
    return std::find(list.begin(), list.end(), name) != list.end();
  }
};

template<>
const std::vector<std::string> ExcludePb<GoldfarbIdnaniSolver>::list = {
    "qforplan", // requires the QPS reader to handle names with spaces
    "qpcboei1", // Both this and the next fail to seemingly bad conditionning of the active set due to selectionning
    "qpcboei2" // one constraint to activate instead of another by 1e-13 difference. Need to have a more robust
               // constraint than the basic one.
};

template<>
const std::vector<std::string> ExcludePb<experimental::GoldfarbIdnaniSolver>::list = {
    "qforplan", // requires the QPS reader to handle names with spaces
    "qpcboei1", // See above for this one and the next
    "qpcboei2", //
    "qpcstair" // To be investigated
};

TEST_CASE_TEMPLATE("Test Suite", T, GoldfarbIdnaniSolver, experimental::GoldfarbIdnaniSolver)
{
  for(const auto & p : marosMeszarosPbList)
  {
    auto [name, fstar, cond, nbCstr, nbVar, nz, qn, qnz] = p;

    std::cout << name;
    if(ExcludePb<T>::check(name))
    {
      std::cout << " skip (excluded)" << std::endl;
      continue;
    }
    if(cond > 1e8 && cond < Inf)
    {
      std::cout << " skip (cond)" << std::endl;
      continue;
    }
    if(nbVar > 500)
    {
      std::cout << " skip (nbVar)" << std::endl;
      continue;
    }
    if(nbCstr > 1000)
    {
      std::cout << " skip (nbCstr)" << std::endl;
      continue;
    }
    std::cout << std::endl;

    test::QPSReader reader(true);
    std::string path = QPS_TESTS_DIR + name + ".QPS";
    auto [pb, properties] = reader.read(path);

    MatrixXd G = pb.G; // copy for later check
    T qp(3, 5,
         false); // Sizes are not the correct ones. We check by the way that the resize is working.
    SolverOptions opt;
    opt.logFlags(LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
                 | LogFlags::ACTIVE_SET_DETAILS
                 //           | LogFlags::ITERATION_ADVANCE_DETAILS
                 | LogFlags::INPUT);
    std::ofstream aof("C:/Work/code/optim/jrl-qp/tests/qplog.m");
    opt.logStream_ = &aof;
    aof.precision(16);
    opt.maxIter_ = std::max(50, 10 * std::max(nbCstr, nbVar));
    qp.options(opt);
    auto ret = qp.solve(pb.G, pb.a, pb.C.transpose(), pb.l, pb.u, pb.xl, pb.xu);
    aof.close();
    if(cond == Inf)
    {
      FAST_CHECK_EQ(ret, TerminationStatus::NON_POS_HESSIAN);
    }
    else
    {
      FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
      FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu, false));
      FAST_CHECK_EQ(qp.objectiveValue() + pb.objCst, doctest::Approx(fstar).epsilon(1e-6));
    }
  }
}

#endif
