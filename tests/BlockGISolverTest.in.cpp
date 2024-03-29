/* Copyright 2020-2021 CNRS-AIST JRL */

#include <chrono>
#include <iostream>
#include <numeric>

#include <Eigen/Cholesky>

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/BlockGISolver.h>
#include <jrl-qp/test/randomMatrices.h>
#include <jrl-qp/test/randomProblems.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include "IKmatReader.h"

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::experimental;
using namespace jrl::qp::structured;
using namespace jrl::qp::test;

#define WITH_BENCH1 0
#define WITH_BENCH2 1

MatrixXd biBlockDiagRandom(const std::vector<int> & n)
{
  int s = std::accumulate(n.begin(), n.end(), 0);
  MatrixXd B = MatrixXd::Zero(s, s);

  int k = 0;
  for(size_t i = 0; i < n.size() - 1; ++i)
  {
    int ni = n[i];
    B.block(k, k, ni, ni).setRandom();
    B.block(k + ni, k, n[i + 1], ni).setRandom();
    k += ni;
  }
  int nb = n.back();
  B.bottomRightCorner(nb, nb).setRandom();

  return B;
}

MatrixXd blockDiagAndOneColDiagRandom(const std::vector<int> & n, bool first)
{
  int s = std::accumulate(n.begin(), n.end(), 0);
  MatrixXd B = MatrixXd::Zero(s, s);

  int k = 0;
  for(size_t i = 0; i < n.size(); ++i)
  {
    int ni = n[i];
    B.block(k, k, ni, ni).setRandom();
    if(first)
      B.block(k, 0, ni, n[0]).setRandom();
    else
      B.block(k, s - n.back(), ni, n.back()).setRandom();
    k += ni;
  }
  int nb = n.back();
  B.bottomRightCorner(nb, nb).setRandom();

  return B;
}

TEST_CASE("Small problem tridiag obj, ineq only")
{
  std::vector n = {3, 5, 2, 3};
  std::vector mi = {3, 3, 3, 3};
  MatrixXd A = biBlockDiagRandom(n);
  MatrixXd GDense = A * A.transpose();
  MatrixXd G0 = GDense;
  // Packing as vector
  std::vector<MatrixRef> D = {GDense.block(0, 0, 3, 3), GDense.block(3, 3, 5, 5), GDense.block(8, 8, 2, 2),
                              GDense.block(10, 10, 3, 3)};
  std::vector<MatrixRef> S = {GDense.block(3, 0, 5, 3), GDense.block(8, 3, 2, 5), GDense.block(10, 8, 3, 2)};
  StructuredG G(StructuredG::Type::TriBlockDiagonal, D, S);

  VectorXd a = VectorXd::Random(13);

  MatrixXd C0 = MatrixXd::Zero(13, 12);
  std::vector<MatrixConstRef> Cs;
  VectorXd l(12), u(12);
  VectorXd xl(0), xu(0);
  int r = 0;
  int c = 0;
  for(int i = 0; i < 4; ++i)
  {
    auto pb = randomProblem(ProblemCharacteristics(n[i], 0, 0, mi[i]).doubleSidedIneq(true));
    C0.block(r, c, n[i], mi[i]) = pb.C.transpose();
    Cs.push_back(C0.block(r, c, n[i], mi[i]));
    l.segment(c, mi[i]) = pb.l;
    u.segment(c, mi[i]) = pb.u;
    r += n[i];
    c += mi[i];
  }
  StructuredC C(Cs);

  GoldfarbIdnaniSolver solverD(13, 12, false);
  // SolverOptions optD;
  // optD.logFlags(LogFlags::INPUT | LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
  //              | LogFlags::ACTIVE_SET_DETAILS);
  // solverD.options(optD);
  auto retD = solverD.solve(G0, a, C0, l, u, xl, xu);

  BlockGISolver solverB(13, 12, false);
  // SolverOptions optB;
  // optB.logFlags(LogFlags::INPUT | LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
  //              | LogFlags::ACTIVE_SET_DETAILS);
  // solverB.options(optB);
  auto retB = solverB.solve(G, a, C, l, u, xl, xu);

  FAST_CHECK_EQ(retB, retD);
  FAST_CHECK_UNARY(solverB.solution().isApprox(solverD.solution(), 1e-8));
}

TEST_CASE("Small problem arrow up obj, ineq only")
{
  std::vector n = {3, 5, 2, 3};
  std::vector mi = {3, 3, 3, 3};
  MatrixXd A = blockDiagAndOneColDiagRandom(n, true);
  MatrixXd GDense = A.transpose() * A;
  MatrixXd G0 = GDense;
  // Packing as vector
  std::vector<MatrixRef> D = {GDense.block(0, 0, 3, 3), GDense.block(3, 3, 5, 5), GDense.block(8, 8, 2, 2),
                              GDense.block(10, 10, 3, 3)};
  std::vector<MatrixRef> S = {GDense.block(3, 0, 5, 3), GDense.block(8, 0, 2, 3), GDense.block(10, 0, 3, 3)};
  StructuredG G(StructuredG::Type::BlockArrowUp, D, S);

  VectorXd a = VectorXd::Random(13);

  MatrixXd C0 = MatrixXd::Zero(13, 12);
  std::vector<MatrixConstRef> Cs;
  VectorXd l(12), u(12);
  VectorXd xl(0), xu(0);
  int r = 0;
  int c = 0;
  for(int i = 0; i < 4; ++i)
  {
    auto pb = randomProblem(ProblemCharacteristics(n[i], 0, 0, mi[i]).doubleSidedIneq(true));
    C0.block(r, c, n[i], mi[i]) = pb.C.transpose();
    Cs.push_back(C0.block(r, c, n[i], mi[i]));
    l.segment(c, mi[i]) = pb.l;
    u.segment(c, mi[i]) = pb.u;
    r += n[i];
    c += mi[i];
  }
  StructuredC C(Cs);

  GoldfarbIdnaniSolver solverD(13, 12, false);
  // SolverOptions optD;
  // optD.logFlags(LogFlags::INPUT | LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
  //            | LogFlags::ACTIVE_SET_DETAILS);
  // solverD.options(optD);
  auto retD = solverD.solve(G0, a, C0, l, u, xl, xu);

  BlockGISolver solverB(13, 12, false);
  // SolverOptions optB;
  // optB.logFlags(LogFlags::INPUT | LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
  //            | LogFlags::ACTIVE_SET_DETAILS);
  // solverB.options(optB);
  auto retB = solverB.solve(G, a, C, l, u, xl, xu);

  FAST_CHECK_EQ(retB, retD);
  FAST_CHECK_UNARY(solverB.solution().isApprox(solverD.solution(), 1e-8));
}

TEST_CASE("Sequential IK")
{
  const std::string dir = "@CMAKE_CURRENT_BINARY_DIR@";
  MatrixXd G0 = readMat(dir + "/MultiIK/triBlockDiag_G.txt");
  MatrixXd C = readMat(dir + "/MultiIK/triBlockDiag_C.txt").transpose();
  VectorXd a = readMat(dir + "/MultiIK/triBlockDiag_a.txt");
  VectorXd u = readMat(dir + "/MultiIK/triBlockDiag_u.txt");
  VectorXd l = VectorXd::Constant(u.size(), -std::numeric_limits<double>::infinity());
  VectorXd x = readMat(dir + "/MultiIK/triBlockDiag_sol.txt");

  MatrixXd GD = G0;
  GoldfarbIdnaniSolver solverD(static_cast<int>(G0.rows()), static_cast<int>(C.cols()), false);
  auto retD = solverD.solve(GD, a, C, l, u, VectorXd(0), VectorXd(0));
  VectorXd xd0 = solverD.solution();

  FAST_CHECK_EQ(retD, TerminationStatus::SUCCESS);
  FAST_CHECK_LE((x - xd0).lpNorm<Infinity>(), 1e-4); // small precision because files don't have high precision

  // scan C
  const int nDofs = 43;
  std::vector<int> nbCstr;
  for(int i = 0; i < C.cols(); ++i)
  {
    int j = 0;
    while(C(j, i) == 0) ++j;

    if(j >= static_cast<int>(nbCstr.size()) * nDofs)
    {
      nbCstr.push_back(1);
    }
    else
    {
      ++nbCstr.back();
    }
  }

  MatrixXd GBD = G0;
  std::vector<MatrixRef> D;
  std::vector<MatrixRef> S;
  std::vector<MatrixConstRef> Ci;
  int k = 0;
  for(int i = 0; i < 9; ++i)
  {
    D.push_back(GBD.block(i * nDofs, i * nDofs, nDofs, nDofs));
    if(i > 0) S.push_back(GBD.block(i * nDofs, (i - 1) * nDofs, nDofs, nDofs));
    Ci.push_back(C.block(i * nDofs, k, nDofs, nbCstr[i]));
    k += nbCstr[i];
  }

  StructuredG GB(StructuredG::Type::TriBlockDiagonal, D, S);
  StructuredC CB(Ci);
  BlockGISolver solverB(9 * nDofs, k, false);
  auto retB = solverB.solve(GB, a, CB, l, u, VectorXd(0), VectorXd(0));
  VectorXd xb0 = solverB.solution();
  FAST_CHECK_EQ(retB, TerminationStatus::SUCCESS);
  FAST_CHECK_UNARY(xb0.isApprox(xd0, 1e-8));

#ifdef NDEBUG
#  if WITH_BENCH1
  const int nTestD = 100;
  std::vector<MatrixXd> GDs;
  std::fill_n(std::back_inserter(GDs), nTestD, G0);

  const int nTestB = 1000;
  std::vector<MatrixXd> GBDs;
  std::fill_n(std::back_inserter(GBDs), nTestB, G0);
  std::vector<StructuredG> GBs;
  for(int j = 0; j < nTestB; ++j)
  {
    std::vector<MatrixRef> D;
    std::vector<MatrixRef> S;
    for(int i = 0; i < 9; ++i)
    {
      D.push_back(GBDs[j].block(i * nDofs, i * nDofs, nDofs, nDofs));
      if(i > 0) S.push_back(GBDs[j].block(i * nDofs, (i - 1) * nDofs, nDofs, nDofs));
    }
    GBs.emplace_back(StructuredG::Type::TriBlockDiagonal, D, S);
  }

  int dummy = 0;
  auto t0 = std::chrono::steady_clock::now();
  for(int i = 0; i < nTestD; ++i)
  {
    auto retDi = solverD.solve(GDs[i], a, C, l, u, VectorXd(0), VectorXd(0));
    dummy += static_cast<int>(retDi);
  }

  auto t1 = std::chrono::steady_clock::now();
  for(int i = 0; i < nTestB; ++i)
  {
    auto retBi = solverB.solve(GBs[i], a, CB, l, u, VectorXd(0), VectorXd(0));
    dummy += static_cast<int>(retBi);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << dummy << std::endl;
  std::cout << "Vanilla   : " << std::chrono::duration<double>(t1 - t0).count() / nTestD * 1000 << "ms" << std::endl;
  std::cout << "Structured: " << std::chrono::duration<double>(t2 - t1).count() / nTestB * 1000 << "ms" << std::endl;
#  endif
#endif
}

TEST_CASE("Simultaneous IK")
{
  const std::string dir = "@CMAKE_CURRENT_BINARY_DIR@";
  auto [G0, a, E, f, C0, u, xl, xu] = readIKPbFile(dir + "/MultiIK/arrowAllData.txt");
  MatrixXd C = C0.transpose();
  VectorXd l = VectorXd::Constant(u.size(), -std::numeric_limits<double>::infinity());

  MatrixXd GD = G0;
  GoldfarbIdnaniSolver solverD(static_cast<int>(G0.rows()), static_cast<int>(C.cols()), true);
  auto retD = solverD.solve(GD, a, C, l, u, xl, xu);
  FAST_CHECK_EQ(retD, TerminationStatus::SUCCESS);
  VectorXd xd0 = solverD.solution();

  MatrixXd GBD = G0;
  std::vector<MatrixRef> D;
  std::vector<MatrixRef> S;
  std::vector<MatrixConstRef> Ci;
  const int nDofs = 42;
  int nbCstr[] = {5, 5, 5, 5, 5};
  int k = 0;
  for(int i = 0; i < 5; ++i)
  {
    D.push_back(GBD.block(i * nDofs, i * nDofs, nDofs, nDofs));
    if(i > 0) S.push_back(GBD.block(i * nDofs, 0, nDofs, nDofs));
    Ci.push_back(C.block(i * nDofs, k, nDofs, nbCstr[i]));
    k += nbCstr[i];
  }

  StructuredG GB(StructuredG::Type::BlockArrowUp, D, S);
  StructuredC CB(Ci);
  BlockGISolver solverB(5 * nDofs, k, true);
  auto retB = solverB.solve(GB, a, CB, l, u, xl, xu);
  VectorXd xb0 = solverB.solution();
  FAST_CHECK_EQ(retB, TerminationStatus::SUCCESS);
  FAST_CHECK_UNARY(xb0.isApprox(xd0, 1e-8));

#ifdef NDEBUG
#  if WITH_BENCH2
  const int nTestD = 100;
  std::vector<MatrixXd> GDs;
  std::fill_n(std::back_inserter(GDs), nTestD, G0);

  const int nTestB = 1000;
  std::vector<MatrixXd> GBDs;
  std::fill_n(std::back_inserter(GBDs), nTestB, G0);
  std::vector<StructuredG> GBs;
  for(int j = 0; j < nTestB; ++j)
  {
    std::vector<MatrixRef> D;
    std::vector<MatrixRef> S;
    for(int i = 0; i < 5; ++i)
    {
      D.push_back(GBDs[j].block(i * nDofs, i * nDofs, nDofs, nDofs));
      if(i > 0) S.push_back(GBDs[j].block(i * nDofs, 0, nDofs, nDofs));
    }
    GBs.emplace_back(StructuredG::Type::BlockArrowUp, D, S);
  }

  int dummy = 0;
  auto t0 = std::chrono::steady_clock::now();
  for(int i = 0; i < nTestD; ++i)
  {
    auto retDi = solverD.solve(GDs[i], a, C, l, u, xl, xu);
    dummy += static_cast<int>(retDi);
  }

  auto t1 = std::chrono::steady_clock::now();
  for(int i = 0; i < nTestB; ++i)
  {
    auto retBi = solverB.solve(GBs[i], a, CB, l, u, xl, xu);
    dummy += static_cast<int>(retBi);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << dummy << std::endl;
  std::cout << "Vanilla   : " << std::chrono::duration<double>(t1 - t0).count() / nTestD * 1000 << "ms" << std::endl;
  std::cout << "Structured: " << std::chrono::duration<double>(t2 - t1).count() / nTestB * 1000 << "ms" << std::endl;
#  endif
#endif
}
