/* Copyright 2020-2021 CNRS-AIST JRL */

#include <iostream>
#include <numeric>

#include <Eigen/Cholesky>

#include <jrl-qp/decomposition/blockArrowLLT.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace jrl::qp;

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

TEST_CASE("Block down arrow LLT")
{
  MatrixXd A = blockDiagAndOneColDiagRandom({3, 5, 2, 3}, false);
  MatrixXd H1 = A.transpose() * A;
  MatrixXd H2 = H1;

  // Packing as vector
  std::vector<MatrixRef> D = {H1.block(0, 0, 3, 3), H1.block(3, 3, 5, 5), H1.block(8, 8, 2, 2), H1.block(10, 10, 3, 3)};
  std::vector<MatrixRef> S = {H1.block(10, 0, 3, 3), H1.block(10, 3, 3, 5), H1.block(10, 8, 3, 2)};

  // Decomposition
  bool ret = decomposition::blockArrowLLT(D, S);
  Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(H2);

  FAST_CHECK_UNARY(ret);
  FAST_CHECK_UNARY(H1.isApprox(H2, 1e-8));

  //Solve
  {
    MatrixXd B = MatrixXd::Zero(13, 5);
    B.middleRows(4, 5).setRandom();

    MatrixXd B0 = B;
    auto L2 = H2.template triangularView<Eigen::Lower>();
    L2.solveInPlace(B0);

    MatrixXd B1 = B;
    decomposition::blockArrowLSolve(D, S, false, B1);
    FAST_CHECK_UNARY(B1.isApprox(B0, 1e-8));

    MatrixXd B2 = B;
    decomposition::blockArrowLSolve(D, S, false, B2, 4);
    FAST_CHECK_UNARY(B2.isApprox(B0, 1e-8));
  }

  // Solve tranpose
  {
    MatrixXd B = MatrixXd::Zero(13, 5);
    B.middleRows(4, 5).setRandom();

    MatrixXd B0 = B;
    auto L2 = H2.template triangularView<Eigen::Lower>();
    L2.transpose().solveInPlace(B0);

    MatrixXd B1 = B;
    decomposition::blockArrowLTransposeSolve(D, S, false, B1);
    FAST_CHECK_UNARY(B1.isApprox(B0, 1e-8));

    MatrixXd B2 = B;
    decomposition::blockArrowLTransposeSolve(D, S, false, B2, 9);
    FAST_CHECK_UNARY(B2.isApprox(B0, 1e-8));
  }
}

TEST_CASE("Block up arrow LLT")
{
  MatrixXd A = blockDiagAndOneColDiagRandom({3, 5, 2, 3}, true);
  MatrixXd H1 = A.transpose() * A;
  MatrixXd P(13, 13);
  P << MatrixXd::Zero(3, 10), MatrixXd::Identity(3, 3), MatrixXd::Identity(10, 10), MatrixXd::Zero(10, 3);
  MatrixXd H2 = P.transpose() * H1 * P;

  // Packing as vector
  std::vector<MatrixRef> D = {H1.block(0, 0, 3, 3), H1.block(3, 3, 5, 5), H1.block(8, 8, 2, 2), H1.block(10, 10, 3, 3)};
  std::vector<MatrixRef> S = {H1.block(3, 0, 5, 3), H1.block(8, 0, 2, 3), H1.block(10, 0, 3, 3)};

  // Decomposition
  bool ret = decomposition::blockArrowLLT(D, S, true);
  Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(H2);

  FAST_CHECK_UNARY(ret);
  FAST_CHECK_UNARY(H1.bottomRightCorner(10,10).isApprox(H2.topLeftCorner(10,10), 1e-8));
  FAST_CHECK_UNARY(H1.topLeftCorner(3, 3).isApprox(H2.bottomRightCorner(3, 3), 1e-8));
  FAST_CHECK_UNARY(H1.bottomLeftCorner(10, 3).transpose().isApprox(H2.bottomLeftCorner(3, 10), 1e-8));

  // Solve
  {
    MatrixXd B = MatrixXd::Zero(13, 5);
    B.middleRows(4, 5).setRandom();

    MatrixXd B0 = B;
    auto L2 = H2.template triangularView<Eigen::Lower>();
    L2.solveInPlace(B0);

    MatrixXd B1 = B;
    decomposition::blockArrowLSolve(D, S, true, B1);
    
    FAST_CHECK_UNARY(B1.isApprox(B0, 1e-8));

    MatrixXd B2 = B;
    decomposition::blockArrowLSolve(D, S, true, B2, 4);
    FAST_CHECK_UNARY(B2.isApprox(B0, 1e-8));
  }

  // Solve tranpose
  {
    MatrixXd B = MatrixXd::Zero(13, 5);
    B.middleRows(4, 5).setRandom();

    MatrixXd B0 = B;
    auto L2 = H2.template triangularView<Eigen::Lower>();
    L2.transpose().solveInPlace(B0);

    MatrixXd B1 = B;
    decomposition::blockArrowLTransposeSolve(D, S, true, B1);
    FAST_CHECK_UNARY(B1.isApprox(B0, 1e-8));

    MatrixXd B2 = B;
    decomposition::blockArrowLTransposeSolve(D, S, true, B2, 9);
    std::cout << B0 << "\n\n";
    std::cout << B2 << "\n\n";
    FAST_CHECK_UNARY(B2.isApprox(B0, 1e-8));
  }
}