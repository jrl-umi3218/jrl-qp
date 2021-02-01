/* Copyright 2020-2021 CNRS-AIST JRL */

#include <iostream>
#include <numeric>

#include <Eigen/Cholesky>

#include <jrl-qp/decomposition/triBlockDiagLLT.h>
#include <jrl-qp/test/randomMatrices.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

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

TEST_CASE("Block tri-diagonal LLT")
{
  MatrixXd A = biBlockDiagRandom({3, 5, 2, 3});
  MatrixXd H1 = A * A.transpose();
  MatrixXd H2 = H1;

  // Packing as vector
  std::vector<MatrixRef> D = {H1.block(0, 0, 3, 3), H1.block(3, 3, 5, 5), H1.block(8, 8, 2, 2), H1.block(10, 10, 3, 3)};
  std::vector<MatrixRef> S = {H1.block(3, 0, 5, 3), H1.block(8, 3, 2, 5), H1.block(10, 8, 3, 2)};

  bool ret = decomposition::triBlockDiagLLT(D, S);
  Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(H2);

  std::cout << H2 - H1 << std::endl;
  FAST_CHECK_UNARY(ret);
  FAST_CHECK_UNARY(H1.isApprox(H2,1e-8));
}