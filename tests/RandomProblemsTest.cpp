#include <array>
#include <Eigen/LU>
#include <Eigen/QR>

#include <jrl-qp/test/kkt.h>
#include <jrl-qp/test/randomMatrices.h>
#include <jrl-qp/test/randomProblems.h>


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace jrl::qp::test;

//TEST_CASE("Random orthogonal matrices")
//{
//  const int N = 1000;
//  int positive = 0;
//  for (int i = 0; i < N; ++i)
//  {
//    int s = 5 + (10 * i) / N;
//    Eigen::MatrixXd Q = randOrtho(s);
//    FAST_REQUIRE_EQ(Q.rows(), Q.cols());
//    FAST_CHECK_EQ(Q.cols(), s);
//
//    double det = Q.lu().determinant();
//    FAST_CHECK_EQ(std::abs(det), doctest::Approx(1).epsilon(1e-8));
//    if (det > 0) ++positive;
//    
//    FAST_CHECK_UNARY((Q.transpose() * Q).isIdentity(1e-8));
//    FAST_CHECK_UNARY((Q * Q.transpose()).isIdentity(1e-8));
//  }
//  FAST_CHECK_UNARY(429 <= positive && positive <= 571); //this can fails in less than 0.001% of cases.
//
//  for (int i = 0; i < N; ++i)
//  {
//    int s = 5 + (10 * i) / N;
//    Eigen::MatrixXd Q = randOrtho(s, true);
//    double det = Q.lu().determinant();
//    FAST_CHECK_EQ(det, doctest::Approx(1).epsilon(1e-8));
//  }
//}
//
//TEST_CASE("Random rank deficient matrices")
//{
//  std::array<Eigen::Index, 5> sizes = {10,25,50,75,100};
//  Eigen::Index n = 0;
//  double mean = 0;
//  double var = 0;
//  Precompute the required size for storing all coefficients
//  for (auto rows : sizes)
//  {
//    for (auto cols : sizes)
//    {
//      for (auto rank : sizes)
//      {
//        if (rank <= rows && rank <= cols)
//          n += rows * cols;
//      }
//    }
//  }
//  std::vector<double> val;
//  val.reserve(n);
//
//  Testing the rank of generated matrices
//  for (auto rows: sizes)
//  {
//    for (auto cols: sizes)
//    {
//      for (auto rank: sizes)
//      {
//        if (rank > rows || rank > cols) //we skip when rank > min(rows,cols)
//          continue;
//        Eigen::MatrixXd M = randn(rows, cols, rank);
//        FAST_CHECK_EQ(M.colPivHouseholderQr().rank(), rank);
//        std::copy(M.data(), M.data() + rows * cols, std::back_inserter(val));
//      }
//    }
//  }
//
//  Computing and testing mean and variance
//  for (int i = 0; i < n; ++i)
//  {
//    mean += val[i];
//  }
//  mean /= n;
//  for (int i = 0; i < n; ++i)
//  {
//    var += (val[i] - mean) * (val[i] - mean);
//  }
//  var /= n;
//  
//  FAST_CHECK_EQ(std::abs(mean), doctest::Approx(0).epsilon(1e-2));
//  FAST_CHECK_EQ(var, doctest::Approx(1).epsilon(2.5e-2));
//}
//
//TEST_CASE("Random dependent matrices")
//{
//  std::vector<std::array<Eigen::Index, 6>> params = {
//    { 15, 5, 4, 7, 7, 9 },
//    { 15, 5, 4, 7, 5, 9 },
//    { 10, 5, 4, 7, 7, 9 },
//    { 7, 8, 4, 12, 6, 7 },
//    { 7, 8, 4, 12, 6, 6 } };
//  for (const auto& p: params)
//  {
//    auto [cols, rowsA, rankA, rowsB, rankB, rankAB] = p;
//    auto [A, B] = randDependent(cols, rowsA, rankA, rowsB, rankB, rankAB);
//    FAST_REQUIRE_EQ(A.rows(), rowsA);
//    FAST_REQUIRE_EQ(A.cols(), cols);
//    FAST_REQUIRE_EQ(B.rows(), rowsB);
//    FAST_REQUIRE_EQ(B.cols(), cols);
//    Eigen::MatrixXd M(rowsA+rowsB, cols); M << A, B;
//    FAST_CHECK_EQ(A.colPivHouseholderQr().rank(), rankA);
//    FAST_CHECK_EQ(B.colPivHouseholderQr().rank(), rankB);
//    FAST_CHECK_EQ(M.colPivHouseholderQr().rank(), rankAB);
//  }
//}

void testRandomProblem(const RandomLeastSquare& pb)
{
  FAST_CHECK_UNARY(pb.wellFormed());
  Eigen::VectorXd mult(pb.l.size() + pb.f.size() + pb.xl.size());
  mult << pb.lambdaEq, pb.lambdaIneq, pb.lambdaBnd;
  FAST_CHECK_UNARY(testKKT(pb.x, mult, pb));
}

TEST_CASE("Random Least Squares")
{
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 3)));
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 3).nEq(2)));
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 0).nEq(2)));
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 3).nIneq(5)));
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 3).nIneq(5).nStrongActIneq(2)));
  testRandomProblem(randomProblem(ProblemCharacteristics(5, 3).nIneq(5).nStrongActIneq(4)));
  testRandomProblem(randomProblem({ 5, 3, 2, 5, 3, 0, 2, 0, 0, 0, false, false, false }));
  testRandomProblem(randomProblem({ 5, 3, 1, 5, 3, 0, 1, 0, 0, 0, false, false, false }));
}

