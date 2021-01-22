/* Copyright 2020 CNRS-AIST JRL */

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>

#include <benchmark/benchmark.h>

#include "common.h"

using namespace Eigen;

static void BM_TriangularSolve_NoInversePrecompute(benchmark::State & state)
{
  MatrixXd R = MatrixXd::Random(50, 50);
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(A);
  VectorXd b = VectorXd::Random(50);
  for(auto _ : state)
  {
    for(int i = 0; i < state.range(0); ++i)
    {
      llt.matrixL().solveInPlace(b);
    }
  }
}
// BENCHMARK(BM_TriangularSolve_NoInversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularSolve_InversePrecompute(benchmark::State & state)
{
  MatrixXd R = MatrixXd::Random(50, 50);
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(A);
  VectorXd b = VectorXd::Random(50);
  VectorXd x = VectorXd::Random(50);
  MatrixXd invA(50, 50);
  for(auto _ : state)
  {
    invA.setIdentity();
    llt.matrixL().solveInPlace(invA);
    for(int i = 0; i < state.range(0); ++i)
    {
      benchmark::DoNotOptimize(x.noalias() = invA.triangularView<Lower>() * b);
    }
  }
}
//BENCHMARK(BM_TriangularSolve_InversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_AtOnce(benchmark::State & state)
{
  const int n = static_cast<int>(state.range(0));
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  for(auto _ : state)
  {
    invA.setIdentity();
    A.template triangularView<Eigen::Lower>().solveInPlace(invA);
  }
}
//BENCHMARK(BM_TriangularInverse_AtOnce)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_Transpose(benchmark::State & state)
{
  const int n = static_cast<int>(state.range(0));
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  for(auto _ : state)
  {
    invA.setIdentity();
    A.template triangularView<Eigen::Lower>().transpose().solveInPlace(invA);
  }
}
BENCHMARK(BM_TriangularInverse_Transpose)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_Transpose_ByHand(benchmark::State & state)
{
  const int n = static_cast<int>(state.range(0));
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  for(auto _ : state)
  {
    //for(int i=n-1; i>=0; --i)
    //{
    //  invA(i, i) = 1. / A(i, i);
    //  invA.col(i).head(i - 1) = -invA(i, i) * A.row(i).head(i - 1).transpose();
    //  //invA.col(i).tail(n - i - 1).setZero();
    //}
    invA(0, 0) = 1. / A(0, 0);
    for(int i=0; i<n-1; ++i)
    {
      int i1 = i + 1;
      invA(i1, i1) = 1. / A(i1, i1);
      invA.col(i1).head(i1 - 1) = -invA(i1, i1) * A.row(i1).head(i1 - 1).transpose();
      invA.col(i + 1).head(i) = invA.topLeftCorner(i, i) * invA.col(i + 1).head(i);
    }
    A.template triangularView<Eigen::Lower>().transpose().solveInPlace(invA);
  }
}
BENCHMARK(BM_TriangularInverse_Transpose_ByHand)->Apply(testSizes)->Unit(benchmark::kMicrosecond);


static void BM_TriangularInverse_ByCol(benchmark::State & state)
{
  const int n = static_cast<int>(state.range(0));
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  VectorXd e(n);
  for(auto _ : state)
  {
    e.setZero();
    for(int i = 0; i < n; ++i)
    {
      e[i] = 1;
      invA.col(i) = A.template triangularView<Eigen::Lower>().solve(e);
      e[i] = 0;
    }
  }
}
//BENCHMARK(BM_TriangularInverse_ByCol)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_PSD_Solve_Overhead(benchmark::State & state)
{
  Matrix<double, 6, 6> R;
  R.setRandom();
  for(auto _ : state)
  {
    Matrix<double, 6, 6> A = R.transpose() * R;
  }
}
//BENCHMARK(BM_PSD_Solve_Overhead);

static void BM_PSD_Solve(benchmark::State & state)
{
  Matrix<double, 6, 6> R;
  R.setRandom();
  Matrix<double, 6, 1> b;
  Matrix<double, 6, 1> x;
  b.setRandom();
  for(auto _ : state)
  {
    Matrix<double, 6, 6> A = R.transpose() * R;
    x = A.llt().solve(b);
  }
}
//BENCHMARK(BM_PSD_Solve);

static void BM_PSD_SolveInPlace(benchmark::State & state)
{
  Matrix<double, 6, 6> R;
  R.setRandom();
  Matrix<double, 6, 1> b;
  Matrix<double, 6, 1> x;
  b.setRandom();
  for(auto _ : state)
  {
    Matrix<double, 6, 6> A = R.transpose() * R;
    A.llt().solveInPlace(b);
  }
}
//BENCHMARK(BM_PSD_SolveInPlace);

static void BM_PSD_AllInPlace(benchmark::State & state)
{
  Matrix<double, 6, 6> R;
  R.setRandom();
  Matrix<double, 6, 1> b;
  Matrix<double, 6, 1> x;
  b.setRandom();
  for(auto _ : state)
  {
    Matrix<double, 6, 6> A = R.transpose() * R;
    LLT<Ref<Matrix<double, 6, 6>>> llt(A);
    llt.solveInPlace(b);
  }
}
//BENCHMARK(BM_PSD_AllInPlace);

static void BM_PSD_AllInPlaceByHand(benchmark::State & state)
{
  Matrix<double, 6, 6> R;
  R.setRandom();
  Matrix<double, 6, 1> b;
  Matrix<double, 6, 1> x;
  b.setRandom();
  for(auto _ : state)
  {
    Matrix<double, 6, 6> A = R.transpose() * R;
    Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(A);
    auto L = A.template triangularView<Eigen::Lower>();
    L.solveInPlace(b);
    L.transpose().solveInPlace(b);
  }
}
//BENCHMARK(BM_PSD_AllInPlaceByHand);

BENCHMARK_MAIN();
