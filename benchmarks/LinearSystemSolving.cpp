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
BENCHMARK(BM_TriangularSolve_InversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_AtOnce(benchmark::State & state)
{
  const int n = state.range(0);
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  for(auto _ : state)
  {
    invA.setIdentity();
    A.template triangularView<Eigen::Lower>().solveInPlace(invA);
  }
}
BENCHMARK(BM_TriangularInverse_AtOnce)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_Transpose(benchmark::State & state)
{
  const int n = state.range(0);
  MatrixXd A = MatrixXd::Random(n, n);
  MatrixXd invA(state.range(0), state.range(0));
  for(auto _ : state)
  {
    invA.setIdentity();
    A.template triangularView<Eigen::Lower>().transpose().solveInPlace(invA);
  }
}
BENCHMARK(BM_TriangularInverse_Transpose)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularInverse_ByCol(benchmark::State & state)
{
  const int n = state.range(0);
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
BENCHMARK(BM_TriangularInverse_ByCol)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
