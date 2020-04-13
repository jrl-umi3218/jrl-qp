/* Copyright 2020 CNRS-AIST JRL
 */

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <benchmark/benchmark.h>

#include "common.h"

using namespace Eigen;

static void BM_TriangularSolve_NoInversePrecompute(benchmark::State& state)
{
  MatrixXd R = MatrixXd::Random(50, 50);
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(A);
  VectorXd b = VectorXd::Random(50);
  for (auto _ : state)
  {
    for (int i = 0; i < state.range(0); ++i)
    {
      llt.matrixL().solveInPlace(b);
    }
  }
}
//BENCHMARK(BM_TriangularSolve_NoInversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularSolve_InversePrecompute(benchmark::State& state)
{
  MatrixXd R = MatrixXd::Random(50, 50);
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(A);
  VectorXd b = VectorXd::Random(50);
  VectorXd x = VectorXd::Random(50);
  MatrixXd invA(50, 50);
  for (auto _ : state)
  {
    invA.setIdentity();
    llt.matrixL().solveInPlace(invA);
    for (int i = 0; i < state.range(0); ++i)
    {
      benchmark::DoNotOptimize(x.noalias() = invA.triangularView<Lower>() * b);
    }
  }
}
BENCHMARK(BM_TriangularSolve_InversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);

static void BM_TriangularSolve_SpecialInversePrecompute(benchmark::State& state)
{
  MatrixXd R = MatrixXd::Random(50, 50);
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(A);
  VectorXd b = VectorXd::Random(50);
  VectorXd x = VectorXd::Random(50);
  MatrixXd invA(50, 50);
  for (auto _ : state)
  {
    
    llt.matrixL().solveInPlace(invA);
    for (int i = 0; i < state.range(0); ++i)
    {
      benchmark::DoNotOptimize(x.noalias() = invA.triangularView<Lower>() * b);
    }
  }
}
BENCHMARK(BM_TriangularSolve_SpecialInversePrecompute)->Apply(testSizes)->Unit(benchmark::kMicrosecond);


BENCHMARK_MAIN();
