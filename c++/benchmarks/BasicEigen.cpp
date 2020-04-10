/* Copyright 2020 CNRS-AIST JRL
 */


#include <Eigen/Core>

#include <benchmark/benchmark.h>

#include "common.h"

using namespace Eigen;


// A = B
static void BM_Copy_MatrixXd(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target = source;
}
MAT_BENCHMARK(BM_Copy_MatrixXd);

// A = B^T
static void BM_Copy_MatrixXd_Source_Transpose(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target = source.transpose();
}
MAT_BENCHMARK(BM_Copy_MatrixXd_Source_Transpose);

// A^T = B
static void BM_Copy_MatrixXd_Target_Transpose(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.transpose() = source;
}
MAT_BENCHMARK(BM_Copy_MatrixXd_Target_Transpose);

// A += B
static void BM_Add_MatrixXd(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target += source;
}
MAT_BENCHMARK(BM_Add_MatrixXd);

// A += B^T
static void BM_Add_MatrixXd_Source_Transpose(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target += source.transpose();
}
MAT_BENCHMARK(BM_Add_MatrixXd_Source_Transpose);

// A += B^T
static void BM_Add_MatrixXd_Target_Transpose(benchmark::State& state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.transpose() += source;
}
MAT_BENCHMARK(BM_Add_MatrixXd_Target_Transpose);

// y = A*x
static void BM_Mult_VectorXd(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  for (auto _ : state)
    y.noalias() = A*x;
}
MAT_BENCHMARK(BM_Mult_VectorXd);

// row = (A*x)^T
static void BM_Mult_VectorXd_Source_Transpose(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  MatrixXd y(2,state.range(0));

  for (auto _ : state)
    y.row(1).noalias() = (A * x).transpose();
}
MAT_BENCHMARK(BM_Mult_VectorXd_Source_Transpose);

// row^T = A*x
static void BM_Mult_VectorXd_Target_Transpose(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  MatrixXd y(2, state.range(0));

  for (auto _ : state)
    y.row(1).transpose().noalias() = A * x;
}
MAT_BENCHMARK(BM_Mult_VectorXd_Target_Transpose);

// C = A*B
static void BM_Mult_MatrixXd(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.noalias() = A * B;
}
MAT_BENCHMARK(BM_Mult_MatrixXd);

// C = A^T*B
static void BM_Mult_MatrixXd_AT(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.noalias() = A.transpose() * B;
}
MAT_BENCHMARK(BM_Mult_MatrixXd_AT);

// C = A*B^T
static void BM_Mult_MatrixXd_BT(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.noalias() = A * B.transpose();
}
MAT_BENCHMARK(BM_Mult_MatrixXd_BT);

// C = A^T*B^T
static void BM_Mult_MatrixXd_AT_BT(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.noalias() = A.transpose() * B.transpose();
}
MAT_BENCHMARK(BM_Mult_MatrixXd_AT_BT);

// C = A*B
static void BM_Mult_MatrixXd_Target_Transpose(benchmark::State& state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for (auto _ : state)
    target.transpose().noalias() = A * B;
}
MAT_BENCHMARK(BM_Mult_MatrixXd_Target_Transpose);


BENCHMARK_MAIN();
