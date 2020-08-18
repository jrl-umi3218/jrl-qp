/* Copyright 2020 CNRS-AIST JRL */

#include <Eigen/Core>

#include <benchmark/benchmark.h>

#include "common.h"

using namespace Eigen;

// A = B
static void BM_Copy_MatrixXd(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target = source;
}

// A = B^T
static void BM_Copy_MatrixXd_Source_Transpose(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target = source.transpose();
}

// A^T = B
static void BM_Copy_MatrixXd_Target_Transpose(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.transpose() = source;
}

// A += B
static void BM_Add_MatrixXd(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target += source;
}

// A += B^T
static void BM_Add_MatrixXd_Source_Transpose(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target += source.transpose();
}

// A += B^T
static void BM_Add_MatrixXd_Target_Transpose(benchmark::State & state)
{
  MatrixXd source = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.transpose() += source;
}

// y = A*x
static void BM_Mult_VectorXd(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  for(auto _ : state) y.noalias() = A * x;
}

// row = (A*x)^T
static void BM_Mult_VectorXd_Source_Transpose(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  MatrixXd y(2, state.range(0));

  for(auto _ : state) y.row(1).noalias() = (A * x).transpose();
}

// row^T = A*x
static void BM_Mult_VectorXd_Target_Transpose(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  MatrixXd y(2, state.range(0));

  for(auto _ : state) y.row(1).transpose().noalias() = A * x;
}

// y = L*x
static void BM_Mult_VectorXd_Triangular(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  for(auto _ : state) y.noalias() = A.template triangularView<Lower>() * x;
}

// y = L*x
static void BM_Mult_VectorXd_TriangularOptim4(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  const int bsize = 4;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    y.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      y.tail(r).noalias() += A.block(s, s, r, bsize) * x.segment(s, bsize);
      s += bsize;
      r -= bsize;
    }
    y.tail(r).noalias() += A.bottomRightCorner(r, r) * x.tail(r);
  }
}

// y = L*x
static void BM_Mult_VectorXd_TriangularOptim8(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  const int bsize = 8;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    y.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      y.tail(r).noalias() += A.block(s, s, r, bsize) * x.segment(s, bsize);
      s += bsize;
      r -= bsize;
    }
    y.tail(r).noalias() += A.bottomRightCorner(r, r) * x.tail(r);
  }
}

// y = L*x
static void BM_Mult_VectorXd_TriangularOptim16(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  const int bsize = 16;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    y.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      y.tail(r).noalias() += A.block(s, s, r, bsize) * x.segment(s, bsize);
      s += bsize;
      r -= bsize;
    }
    y.tail(r).noalias() += A.bottomRightCorner(r, r) * x.tail(r);
  }
}

// C = A*B
static void BM_Mult_MatrixXd(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.noalias() = A * B;
}

// C = A^T*B
static void BM_Mult_MatrixXd_AT(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.noalias() = A.transpose() * B;
}

// C = A*B^T
static void BM_Mult_MatrixXd_BT(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.noalias() = A * B.transpose();
}

// C = A^T*B^T
static void BM_Mult_MatrixXd_AT_BT(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.noalias() = A.transpose() * B.transpose();
}

// C^T = A*B
static void BM_Mult_MatrixXd_Target_Transpose(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.transpose().noalias() = A * B;
}

// Y = L*X
static void BM_Mult_MatrixXd_Triangular(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd C = MatrixXd::Random(state.range(0), state.range(0));

  for(auto _ : state) C.noalias() = A.template triangularView<Lower>() * B;
}

// y = L*x
static void BM_Mult_MatrixXd_TriangularOptim4(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd C = MatrixXd::Random(state.range(0), state.range(0));

  const int bsize = 4;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    C.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      C.bottomRows(r).noalias() += A.block(s, s, r, bsize) * B.middleRows(s, bsize);
      s += bsize;
      r -= bsize;
    }
    C.bottomRows(r).noalias() += A.bottomRightCorner(r, r) * B.bottomRows(r);
  }
}

// y = L*x
static void BM_Mult_MatrixXd_TriangularOptim8(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd C = MatrixXd::Random(state.range(0), state.range(0));

  const int bsize = 8;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    C.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      C.bottomRows(r).noalias() += A.block(s, s, r, bsize) * B.middleRows(s, bsize);
      s += bsize;
      r -= bsize;
    }
    C.bottomRows(r).noalias() += A.bottomRightCorner(r, r) * B.bottomRows(r);
  }
}

// y = L*x
static void BM_Mult_MatrixXd_TriangularOptim16(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  A.template triangularView<StrictlyUpper>().setZero();
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd C = MatrixXd::Random(state.range(0), state.range(0));

  const int bsize = 16;
  for(auto _ : state)
  {
    int nBlock = A.cols() / bsize;
    C.setZero();
    int s = 0;
    int r = A.cols();
    for(int i = 0; i < nBlock; ++i)
    {
      C.bottomRows(r).noalias() += A.block(s, s, r, bsize) * B.middleRows(s, bsize);
      s += bsize;
      r -= bsize;
    }
    C.bottomRows(r).noalias() += A.bottomRightCorner(r, r) * B.bottomRows(r);
  }
}

// y = D*x
static void BM_Mult_Diagonal_VectorXd(benchmark::State & state)
{
  VectorXd d = VectorXd::Random(state.range(0));
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  for(auto _ : state) y.noalias() = d.asDiagonal() * x;
}

// y = d*x
static void BM_Mult_scalar_VectorXd(benchmark::State & state)
{
  double d = VectorXd::Random(1)[0];
  VectorXd x = VectorXd::Random(state.range(0));
  VectorXd y(state.range(0));

  for(auto _ : state) y.noalias() = d * x;
}

// MAT_BENCHMARK(BM_Copy_MatrixXd);
// MAT_BENCHMARK(BM_Copy_MatrixXd_Source_Transpose);
// MAT_BENCHMARK(BM_Copy_MatrixXd_Target_Transpose);
// MAT_BENCHMARK(BM_Add_MatrixXd);
// MAT_BENCHMARK(BM_Add_MatrixXd_Source_Transpose);
// MAT_BENCHMARK(BM_Add_MatrixXd_Target_Transpose);
// MAT_BENCHMARK(BM_Mult_VectorXd);
// MAT_BENCHMARK(BM_Mult_VectorXd_Source_Transpose);
// MAT_BENCHMARK(BM_Mult_VectorXd_Target_Transpose);
// MAT_BENCHMARK(BM_Mult_VectorXd_Triangular);
// MAT_BENCHMARK(BM_Mult_VectorXd_TriangularOptim4);
// MAT_BENCHMARK(BM_Mult_VectorXd_TriangularOptim8);
// MAT_BENCHMARK(BM_Mult_VectorXd_TriangularOptim16);
MAT_BENCHMARK(BM_Mult_MatrixXd);
// MAT_BENCHMARK(BM_Mult_MatrixXd_AT);
// MAT_BENCHMARK(BM_Mult_MatrixXd_BT);
// MAT_BENCHMARK(BM_Mult_MatrixXd_AT_BT);
// MAT_BENCHMARK(BM_Mult_MatrixXd_Target_Transpose);
// MAT_BENCHMARK(BM_Mult_MatrixXd_Triangular);
// MAT_BENCHMARK(BM_Mult_MatrixXd_TriangularOptim4);
// MAT_BENCHMARK(BM_Mult_MatrixXd_TriangularOptim8);
// MAT_BENCHMARK(BM_Mult_MatrixXd_TriangularOptim16);
MAT_BENCHMARK(BM_Mult_Diagonal_VectorXd);
MAT_BENCHMARK(BM_Mult_scalar_VectorXd);

BENCHMARK_MAIN();
