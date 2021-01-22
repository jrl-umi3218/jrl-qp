/* Copyright 2020 CNRS-AIST JRL */

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>

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
MAT_BENCHMARK(BM_Copy_MatrixXd);

// C = A*B
static void BM_Mult_MatrixXd(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd B = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd target(state.range(0), state.range(0));

  for(auto _ : state) target.noalias() = A * B;
}
MAT_BENCHMARK(BM_Mult_MatrixXd);

// LLT of A
static void BM_LLT_Decomposition(benchmark::State & state)
{
  MatrixXd R = MatrixXd::Random(state.range(0), state.range(0));
  MatrixXd A = R.transpose() * R;
  LLT<MatrixXd> llt(state.range(0));
  for(auto _ : state)
  {
    llt.compute(A);
  }
}
MAT_BENCHMARK(BM_LLT_Decomposition);

// QR of A
static void BM_QR_Decomposition(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  HouseholderQR<MatrixXd> qr(state.range(0), state.range(0));
  for(auto _ : state)
  {
    qr.compute(A);
  }
}
MAT_BENCHMARK(BM_QR_Decomposition);

// QR of A in place
static void BM_QR_Decomposition_Inplace(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  HouseholderQR<Ref<MatrixXd>> qr(A);
  for(auto _ : state)
  {
    qr.compute(A);
  }
}
MAT_BENCHMARK(BM_QR_Decomposition_Inplace);

// QR of A in place using internals
static void BM_QR_Decomposition_Inplace_Internal(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  HouseholderQR<MatrixXd> qr(A);
  Index rows = qr.rows();
  Index cols = qr.cols();
  Index size = (std::min)(rows, cols);

  VectorXd m_hCoeffs(size);
  VectorXd m_temp(cols);

  for(auto _ : state)
  {
    internal::householder_qr_inplace_blocked<MatrixXd, VectorXd>::run(A, m_hCoeffs, 48, m_temp.data());
  }
}
MAT_BENCHMARK(BM_QR_Decomposition_Inplace_Internal);

// Col piv QR of A
static void BM_ColPivQR_Decomposition(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  ColPivHouseholderQR<MatrixXd> qr(state.range(0), state.range(0));
  for(auto _ : state)
  {
    qr.compute(A);
  }
}
MAT_BENCHMARK(BM_ColPivQR_Decomposition);

// Col piv QR of A in place
static void BM_ColPivQR_Decomposition_Inplace(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  ColPivHouseholderQR<Ref<MatrixXd>> qr(A);
  for(auto _ : state)
  {
    qr.compute(A);
  }
}
MAT_BENCHMARK(BM_ColPivQR_Decomposition_Inplace);

// Col piv QR of A^T
static void BM_ColPivQR_Decomposition_Transpose(benchmark::State & state)
{
  MatrixXd A = MatrixXd::Random(state.range(0), state.range(0));
  ColPivHouseholderQR<MatrixXd> qr(state.range(0), state.range(0));
  for(auto _ : state)
  {
    qr.compute(A.transpose());
  }
}
MAT_BENCHMARK(BM_ColPivQR_Decomposition_Transpose);

BENCHMARK_MAIN();
