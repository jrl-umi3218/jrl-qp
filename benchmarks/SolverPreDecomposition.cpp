/* Copyright 2024 CNRS-AIST JRL, Inria */

#include <array>
#include <iostream>

#include <benchmark/benchmark.h>

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/test/randomProblems.h>

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

template<int N, int EqPercentage>
class ProblemFixture : public ::benchmark::Fixture
{
public:
  void SetUp(const ::benchmark::State & st)
  {
    i = 0;

    int n = static_cast<int>(st.range(0));
    int neq = nEq(n);

    for(int k = 0; k < N; ++k)
    {
      qpp[k] = QPProblem(randomProblem(ProblemCharacteristics(n, n, neq, 0)));
      qpp[k].C.transposeInPlace();
      G[k] = qpp[k].G;
    }
  }

  void TearDown(const ::benchmark::State &) {}

  int idx() const
  {
    int ret = i % N;
    ++i;
    return ret;
  }

  static int nEq(int nVar)
  {
    return static_cast<int>((nVar * EqPercentage) / 100);
  }

  QPProblem<> & getGIPb()
  {
    int i = idx();
    qpp[i].G = G[i];
    return qpp[i];
  }

private:
  mutable int i = 0;
  std::array<MatrixXd, N> G;
  std::array<QPProblem<>, N> qpp;
};

using test0 = ProblemFixture<1000, 0>;
using test50 = ProblemFixture<1000, 50>;

BENCHMARK_DEFINE_F(test0, LLT_INTERNAL)(benchmark::State & st)
{
  for(auto _ : st)
  {
    auto & pb = getGIPb();
    Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb.G);
  }
}
BENCHMARK_REGISTER_F(test0, LLT_INTERNAL)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCHMARK_DEFINE_F(test0, LLT_FUNCTION)(benchmark::State & st)
{
  for(auto _ : st)
  {
    auto & pb = getGIPb();
    auto llt = pb.G.llt();
  }
}
BENCHMARK_REGISTER_F(test0, LLT_FUNCTION)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCHMARK_DEFINE_F(test0, LLT_OBJECT)(benchmark::State & st)
{
  for(auto _ : st)
  {
    auto & pb = getGIPb();
    LLT<MatrixXd> llt(pb.G);
  }
}
BENCHMARK_REGISTER_F(test0, LLT_OBJECT)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCHMARK_DEFINE_F(test0, LLT_PREALLOC)(benchmark::State & st)
{
  int n = static_cast<int>(st.range(0));
  LLT<MatrixXd> llt(n);

  for(auto _ : st)
  {
    auto & pb = getGIPb();
    llt.compute(pb.G);
  }
}
BENCHMARK_REGISTER_F(test0, LLT_PREALLOC)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

#define BENCH_DECOMP(fixture)                                                                           \
  BENCHMARK_DEFINE_F(fixture, NO_PRE_DECOMP)(benchmark::State & st)                                     \
  {                                                                                                     \
    int n = static_cast<int>(st.range(0));                                                              \
    int neq = fixture::nEq(n);                                                                          \
    GoldfarbIdnaniSolver solver(n, neq, false);                                                         \
                                                                                                        \
    for(auto _ : st)                                                                                    \
    {                                                                                                   \
      auto & pb = getGIPb();                                                                            \
      solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);                                         \
    }                                                                                                   \
  }                                                                                                     \
  BENCHMARK_REGISTER_F(fixture, NO_PRE_DECOMP)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10); \
                                                                                                        \
  BENCHMARK_DEFINE_F(fixture, DECOMP_INV)(benchmark::State & st)                                        \
  {                                                                                                     \
    int n = static_cast<int>(st.range(0));                                                              \
    int neq = fixture::nEq(n);                                                                          \
    GoldfarbIdnaniSolver solver(n, neq, false);                                                         \
    MatrixXd J(n, n);                                                                                   \
                                                                                                        \
    for(auto _ : st)                                                                                    \
    {                                                                                                   \
      auto & pb = getGIPb();                                                                            \
      Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb.G);                                \
      auto L = pb.G.template triangularView<Eigen::Lower>();                                            \
      J.setIdentity();                                                                                  \
      L.solveInPlace(J);                                                                                \
    }                                                                                                   \
  }                                                                                                     \
  BENCHMARK_REGISTER_F(fixture, DECOMP_INV)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);    \
                                                                                                        \
  BENCHMARK_DEFINE_F(fixture, DECOMP_INVT)(benchmark::State & st)                                       \
  {                                                                                                     \
    int n = static_cast<int>(st.range(0));                                                              \
    int neq = fixture::nEq(n);                                                                          \
    GoldfarbIdnaniSolver solver(n, neq, false);                                                         \
    MatrixXd J(n, n);                                                                                   \
                                                                                                        \
    for(auto _ : st)                                                                                    \
    {                                                                                                   \
      auto & pb = getGIPb();                                                                            \
      Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb.G);                                \
      auto L = pb.G.template triangularView<Eigen::Lower>();                                            \
      J.setIdentity();                                                                                  \
      L.solveInPlace(J);                                                                                \
      J.transposeInPlace();                                                                             \
    }                                                                                                   \
  }                                                                                                     \
  BENCHMARK_REGISTER_F(fixture, DECOMP_INVT)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);   \
                                                                                                        \
  BENCHMARK_DEFINE_F(fixture, DECOMP_TINV)(benchmark::State & st)                                       \
  {                                                                                                     \
    int n = static_cast<int>(st.range(0));                                                              \
    int neq = fixture::nEq(n);                                                                          \
    GoldfarbIdnaniSolver solver(n, neq, false);                                                         \
    MatrixXd J(n, n);                                                                                   \
                                                                                                        \
    for(auto _ : st)                                                                                    \
    {                                                                                                   \
      auto & pb = getGIPb();                                                                            \
      Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb.G);                                \
      auto L = pb.G.template triangularView<Eigen::Lower>();                                            \
      J.setIdentity();                                                                                  \
      L.transpose().solveInPlace(J);                                                                    \
    }                                                                                                   \
  }                                                                                                     \
  BENCHMARK_REGISTER_F(fixture, DECOMP_TINV)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCH_DECOMP(test0)
BENCH_DECOMP(test50)

BENCHMARK_MAIN();