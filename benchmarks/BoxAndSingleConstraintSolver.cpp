/* Copyright 2020 CNRS-AIST JRL
 */

#include <iostream>
#include <vector>

#include <benchmark/benchmark.h>

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/BoxAndSingleConstraintSolver.h>

using namespace Eigen;
using namespace jrlqp;
using namespace jrlqp::test;


template<int N, bool act>
class ProblemFixture : public ::benchmark::Fixture
{
public:
  void SetUp(const ::benchmark::State& st)
  {
    i = 0;

    //std::cout << "initialize for nbVar = " << st.range(0) << std::endl;
    for (int k = 0; k < N; ++k)
    {
      pb.emplace_back(generateBoxAndSingleConstraintProblem(st.range(0), act));
      qpp.emplace_back(pb.back());
    }
  }

  void TearDown(const ::benchmark::State&)
  {
    //std::cout << "tear down" << std::endl;
    pb.clear();
    qpp.clear();
  }

  int idx() const
  {
    int ret = i % N;
    ++i;
    return ret;
  }

  const LeastSquareProblem<>& getBSCPb() const
  {
    return pb[idx()];
  }

  QPProblem<>& getGIPb()
  {
    int i = idx();
    qpp[i].G.setIdentity();
    return qpp[i];
  }

private:
  mutable int i;
  std::vector<LeastSquareProblem<>> pb;
  std::vector<QPProblem<>> qpp;
};

using test1 = ProblemFixture<1000, false>;
BENCHMARK_DEFINE_F(test1, BSC_INACTIVE)(benchmark::State& st)
{
  experimental::BoxAndSingleConstraintSolver solver(st.range(0));

  for (auto _ : st)
  {
    const auto& pb = getBSCPb();
    solver.solve(pb.b, pb.C, pb.l[0], pb.xl, pb.xu);
  }
}
BENCHMARK_REGISTER_F(test1, BSC_INACTIVE)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCHMARK_DEFINE_F(test1, GI_INACTIVE)(benchmark::State& st)
{
  GoldfarbIdnaniSolver solver(st.range(0), 1, true);

  for (auto _ : st)
  {
    auto& pb = getGIPb();
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
  }
}
BENCHMARK_REGISTER_F(test1, GI_INACTIVE)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);


using test2 = ProblemFixture<1000, true>;
BENCHMARK_DEFINE_F(test2, BSC_ACTIVE)(benchmark::State& st)
{
  experimental::BoxAndSingleConstraintSolver solver(st.range(0));

  for (auto _ : st)
  {
    const auto& pb = getBSCPb();
    solver.solve(pb.b, pb.C, pb.l[0], pb.xl, pb.xu);
  }
}
BENCHMARK_REGISTER_F(test2, BSC_ACTIVE)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);

BENCHMARK_DEFINE_F(test2, GI_ACTIVE)(benchmark::State& st)
{
  GoldfarbIdnaniSolver solver(st.range(0), 1, true);

  for (auto _ : st)
  {
    auto& pb = getGIPb();
    solver.solve(pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu);
  }
}
BENCHMARK_REGISTER_F(test2, GI_ACTIVE)->Unit(benchmark::kMicrosecond)->DenseRange(10, 100, 10);



BENCHMARK_MAIN();
