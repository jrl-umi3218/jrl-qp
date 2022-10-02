/* Copyright 2020-2021 CNRS-AIST JRL */

#include <array>
#include <iostream>
#include <map>

#include <benchmark/benchmark.h>

#ifdef JRLQP_USE_LSSOL
#  include <eigen-lssol/LSSOL_QP.h>
#endif
#ifdef JRLQP_USE_QUADPROG
#  include <eigen-quadprog/QuadProg.h>
#endif
#ifdef JRLQP_USE_QLD
#  include <eigen-qld/QLDDirect.h>
#endif

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/experimental/GoldfarbIdnaniSolver.h>
#include <jrl-qp/test/problems.h>
#include <jrl-qp/test/randomMatrices.h>

#include "eiquadprog.hpp"
#include "problemAdaptors.h"

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

/** Generate the problem
 * min 1/2 ||x||^2
 * s.t. -1<=Cx<=1
 *      -1<=x<=1
 * where the inequality constraints are made by planes tangent to the unit sphere.
 */
QPProblem<true> generateWSProblem(int n, int mi, bool doubleSided)
{
  QPProblem<true> pb;
  pb.G = MatrixXd::Identity(n, n);
  pb.a = VectorXd::Zero(n);
  pb.E.resize(0, n);
  pb.f.resize(0);
  pb.C.resize(mi, n);
  if(doubleSided)
    pb.l = -VectorXd::Ones(mi);
  else
    pb.l = VectorXd::Constant(mi, -std::numeric_limits<double>::infinity());
  pb.u = VectorXd::Ones(mi);
  pb.xl = -VectorXd::Ones(n);
  pb.xu = VectorXd::Ones(n);

  for(int i = 0; i < mi; ++i)
  {
    pb.C.row(i) = randUnitVec(n);
  }

  return pb;
}

struct ProblemCollection
{
  void generate(int n, int mi, bool doubleSided)
  {
    const double r = 2 * sqrt(n);
    for(int i = 0; i < NbPb; ++i)
    {
      auto pb = generateWSProblem(n, mi, doubleSided);
      giPb[i] = pb;
      lssolPb[i] = pb;
      quadprogPb[i] = pb;
      eiquadprogPb[i] = pb;
      qldPb[i] = pb;
      // We want p1 and p2 on the sphere of radius r.
      p1[i] = r * randUnitVec(n);
      p2[i] = r * randUnitVec(n);
      std::tie(v[i], a[i]) = decomposedRotation(p1[i], p2[i]);
    }
    nVar = n;
    nIneq = mi;
    nSSIneq = doubleSided ? (2 * mi) : mi;
    this->doubleSided = doubleSided;
  }

  /** Find v and a, such that p2 = cos(a) p1 + sin(a) v for p1 and p2 two vectors of same size and norm.*/
  static std::pair<VectorXd, double> decomposedRotation(const VectorXd & p1, const VectorXd & p2)
  {
    assert(p1.size() == p2.size());
    assert((p1.norm() - p2.norm() < 1e-14));

    double c = p1.dot(p2) / p1.squaredNorm();

    VectorXd v = p2 - c * p1;
    v *= p1.norm() / v.norm();

    return {v, std::acos(c)};
  }

  static const int NbPb = 50;
  std::array<GIPb, NbPb> giPb;
  std::array<LssolPb, NbPb> lssolPb;
  std::array<EigenQuadprogPb, NbPb> quadprogPb;
  std::array<EiQuadprogPb, NbPb> eiquadprogPb;
  std::array<QLDPb, NbPb> qldPb;
  std::array<ProxSuitePb, NbPb> proxsuitePb;
  std::array<VectorXd, NbPb> p1;
  std::array<VectorXd, NbPb> p2;
  std::array<VectorXd, NbPb> v;
  std::array<double, NbPb> a;
  int nVar;
  int nIneq;
  int nSSIneq;
  bool doubleSided;
};

template<int IneqPercentage, bool doubleSided>
class ProblemFixture : public ::benchmark::Fixture
{
public:
  void SetUp(const ::benchmark::State & st)
  {
    int n = st.range(0);
    if(problems.find(n) == problems.end())
    {
      i = -1;
      problems[n] = {};
      problems[n].generate(n, IneqPercentage * n / 100, doubleSided);
    }
  }

  void TearDown(const ::benchmark::State &) {}

  GIPb & getGIPb(int n)
  {
    int k = idx();
    return problems[n].giPb[k];
  }

  LssolPb & getLssolPb(int n)
  {
    int k = idx();
    return problems[n].lssolPb[k];
  }

  EigenQuadprogPb & getQuadprogPb(int n)
  {
    int k = idx();
    return problems[n].quadprogPb[k];
  }

  EiQuadprogPb & getEiQuadprogPb(int n)
  {
    int k = idx();
    return problems[n].eiquadprogPb[k];
  }

  QLDPb & getQLDPb(int n)
  {
    int k = idx();
    return problems[n].qldPb[k];
  }

  ProxSuitePb & getProxsuiteb(int n)
  {
    int k = idx();
    return problems[n].proxsuitePb[k];
  }

  const VectorXd & geta(int n, double t)
  {
    int k = i % ProblemCollection::NbPb;
    const auto & pb = problems[n];
    a = std::cos(t * pb.a[k]) * pb.p1[k] + std::sin(t * pb.a[k]) * pb.v[k];
    // a = problems.p1[k];
    return a;
  }

  int nVar(int n) const
  {
    return problems.at(n).nVar;
  }
  int nEq(int) const
  {
    return 0;
  }
  int nIneq(int n) const
  {
    return problems.at(n).nIneq;
  }
  // Number of single-sided constraints
  int nSSIneq(int n) const
  {
    return problems.at(n).nSSIneq;
  }
  // Number of single-sided constraints including bounds
  int nSSIneqAndBnd(int n) const
  {
    return 2 * nVar(n) + nSSIneq(n);
  }
  int nCstr(int n) const
  {
    return nIneq(n);
  }
  int bounds(int) const
  {
    return true;
  }

private:
  int idx()
  {
    ++i;
    int ret = i % ProblemCollection::NbPb;
    return ret;
  }

  int i = -1;

  VectorXd a;
  std::map<int, ProblemCollection> problems = {};
};

#define NOP

const int STEPS = 100;
const double f = 1. / STEPS;

#define BENCH_OVERHEAD(fixture, otherArgs)                                     \
  BENCHMARK_DEFINE_F(fixture, Overhead)(benchmark::State & st)                 \
  {                                                                            \
    int n = static_cast<int>(st.range(0));                                     \
    for(auto _ : st)                                                           \
    {                                                                          \
      benchmark::DoNotOptimize(getGIPb(n));                                    \
      for(int i = 0; i < STEPS; ++i) benchmark::DoNotOptimize(geta(n, i * f)); \
    }                                                                          \
    st.counters["it"] = 0;                                                     \
  }                                                                            \
  BENCHMARK_REGISTER_F(fixture, Overhead)->Unit(benchmark::kMicrosecond) otherArgs

#define BENCH_GI(fixture, otherArgs)                               \
  BENCHMARK_DEFINE_F(fixture, GI)(benchmark::State & st)           \
  {                                                                \
    int it = 0;                                                    \
    int n = static_cast<int>(st.range(0));                         \
    GoldfarbIdnaniSolver solver(nVar(n), nCstr(n), true);          \
    for(auto _ : st)                                               \
    {                                                              \
      auto & qp = getGIPb(n);                                      \
      for(int i = 0; i < STEPS; ++i)                               \
      {                                                            \
        const auto & a = geta(n, i * f);                           \
        solver.solve(qp.G, a, qp.C, qp.l, qp.u, qp.xl, qp.xu);     \
        it += solver.iterations();                                 \
      }                                                            \
    }                                                              \
    st.counters["it"] = static_cast<double>(it) / st.iterations(); \
  }                                                                \
  BENCHMARK_REGISTER_F(fixture, GI)->Unit(benchmark::kMicrosecond) otherArgs

#define BENCH_GI_EX(fixture, otherArgs)                                 \
  BENCHMARK_DEFINE_F(fixture, GI_EX)(benchmark::State & st)             \
  {                                                                     \
    int n = static_cast<int>(st.range(0));                              \
    experimental::GoldfarbIdnaniSolver solver(nVar(n), nCstr(n), true); \
    SolverOptions opt;                                                  \
    opt.warmStart_ = true;                                              \
    solver.options(opt);                                                \
    int it = 0;                                                         \
    for(auto _ : st)                                                    \
    {                                                                   \
      solver.resetActiveSet();                                          \
      auto & qp = getGIPb(n);                                           \
      for(int i = 0; i < STEPS; ++i)                                    \
      {                                                                 \
        const auto & a = geta(n, i * f);                                \
        solver.solve(qp.G, a, qp.C, qp.l, qp.u, qp.xl, qp.xu);          \
        it += solver.iterations();                                      \
      }                                                                 \
    }                                                                   \
    st.counters["it"] = static_cast<double>(it) / st.iterations();      \
  }                                                                     \
  BENCHMARK_REGISTER_F(fixture, GI_EX)->Unit(benchmark::kMicrosecond) otherArgs

//#define BENCH_EIQP(fixture, otherArgs)                                     \
//  BENCHMARK_DEFINE_F(fixture, EIQP)(benchmark::State & st)                 \
//  {                                                                        \
//    auto sig = signature(st);                                              \
//    if(skipEiQuadprog(sig)) st.SkipWithError("Skipping EiQuadprog");       \
//    Eigen::VectorXd x(nVar(sig));                                          \
//    for(auto _ : st)                                                       \
//    {                                                                      \
//      auto & qp = getEiQuadprogPb(sig);                                    \
//      Eigen::solve_quadprog(qp.G, qp.g0, qp.CE, qp.ce0, qp.CI, qp.ci0, x); \
//    }                                                                      \
//  }                                                                        \
//  BENCHMARK_REGISTER_F(fixture, EIQP)->Unit(benchmark::kMicrosecond) otherArgs
#define BENCH_EIQP(fixture, otherArgs) NOP
//
//#ifdef JRLQP_USE_QUADPROG
//#  define BENCH_QUADPROG(fixture, otherArgs)                                \
//    BENCHMARK_DEFINE_F(fixture, QuadProg)(benchmark::State & st)            \
//    {                                                                       \
//      auto sig = signature(st);                                             \
//      if(skipQuadprog(sig)) st.SkipWithError("Skipping Quadprog");          \
//      Eigen::QuadProgDense solver(nVar(sig), nEq(sig), nSSIneqAndBnd(sig)); \
//                                                                            \
//      for(auto _ : st)                                                      \
//      {                                                                     \
//        auto & qp = getQuadprogPb(sig);                                     \
//        solver.solve(qp.Q, qp.c, qp.Aeq, qp.beq, qp.Aineq, qp.bineq);       \
//      }                                                                     \
//    }                                                                       \
//    BENCHMARK_REGISTER_F(fixture, QuadProg)->Unit(benchmark::kMicrosecond) otherArgs
//#else
#define BENCH_QUADPROG(fixture, otherArgs) NOP
//#endif

#ifdef JRLQP_USE_LSSOL
#  define BENCH_LSSOL(fixture, otherArgs)                            \
    BENCHMARK_DEFINE_F(fixture, Lssol)(benchmark::State & st)        \
    {                                                                \
      int n = static_cast<int>(st.range(0));                         \
      Eigen::LSSOL_QP solver(nVar(n), nCstr(n), Eigen::lssol::QP2);  \
      solver.optimalityMaxIter(5000);                                \
      solver.feasibilityMaxIter(5000);                               \
      solver.warm(true);                                             \
      solver.persistence(true);                                      \
      int it = 0;                                                    \
      for(auto _ : st)                                               \
      {                                                              \
        auto & qp = getLssolPb(n);                                   \
        solver.reset();                                              \
        for(int i = 0; i < STEPS; ++i)                               \
        {                                                            \
          const auto & a = geta(n, i * f);                           \
          qp.Q.setIdentity();                                        \
          solver.solve(qp.Q, a, qp.C, qp.l, qp.u);                   \
          it += solver.iter();                                       \
        }                                                            \
      }                                                              \
      st.counters["it"] = static_cast<double>(it) / st.iterations(); \
    }                                                                \
    BENCHMARK_REGISTER_F(fixture, Lssol)->Unit(benchmark::kMicrosecond) otherArgs
#else
#  define BENCH_LSSOL(fixture, otherArgs) NOP
#endif
//
//#ifdef JRLQP_USE_QLD
//#  define BENCH_QLD(fixture, otherArgs)                                  \
//    BENCHMARK_DEFINE_F(fixture, QLD)(benchmark::State & st)              \
//    {                                                                    \
//      auto sig = signature(st);                                          \
//      if(skipQLD(sig)) st.SkipWithError("Skipping QLD");                 \
//      Eigen::QLDDirect solverQLD(nVar(sig), nEq(sig), nSSIneq(sig));     \
//      for(auto _ : st)                                                   \
//      {                                                                  \
//        auto & qp = getQLDPb(sig);                                       \
//        solverQLD.solve(qp.Q, qp.c, qp.A, qp.b, qp.xl, qp.xu, nEq(sig)); \
//      }                                                                  \
//    }                                                                    \
//    BENCHMARK_REGISTER_F(fixture, QLD)->Unit(benchmark::kMicrosecond) otherArgs
//#else
#define BENCH_QLD(fixture, otherArgs) NOP
//#endif

#define BENCH_ALL(fixture, otherArgs) \
  BENCH_OVERHEAD(fixture, otherArgs); \
  BENCH_GI(fixture, otherArgs);       \
  BENCH_GI_EX(fixture, otherArgs);    \
  BENCH_EIQP(fixture, otherArgs);     \
  BENCH_QUADPROG(fixture, otherArgs); \
  BENCH_LSSOL(fixture, otherArgs);    \
  BENCH_QLD(fixture, otherArgs);

auto minl = [](const std::vector<double> & v) { return *(std::min_element(std::begin(v), std::end(v))); };
auto maxl = [](const std::vector<double> & v) { return *(std::max_element(std::begin(v), std::end(v))); };

// Varying size, fixed 40% equality
using test1 = ProblemFixture<1000, true>;
BENCH_ALL(test1, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl))

BENCHMARK_MAIN();
