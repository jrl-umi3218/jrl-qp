/* Copyright 2020 CNRS-AIST JRL
 */

#include <array>

#include <benchmark/benchmark.h>

#include <eigen-lssol/LSSOL_QP.h>
#include <eigen-quadprog/QuadProg.h>
#include <eigen-qld/QLDDirect.h>

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/test/problems.h>
#include <jrl-qp/test/randomProblems.h>

#include "eiquadprog.hpp"
#include "problemAdaptors.h"

using namespace Eigen;
using namespace jrlqp;
using namespace jrlqp::test;

enum class ParamType
{
  Variable,
  Fixed,
  FixedFraction,
  VariableFraction
};

template<int i> 
struct Var
{
  constexpr static int rangeIdx = i;
  constexpr static ParamType type = ParamType::Variable;
  constexpr static int rangeSlot = 1;
  static int value(const ::benchmark::State& st, int) { return st.range(i); }
};

template<int i>
struct Fixed
{
  constexpr static ParamType type = ParamType::Fixed;
  constexpr static int rangeSlot = 0;
  static int value(const ::benchmark::State&, int) { return i; }
};

template<int n, int d=10>
struct FFrac
{
  constexpr static double frac = static_cast<double>(n)/d;
  constexpr static ParamType type = ParamType::FixedFraction;
  constexpr static int rangeSlot = 0;
  static int value(const ::benchmark::State&, double ref) { return static_cast<int>(frac * ref); }
};

template<int i, int d=10>
struct VFrac
{
  constexpr static double invd = 1. / d;
  constexpr static ParamType type = ParamType::VariableFraction;
  constexpr static int rangeIdx = i;
  constexpr static int rangeSlot = 1;
  static int value(const ::benchmark::State& st, double ref) { return static_cast<int>(st.range(i) * ref * invd); }
};

template<bool bounds, bool doubleSided>
constexpr int packBool()
{
  int r;
  if (bounds) r = 1; else r = 0;
  if (doubleSided) r += 2;
  return r;
}

template<typename NVar, typename NEq, typename NIneq, typename NIneqAct, typename NBndAct>
constexpr int rangeSize()
{
  return 1 + NVar::rangeSlot + NEq::rangeSlot + NIneq::rangeSlot + NIneqAct::rangeSlot + NBndAct::rangeSlot;
}

template<typename NVar, typename NEq, typename NIneq, typename NIneqAct, typename NBndAct>
using SignatureType = std::array<int, rangeSize<NVar, NEq, NIneq, NIneqAct, NBndAct>()>;

template<typename NVar, typename NEq, typename NIneq, typename NIneqAct, bool Bounds, typename NBndAct, bool DoubleSided = false>
SignatureType<NVar, NEq, NIneq, NIneqAct, NBndAct> problemSignature(const ::benchmark::State& st)
{
  SignatureType<NVar, NEq, NIneq, NIneqAct, NBndAct> ret;
  ret[0] = packBool<Bounds, DoubleSided>();
  if constexpr (NVar::rangeSlot) ret[nVar::rangeIdx + 1] = st.range(NVar::rangeIdx);
  if constexpr (NEq::rangeSlot) ret[nVar::rangeIdx + 1] = st.range(NEq::rangeIdx);
  if constexpr (NIneq::rangeSlot) ret[nVar::rangeIdx + 1] = st.range(NIneq::rangeIdx);
  if constexpr (NIneqAct::rangeSlot) ret[nVar::rangeIdx + 1] = st.range(NIneqAct::rangeIdx);
  if constexpr (NBndAct::rangeSlot) ret[nVar::rangeIdx + 1] = st.range(NBndAct::rangeIdx);
}

template<int NbPb>
struct ProblemCollection
{
  void generate(int n, int me, int mi, int ma, int na, bool bounds, bool doubleSided)
  {
    for (int k = 0; k < NbPb; ++k)
    {
      original[k] = randomProblem(ProblemCharacteristics(n, n, me, mi)
                                  .nStrongActIneq(ma)
                                  .nStrongActBounds(na)
                                  .bounds(bounds)
                                  .doubleSidedIneq(doubleSided));
      QPProblem<true> qp = original[k];
      GIPb[k] = qp;
      G[k] = GIPb[k].G;
      lssolPb[k] = qp;
      quadprogPb[k] = qp;
      eiquadprogPb[k] = qp;
      qldPb[k] = qp;
    }
    nVar = n;
    nEq = me;
    nIneq = mi;
    nCstr = me + mi;
    this->bounds = bounds;
    this->doubleSided = doubleSided;
  }

  void check()
  {
    Eigen::VectorXd x(nVar);
    GoldfarbIdnaniSolver solverGI(nVar, nCstr, bounds);
    Eigen::QuadProgDense solverQP(nVar, nEq, nIneq);
    Eigen::LSSOL_QP solverLS(nVar, nCstr, Eigen::lssol::QP2);
    solverLS.optimalityMaxIter(500);
    solverLS.feasibilityMaxIter(500);
    //Eigen::QLDDirect solverQLD(nVar, nEq, nIneq);
    for (int k = 0; k < NbPb; ++k)
    {
      {
        auto& qp = GIPb[k];
        solverGI.solve(qp.G, qp.a, qp.C, qp.l, qp.u, qp.xl, qp.xu);
        checkSolution(solverGI.solution(), k, "GI");
      }
      {
        auto& qp = eiquadprogPb[k];
        Eigen::solve_quadprog(qp.G, qp.g0, qp.CE, qp.ce0, qp.CI, qp.ci0, x);
        checkSolution(x, k, "eiQuadprog");
      }
      {
        auto& qp = quadprogPb[k];
        solverQP.solve(qp.Q, qp.c, qp.Aeq, qp.beq, qp.Aineq, qp.bineq);
        checkSolution(solverQP.result(), k, "quadprog");
      }
      {
        auto& qp = lssolPb[k];
        solverLS.solve(qp.Q, qp.p, qp.C, qp.l, qp.u);
        checkSolution(solverLS.result(), k, "lssol");
      }
      //{
      //  auto& qp = qldPb[k];
      //  solverQLD.solve(qp.Q, qp.c, qp.A, qp.b, qp.xl, qp.xu, nEq);
      //  checkSolution(solverQLD.result(), k, "qld");
      //}
    }
  }

  void checkSolution(const VectorConstRef& x, int k, const std::string& name)
  {
    double err = (x - original[k].x).norm();
    if (err > 1e-6)
      throw std::runtime_error("Unexpected solution for " + name);
  }

  int nVar;
  int nEq;
  int nIneq;
  int nCstr;
  bool bounds;
  bool doubleSided;
  std::array<RandomLeastSquare, NbPb> original;
  std::array<MatrixXd, NbPb> G;
  std::array<GIPb, NbPb> GIPb;
  std::array<LssolPb, NbPb> lssolPb;
  std::array<EigenQuadprogPb, NbPb> quadprogPb;
  std::array<EiQuadprogPb, NbPb> eiquadprogPb;
  std::array<QLDPb, NbPb> qldPb;
};


template<int NbPb, typename NVar, typename NEq, typename NIneq, typename NIneqAct, bool Bounds, typename NBndAct, bool DoubleSided = false>
class ProblemFixture : public ::benchmark::Fixture
{
public:
  void SetUp(const ::benchmark::State& st)
  {
    static bool initialized = false;

    i = 0;
    if (!initialized)
    {
      int n = NVar::value(st, 0);
      int me = NEq::value(st, n);
      int mi = NIneq::value(st, n);
      int ma = NIneqAct::value(st, std::min(n, mi));
      int na = NBndAct::value(st, n);

      int nTry;
      for (nTry = 0; nTry < 5; ++nTry)
      {
        std::cout << "initialize for (" << n << ", " << me << ", " << mi << ", " << ma << ", " << na << ", " << Bounds << ", " << DoubleSided << ")" << std::endl;
        problems.generate(n, me, mi, ma, na, Bounds, DoubleSided);
        try
        {
          problems.check();
          break;
        }
        catch (std::runtime_error e)
        {
          std::cout << e.what() << std::endl;
          std::cout << "retry" << std::endl;
        }
      }
      if (nTry >= 5)
        throw std::runtime_error("unable to generate problems");
      initialized = true;
    }
    
  }

  void TearDown(const ::benchmark::State&)
  {
  }

  int idx()
  {
    int ret = i % NbPb;
    ++i;
    return ret;
  }

  int nVar() const { return problems.nVar; }
  int nEq() const { return problems.nEq; }
  int nIneq() const { return problems.nIneq; }
  int nCstr() const { return problems.nCstr; }
  int bounds() const { return problems.bounds; }

  RandomLeastSquare& getOriginal() { return problems.original[idx()]; }

  GIPb& getGIPb() 
  { 
    int i = idx(); 
    problems.GIPb[i].G = problems.G[i];  
    return problems.GIPb[i];
  }

  LssolPb& getLssolPb()
  {
    int i = idx();
    problems.lssolPb[i].Q = problems.G[i];
    return problems.lssolPb[i];
  }

  EigenQuadprogPb& getQuadprogPb()
  {
    int i = idx();
    problems.quadprogPb[i].Q = problems.G[i];
    return problems.quadprogPb[i];
  }

  EiQuadprogPb& getEiQuadprogPb()
  {
    int i = idx();
    problems.eiquadprogPb[i].G = problems.G[i];
    return problems.eiquadprogPb[i];
  }

  QLDPb& getQLDPb()
  {
    int i = idx();
    problems.qldPb[i].Q = problems.G[i];
    return problems.qldPb[i];
  }

private:
  int i;
  
  inline static ProblemCollection<NbPb> problems;
};

#include<iostream>

#define BENCH_OVERHEAD(fixture)                                       \
BENCHMARK_DEFINE_F(fixture, Overhead)(benchmark::State& st)           \
{                                                                     \
  GoldfarbIdnaniSolver solver(nVar(), nCstr(), bounds());             \
  for (auto _ : st)                                                   \
  {                                                                   \
    benchmark::DoNotOptimize(getGIPb());                              \
  }                                                                   \
}                                                                     \
BENCHMARK_REGISTER_F(fixture, Overhead)->Unit(benchmark::kMicrosecond)

#define BENCH_GI(fixture)                                             \
BENCHMARK_DEFINE_F(fixture, GI)(benchmark::State& st)                 \
{                                                                     \
  GoldfarbIdnaniSolver solver(nVar(), nCstr(), bounds());             \
  for (auto _ : st)                                                   \
  {                                                                   \
    auto& qp = getGIPb();                                             \
    solver.solve(qp.G, qp.a, qp.C, qp.l, qp.u, qp.xl, qp.xu);         \
  }                                                                   \
}                                                                     \
BENCHMARK_REGISTER_F(fixture, GI)->Unit(benchmark::kMicrosecond)

#define BENCH_EIQP(fixture)                                               \
BENCHMARK_DEFINE_F(fixture, EIQP)(benchmark::State& st)                   \
{                                                                         \
  Eigen::VectorXd x(nVar());                                              \
  for (auto _ : st)                                                       \
  {                                                                       \
    auto& qp = getEiQuadprogPb();                                         \
    Eigen::solve_quadprog(qp.G, qp.g0, qp.CE, qp.ce0, qp.CI, qp.ci0, x);  \
  }                                                                       \
}                                                                         \
BENCHMARK_REGISTER_F(fixture, EIQP)->Unit(benchmark::kMicrosecond)

#define BENCH_QUADPROG(fixture)                                       \
BENCHMARK_DEFINE_F(fixture, QuadProg)(benchmark::State& st)           \
{                                                                     \
  Eigen::QuadProgDense solver(nVar(),nEq(),nIneq());                  \
                                                                      \
  for (auto _ : st)                                                   \
  {                                                                   \
    auto& qp = getQuadprogPb();                                       \
    solver.solve(qp.Q, qp.c, qp.Aeq, qp.beq, qp.Aineq, qp.bineq);     \
  }                                                                   \
}                                                                     \
BENCHMARK_REGISTER_F(fixture, QuadProg)->Unit(benchmark::kMicrosecond)

#define BENCH_LSSOL(fixture)                                      \
BENCHMARK_DEFINE_F(fixture, Lssol)(benchmark::State& st)          \
{                                                                 \
  Eigen::LSSOL_QP solver(nVar(), nCstr(), Eigen::lssol::QP2);     \
  solver.optimalityMaxIter(500);                                  \
  solver.feasibilityMaxIter(500);                                 \
  for (auto _ : st)                                               \
  {                                                               \
    auto& qp = getLssolPb();                                      \
    solver.solve(qp.Q, qp.p, qp.C, qp.l, qp.u);                   \
  }                                                               \
}                                                                 \
BENCHMARK_REGISTER_F(fixture, Lssol)->Unit(benchmark::kMicrosecond)

//BENCHMARK_DEFINE_F(test1, LssolHackyWarmstart)(benchmark::State& st)
//{
//  Eigen::LSSOL_QP solver(nVar(), nCstr(), Eigen::lssol::QP2);
//  solver.optimalityMaxIter(500);
//  solver.feasibilityMaxIter(500);
//  solver.warm(true);
//  solver.persistence(true);
//  auto& qp = getLssolPb();
//  MatrixXd Q = qp.Q;
//  for (auto _ : st)
//  {
//    qp.Q = Q;
//    solver.solve(qp.Q, qp.p, qp.C, qp.l, qp.u);
//  }
//}
//BENCHMARK_REGISTER_F(test1, LssolHackyWarmstart);

//BENCHMARK_DEFINE_F(test1, QLD)(benchmark::State& st)
//{
//  Eigen::QLDDirect solver(100, 40, 100);
//  for (auto _ : st)
//  {
//    auto& qp = getQLDPb();
//    solver.solve(qp.Q, qp.c, qp.A, qp.b, qp.xl, qp.xu, 40);
//  }
//}
//BENCHMARK_REGISTER_F(test1, QLD);


#define BENCH_ALL(fixture, otherArgs) \
BENCH_OVERHEAD(test1)otherArgs;       \
BENCH_GI(test1)otherArgs;             \
BENCH_EIQP(test1)otherArgs;           \
BENCH_QUADPROG(test1)otherArgs;       \
BENCH_LSSOL(test1)otherArgs;          \

//using test1 = ProblemFixture<100, Fixed<100>, Fixed<40>, Fixed<100>, Fixed<40>, false, Fixed<0>>;
//using test1 = ProblemFixture<100, Fixed<100>, Fixed<0>, Fixed<50>, Fixed<0>, false, Fixed<0>>;
using test1 = ProblemFixture<100, Var<0>, FFrac<4>, Fixed<0>, Fixed<0>, false, Fixed<0>>;
BENCH_ALL(fixture, ->DenseRange(10,100,10));

BENCHMARK_MAIN();
