/* Copyright 2020 CNRS-AIST JRL */

#include <array>
#include <iostream>
#include <map>
#include <set>

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
#ifdef JRLQP_USE_PROXSUITE
#  include <proxsuite/proxqp/dense/dense.hpp>
#endif

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/test/problems.h>
#include <jrl-qp/test/randomProblems.h>

#include "eiquadprog.hpp"
#include "problemAdaptors.h"

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

/** Describe a parameter behavior*/
enum class ParamType
{
  Variable, // Parameter value is variable
  Fixed, // Parameter value is fixed
  FixedFraction, // Parameter value is a fixed fraction of another value
  VariableFraction // Parameter value is a variable fraction of another value
};

/** Helper structure extracting the i-th parameter of a ::benchmark::State range
 *
 * This is part of a collection of helpers to interpret a range.
 */
template<int i>
struct Var
{
  constexpr static int rangeIdx = i;
  constexpr static ParamType type = ParamType::Variable;
  constexpr static int rangeSlot = 1;
  static int value(const ::benchmark::State & st, int)
  {
    return static_cast<int>(st.range(i));
  }
};

/** Helper structure whose value is \p i independently of the ::benchmark::State range
 * is is given.
 *
 * This is part of a collection of helpers to interpret a range.
 */
template<int i>
struct Fixed
{
  constexpr static ParamType type = ParamType::Fixed;
  constexpr static int rangeSlot = 0;
  static int value(const ::benchmark::State &, int)
  {
    return i;
  }
};

/** Helper structure whose value is a fixed fraction n/d of a reference value,
 * independently of the ::benchmark::State range is is given.
 *
 * This is part of a collection of helpers to interpret a range.
 */
template<int n, int d = 100>
struct FFrac
{
  constexpr static double frac = static_cast<double>(n) / d;
  constexpr static ParamType type = ParamType::FixedFraction;
  constexpr static int rangeSlot = 0;
  static int value(const ::benchmark::State &, double ref)
  {
    return static_cast<int>(frac * ref);
  }
};

/** Helper structure whose value is a variable fraction range(i)/d of a reference value,
 * where range is a ::benchmark::State range.
 *
 * This is part of a collection of helpers to interpret a range.
 */
template<int i, int d = 100>
struct VFrac
{
  constexpr static double invd = 1. / d;
  constexpr static ParamType type = ParamType::VariableFraction;
  constexpr static int rangeIdx = i;
  constexpr static int rangeSlot = 1;
  static int value(const ::benchmark::State & st, double ref)
  {
    return static_cast<int>(st.range(i) * ref * invd);
  }
};

/** Compute the decimal representation of the binary number whose digit are \p doubleSided and \p bounds.*/
template<bool bounds, bool doubleSided>
constexpr int packBool()
{
  int r = bounds ? 1 : 0;
  return doubleSided ? r + 2 : r;
}

/** Helper function used for generating the signature of a problem*/
template<typename NVar, typename NEq, typename NIneq, typename NIneqAct, typename NBndAct>
constexpr int rangeSize()
{
  return 1 + NVar::rangeSlot + NEq::rangeSlot + NIneq::rangeSlot + NIneqAct::rangeSlot + NBndAct::rangeSlot;
}

/** An object representing the signature of a problem, used as a key for storing
 * and retrieving problem of a given type and size
 */
template<typename NVar, typename NEq, typename NIneq, typename NIneqAct, typename NBndAct>
using SignatureType = std::array<int, rangeSize<NVar, NEq, NIneq, NIneqAct, NBndAct>()>;

/** Computation of a problem signature for a given ::benchmark::State*/
template<typename NVar,
         typename NEq,
         typename NIneq,
         typename NIneqAct,
         bool Bounds,
         typename NBndAct,
         bool DoubleSided = false>
SignatureType<NVar, NEq, NIneq, NIneqAct, NBndAct> problemSignature(const ::benchmark::State & st)
{
  SignatureType<NVar, NEq, NIneq, NIneqAct, NBndAct> ret;
  ret[0] = packBool<Bounds, DoubleSided>();
  if constexpr(NVar::rangeSlot) ret[NVar::rangeIdx + 1] = static_cast<int>(st.range(NVar::rangeIdx));
  if constexpr(NEq::rangeSlot) ret[NEq::rangeIdx + 1] = static_cast<int>(st.range(NEq::rangeIdx));
  if constexpr(NIneq::rangeSlot) ret[NIneq::rangeIdx + 1] = static_cast<int>(st.range(NIneq::rangeIdx));
  if constexpr(NIneqAct::rangeSlot) ret[NIneqAct::rangeIdx + 1] = static_cast<int>(st.range(NIneqAct::rangeIdx));
  if constexpr(NBndAct::rangeSlot) ret[NBndAct::rangeIdx + 1] = static_cast<int>(st.range(NBndAct::rangeIdx));

  return ret;
}

/** A collection of \p NbPb QP problems with all the variations in representation
 * needed for the different solver tested.
 *
 * Care is taken so that each problems is solvable by all solvers.
 */
template<int NbPb>
struct ProblemCollection
{
  /** Generate all the problems*/
  void generate(int n, int me, int mi, int ma, int na, bool bounds, bool doubleSided)
  {
    nVar = n;
    nEq = me;
    nIneq = mi;
    nIneqAndBnd = mi + (bounds ? nVar : 0);
    nSSIneq = doubleSided ? (2 * mi) : mi;
    nSSIneqAndBnd = (bounds ? 2 * nVar : nVar) + nSSIneq;
    nCstr = me + mi;
    this->bounds = bounds;
    this->doubleSided = doubleSided;

    Eigen::VectorXd x(nVar);
    GoldfarbIdnaniSolver solverGI(nVar, nCstr, bounds);
#ifdef JRLQP_USE_QUADPROG
    Eigen::QuadProgDense solverQP(nVar, nEq, nSSIneqAndBnd);
#endif
#ifdef JRLQP_USE_LSSOL
    Eigen::LSSOL_QP solverLS(nVar, nCstr, Eigen::lssol::QP2);
    solverLS.optimalityMaxIter(500);
    solverLS.feasibilityMaxIter(500);
#endif
#ifdef JRLQP_USE_QLD
    Eigen::QLDDirect solverQLD(nVar, nEq, nSSIneq);
#endif

#ifdef JRLQP_USE_PROXSUITE
    proxsuite::proxqp::dense::QP<double> solverProxSuite(nVar, nEq, nIneqAndBnd);
    solverProxSuite.settings.eps_abs = 1e-8;
#endif

    int failGI = 0;
    int failEigenQuadprog = 0;
    int failEiQuadprog = 0;
    int failLssol = 0;
    int failQLD = 0;
    int failProxSuite = 0;

    for(int k = 0; k < NbPb; ++k)
    {
      generateSingleProblem(k, n, me, mi, ma, na, bounds, doubleSided);

      if(!skipGI)
      {
        auto & qp = giPb[k];
        solverGI.solve(qp.G, qp.a, qp.C, qp.l, qp.u, qp.xl, qp.xu);
        if(!check(solverGI.solution(), k, failGI, skipGI, "GI")) continue;
      }
      if(!skipEiQuadprog)
      {
        auto & qp = eiquadprogPb[k];
        Eigen::solve_quadprog(qp.G, qp.g0, qp.CE, qp.ce0, qp.CI, qp.ci0, x);
        if(!check(x, k, failEiQuadprog, skipEiQuadprog, "eiQuadprog")) continue;
      }
#ifdef JRLQP_USE_QUADPROG
      if(!skipEigenQuadprog)
      {
        auto & qp = quadprogPb[k];
        solverQP.solve(qp.Q, qp.c, qp.Aeq, qp.beq, qp.Aineq, qp.bineq);
        if(!check(solverQP.result(), k, failEigenQuadprog, skipEigenQuadprog, "quadprog")) continue;
      }
#endif
#ifdef JRLQP_USE_LSSOL
      if(!skipLssol)
      {
        auto & qp = lssolPb[k];
        solverLS.solve(qp.Q, qp.p, qp.C, qp.l, qp.u);
        if(!check(solverLS.result(), k, failLssol, skipLssol, "lssol")) continue;
      }
#endif
#ifdef JRLQP_USE_QLD
      if(!skipQLD)
      {
        auto & qp = qldPb[k];
        solverQLD.solve(qp.Q, qp.c, qp.A, qp.b, qp.xl, qp.xu, nEq);
        if(!check(solverQLD.result(), k, failQLD, skipQLD, "qld")) continue;
      }
#endif
#ifdef JRLQP_USE_PROXSUITE
      if(!skipProxSuite)
      {
        auto & qp = proxsuitePb[k];
        solverProxSuite.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
        solverProxSuite.solve();
        if(!check(solverProxSuite.results.x, k, failProxSuite, skipProxSuite, "ProxSuite"))
        {
          //std::cout << (int)solverProxSuite.results.info.status << std::endl;
          continue;
        }
      }
#endif
    }
  }

  bool check(const VectorConstRef & x, int & k, int & failCount, bool & skip, const std::string & name)
  {
    const int maxFail = 20;
    if(!checkSolution(x, k))
    {
      ++failCount;
      if(failCount >= maxFail)
      {
        skip = true;
        std::cout << "Too many errors, skip " << name << "." << std::endl;
        return true;
      }
      else
      {
        --k;
        //std::cout << "Error with " << name << ". Retry" << std::endl;
        return false;
      }
    }
    else
    {
      failCount = 0;
      return true;
    }
  }

  void generateSingleProblem(int k, int n, int me, int mi, int ma, int na, bool bounds, bool doubleSided)
  {
    original[k] = randomProblem(ProblemCharacteristics(n, n, me, mi)
                                    .nStrongActIneq(ma)
                                    .nStrongActBounds(na)
                                    .bounds(bounds)
                                    .doubleSidedIneq(doubleSided));
    QPProblem<true> qp = original[k];
    giPb[k] = qp;
    G[k] = giPb[k].G;
    lssolPb[k] = qp;
    quadprogPb[k] = qp;
    eiquadprogPb[k] = qp;
    qldPb[k] = qp;
    proxsuitePb[k] = qp;
  }

  bool checkSolution(const VectorConstRef & x, int k)
  {
    double err = (x - original[k].x).lpNorm<Eigen::Infinity>();
    if(err <= 5e-6)
    {
      return true;
    }
    else
    {
      //std::cout << err << "  -  ";
      return false;
    }
  }

  int nVar;
  int nEq;
  int nIneq;
  int nIneqAndBnd;
  int nSSIneq;
  int nSSIneqAndBnd;
  int nCstr;
  bool bounds;
  bool doubleSided;
  std::array<RandomLeastSquare, NbPb> original;
  std::array<MatrixXd, NbPb> G;
  std::array<GIPb, NbPb> giPb;
  std::array<LssolPb, NbPb> lssolPb;
  std::array<EigenQuadprogPb, NbPb> quadprogPb;
  std::array<EiQuadprogPb, NbPb> eiquadprogPb;
  std::array<QLDPb, NbPb> qldPb;
  std::array<ProxSuitePb, NbPb> proxsuitePb;
  bool skipGI = false;
  bool skipLssol = false;
  bool skipEigenQuadprog = false;
  bool skipEiQuadprog = false;
  bool skipQLD = false;
  bool skipProxSuite = false;
};

/** A fixture containing and managing collections of problems for all the sizes
 * deduced from the template arguments and the ::benchmark::State
 *
 * The main goal of this class (with all related development) is to ensure that all
 * solvers solve the same set of problems
 *
 * \tparam NbPb Number of different problems for a given set of sizes
 * \tparam NVar Number of variables, specified as one of the helper structures Var,
 * Fixed, VFrac or FFrac, to explain how to get this of variables from a
 * ::benchmark::State range.
 * \tparam NEq Number of equality constraints, specified as one of the helper structures
 * Var, Fixed, VFrac or FFrac, to explain how to get this number from a
 * ::benchmark::State range. If given as a fraction, this is a fraction of NVar.
 * \tparam NIneq Number of inequality constraints, specified as one of the helper
 * structures Var, Fixed, VFrac or FFrac, to explain how to get this number from a
 * ::benchmark::State range. If given as a fraction, this is a fraction of NVar.
 * \tparam NIneqAct Number of active inequality constraints at the solution, specified
 * as one of the helper structures Var, Fixed, VFrac or FFrac, to explain how to get
 * this number from a ::benchmark::State range. If given as a fraction, this is a
 * fraction of min(NVar,NIneq).
 * \tparam Bounds Whether or not the problems have bounds on the variables.
 * \tparam NIneqAct Number of active bounds constraints at the solution, specified as
 * one of the helper structures Var, Fixed, VFrac or FFrac, to explain how to get this
 * number from a ::benchmark::State range. If given as a fraction, this is a fraction of
 * NVar.
 * \tparam DoubleSided Whether or not the constraints are double sided (lower and upper
 * bounds)
 *
 */
template<int NbPb,
         typename NVar,
         typename NEq,
         typename NIneq,
         typename NIneqAct,
         bool Bounds,
         typename NBndAct,
         bool DoubleSided = false>
class ProblemFixture : public ::benchmark::Fixture
{
public:
  using Signature = SignatureType<NVar, NEq, NIneq, NIneqAct, NBndAct>;

  void SetUp(const ::benchmark::State & st)
  {
    auto sig = problemSignature<NVar, NEq, NIneq, NIneqAct, Bounds, NBndAct, DoubleSided>(st);

    i = 0;
    if(problems.find(sig) == problems.end())
    {
      int n = NVar::value(st, 0);
      int me = NEq::value(st, n);
      int mi = NIneq::value(st, n);
      int ma = NIneqAct::value(st, std::min(n, mi));
      int na = NBndAct::value(st, n);

      problems[sig] = {};

      // std::cout << "initialize for (" << n << ", " << me << ", " << mi << ", " << ma << ", " << na << ", " << Bounds
      // << ", " << DoubleSided << ")" << std::endl;
      try
      {
        problems[sig].generate(n, me, mi, ma, na, Bounds, DoubleSided);
      }
      catch(std::exception e)
      {
        std::cout << e.what() << std::endl;
      }
    }
  }

  void TearDown(const ::benchmark::State &) {}

  inline static void clearData()
  {
    problems.clear();
  }

  Signature signature(const ::benchmark::State & st)
  {
    return problemSignature<NVar, NEq, NIneq, NIneqAct, Bounds, NBndAct, DoubleSided>(st);
  }

  int idx()
  {
    int ret = i % NbPb;
    ++i;
    return ret;
  }

  int nVar(const Signature & sig) const
  {
    return problems[sig].nVar;
  }
  int nEq(const Signature & sig) const
  {
    return problems[sig].nEq;
  }
  int nIneq(const Signature & sig) const
  {
    return problems[sig].nIneq;
  }
  int nIneqAndBnd(const Signature & sig) const
  {
    return problems[sig].nIneqAndBnd;
  }
  // Number of single-sided constraints
  int nSSIneq(const Signature & sig) const
  {
    return problems[sig].nSSIneq;
  }
  // Number of single-sided constraints including bounds
  int nSSIneqAndBnd(const Signature & sig) const
  {
    return problems[sig].nSSIneqAndBnd;
  }
  int nCstr(const Signature & sig) const
  {
    return problems[sig].nCstr;
  }
  int bounds(const Signature & sig) const
  {
    return problems[sig].bounds;
  }

  bool skipGI(const Signature & sig) const
  {
    return problems[sig].skipGI;
  }
  bool skipQuadprog(const Signature & sig) const
  {
    return problems[sig].skipEigenQuadprog;
  }
  bool skipEiQuadprog(const Signature & sig) const
  {
    return problems[sig].skipEiQuadprog;
  }
  bool skipLssol(const Signature & sig) const
  {
    return problems[sig].skipLssol;
  }
  bool skipQLD(const Signature & sig) const
  {
    return problems[sig].skipQLD;
  }
  bool skipProxSuite(const Signature & sig) const
  {
    return problems[sig].skipProxSuite;
  }

  RandomLeastSquare & getOriginal()
  {
    return problems[this->sig].original[idx()];
  }

  GIPb & getGIPb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.giPb[i].G = pb.G[i];
    return pb.giPb[i];
  }

  LssolPb & getLssolPb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.lssolPb[i].Q = pb.G[i];
    return pb.lssolPb[i];
  }

  EigenQuadprogPb & getQuadprogPb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.quadprogPb[i].Q = pb.G[i];
    return pb.quadprogPb[i];
  }

  EiQuadprogPb & getEiQuadprogPb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.eiquadprogPb[i].G = pb.G[i];
    return pb.eiquadprogPb[i];
  }

  QLDPb & getQLDPb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.qldPb[i].Q = pb.G[i];
    return pb.qldPb[i];
  }

  ProxSuitePb & getProxsuitePb(const Signature & sig)
  {
    int i = idx();
    auto & pb = problems[sig];
    pb.proxsuitePb[i].H = pb.G[i];
    return pb.proxsuitePb[i];
  }

private:
  int i = 0;

  inline static std::map<Signature, ProblemCollection<NbPb>> problems = {};
};

#include <iostream>

#define NOP

#define BENCH_OVERHEAD(fixture, otherArgs)                     \
  BENCHMARK_DEFINE_F(fixture, Overhead)(benchmark::State & st) \
  {                                                            \
    auto sig = signature(st);                                  \
    for(auto _ : st)                                           \
    {                                                          \
      benchmark::DoNotOptimize(getGIPb(sig));                  \
    }                                                          \
  }                                                            \
  BENCHMARK_REGISTER_F(fixture, Overhead)->Unit(benchmark::kMicrosecond) otherArgs

#define BENCH_GI(fixture, otherArgs)                                 \
  BENCHMARK_DEFINE_F(fixture, GI)(benchmark::State & st)             \
  {                                                                  \
    auto sig = signature(st);                                        \
    if(skipGI(sig)) st.SkipWithError("Skipping GI");                 \
    GoldfarbIdnaniSolver solver(nVar(sig), nCstr(sig), bounds(sig)); \
    for(auto _ : st)                                                 \
    {                                                                \
      auto & qp = getGIPb(sig);                                      \
      solver.solve(qp.G, qp.a, qp.C, qp.l, qp.u, qp.xl, qp.xu);      \
    }                                                                \
  }                                                                  \
  BENCHMARK_REGISTER_F(fixture, GI)->Unit(benchmark::kMicrosecond) otherArgs

#define BENCH_EIQP(fixture, otherArgs)                                     \
  BENCHMARK_DEFINE_F(fixture, EIQP)(benchmark::State & st)                 \
  {                                                                        \
    auto sig = signature(st);                                              \
    if(skipEiQuadprog(sig)) st.SkipWithError("Skipping EiQuadprog");       \
    Eigen::VectorXd x(nVar(sig));                                          \
    for(auto _ : st)                                                       \
    {                                                                      \
      auto & qp = getEiQuadprogPb(sig);                                    \
      Eigen::solve_quadprog(qp.G, qp.g0, qp.CE, qp.ce0, qp.CI, qp.ci0, x); \
    }                                                                      \
  }                                                                        \
  BENCHMARK_REGISTER_F(fixture, EIQP)->Unit(benchmark::kMicrosecond) otherArgs

#ifdef JRLQP_USE_QUADPROG
#  define BENCH_QUADPROG(fixture, otherArgs)                                \
    BENCHMARK_DEFINE_F(fixture, QuadProg)(benchmark::State & st)            \
    {                                                                       \
      auto sig = signature(st);                                             \
      if(skipQuadprog(sig)) st.SkipWithError("Skipping Quadprog");          \
      Eigen::QuadProgDense solver(nVar(sig), nEq(sig), nSSIneqAndBnd(sig)); \
                                                                            \
      for(auto _ : st)                                                      \
      {                                                                     \
        auto & qp = getQuadprogPb(sig);                                     \
        solver.solve(qp.Q, qp.c, qp.Aeq, qp.beq, qp.Aineq, qp.bineq);       \
      }                                                                     \
    }                                                                       \
    BENCHMARK_REGISTER_F(fixture, QuadProg)->Unit(benchmark::kMicrosecond) otherArgs
#else
#  define BENCH_QUADPROG(fixture, otherArgs) NOP
#endif

#ifdef JRLQP_USE_LSSOL
#  define BENCH_LSSOL(fixture, otherArgs)                               \
    BENCHMARK_DEFINE_F(fixture, Lssol)(benchmark::State & st)           \
    {                                                                   \
      auto sig = signature(st);                                         \
      if(skipLssol(sig)) st.SkipWithError("Skipping LSSOL");            \
      Eigen::LSSOL_QP solver(nVar(sig), nCstr(sig), Eigen::lssol::QP2); \
      solver.optimalityMaxIter(500);                                    \
      solver.feasibilityMaxIter(500);                                   \
      for(auto _ : st)                                                  \
      {                                                                 \
        auto & qp = getLssolPb(sig);                                    \
        solver.solve(qp.Q, qp.p, qp.C, qp.l, qp.u);                     \
      }                                                                 \
    }                                                                   \
    BENCHMARK_REGISTER_F(fixture, Lssol)->Unit(benchmark::kMicrosecond) otherArgs
#else
#  define BENCH_LSSOL(fixture, otherArgs) NOP
#endif

#ifdef JRLQP_USE_QLD
#  define BENCH_QLD(fixture, otherArgs)                                  \
    BENCHMARK_DEFINE_F(fixture, QLD)(benchmark::State & st)              \
    {                                                                    \
      auto sig = signature(st);                                          \
      if(skipQLD(sig)) st.SkipWithError("Skipping QLD");                 \
      Eigen::QLDDirect solverQLD(nVar(sig), nEq(sig), nSSIneq(sig));     \
      for(auto _ : st)                                                   \
      {                                                                  \
        auto & qp = getQLDPb(sig);                                       \
        solverQLD.solve(qp.Q, qp.c, qp.A, qp.b, qp.xl, qp.xu, nEq(sig)); \
      }                                                                  \
    }                                                                    \
    BENCHMARK_REGISTER_F(fixture, QLD)->Unit(benchmark::kMicrosecond) otherArgs
#else
#  define BENCH_QLD(fixture, otherArgs) NOP
#endif

#ifdef JRLQP_USE_PROXSUITE
#  define BENCH_PROXSUITE(fixture, otherArgs)                            \
    BENCHMARK_DEFINE_F(fixture, ProxSuite)(benchmark::State & st)        \
    {                                                                    \
      auto sig = signature(st);                                          \
      if(skipProxSuite(sig)) st.SkipWithError("Skipping Proxsuite");     \
      proxsuite::proxqp::dense::QP<double> solverProxSuite(nVar(sig), nEq(sig), nIneqAndBnd(sig)); \
      solverProxSuite.settings.eps_abs = 1e-8;                           \
      for(auto _ : st)                                                   \
      {                                                                  \
        auto & qp = getProxsuitePb(sig);                                 \
        solverProxSuite.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);  \
        solverProxSuite.solve();                                         \
      }                                                                  \
    }                                                                    \
    BENCHMARK_REGISTER_F(fixture, ProxSuite)->Unit(benchmark::kMicrosecond) otherArgs
#else
#  define BENCH_PROXSUITE(fixture, otherArgs) NOP
#endif

#define BENCH_CLEAR(fixture)                                            \
  BENCHMARK_DEFINE_F(fixture, Clear)(benchmark::State & st)             \
  {                                                                     \
    fixture::clearData();                                               \
    st.SkipWithError("Not an error, just a hacky way to clear memory"); \
  }                                                                     \
  BENCHMARK_REGISTER_F(fixture, Clear)

#define BENCH_ALL(fixture, otherArgs) \
  BENCH_OVERHEAD(fixture, otherArgs); \
  BENCH_GI(fixture, otherArgs);       \
  BENCH_EIQP(fixture, otherArgs);     \
  BENCH_QUADPROG(fixture, otherArgs); \
  BENCH_LSSOL(fixture, otherArgs);    \
  BENCH_QLD(fixture, otherArgs);      \
  BENCH_PROXSUITE(fixture, otherArgs);\
  BENCH_CLEAR(fixture)->DenseRange(1, 1, 1);

auto minl = [](const std::vector<double> & v) { return *(std::min_element(std::begin(v), std::end(v))); };
auto maxl = [](const std::vector<double> & v) { return *(std::max_element(std::begin(v), std::end(v))); };

// Varying size, fixed 40% equality
using test1 = ProblemFixture<100, Var<0>, FFrac<40>, Fixed<0>, Fixed<0>, false, Fixed<0>>;
BENCH_ALL(test1, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Fixed nVar = 50 and nIneq=80, varying number of active constraints from 0 to 100%
using test2 = ProblemFixture<100, Fixed<50>, Fixed<0>, Fixed<80>, VFrac<0>, false, Fixed<0>>;
BENCH_ALL(test2, ->DenseRange(0, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Varying size, fixed 20% equality, fixed 100% inequality, with 30% active, bounds
using test3 = ProblemFixture<100, Var<0>, FFrac<20>, FFrac<100>, FFrac<30>, true, FFrac<10>, true>;
BENCH_ALL(test3, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Fixed size, varying equality
using test4 = ProblemFixture<100, Fixed<50>, VFrac<0>, Fixed<0>, Fixed<0>, false, Fixed<0>>;
BENCH_ALL(test4, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Fixed size, as many single-sided inequality, varying active
using test5 = ProblemFixture<100, Fixed<50>, Fixed<0>, Fixed<50>, VFrac<0>, false, Fixed<0>>;
BENCH_ALL(test5, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Fixed size, as many double-sided inequality, varying active
using test6 = ProblemFixture<100, Fixed<50>, Fixed<0>, Fixed<50>, VFrac<0>, false, Fixed<0>, true>;
BENCH_ALL(test6, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

// Fixed size, only bounds, varying active
using test7 = ProblemFixture<100, Fixed<50>, Fixed<0>, Fixed<0>, Fixed<0>, true, VFrac<0>>;
BENCH_ALL(test7, ->DenseRange(10, 100, 10)->ComputeStatistics("min", minl)->ComputeStatistics("max", maxl));

BENCHMARK_MAIN();
