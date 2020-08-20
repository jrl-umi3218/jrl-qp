/* Copyright 2020 CNRS-AIST JRL */

#include <fstream>
#include <iostream>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include "QPSReader.h"
#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <jrl-qp/internal/memoryChecks.h>
#include <jrl-qp/test/kkt.h>
#include <jrl-qp/test/randomProblems.h>

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::test;

TEST_CASE("Simple problem")
{
  MatrixXd G = MatrixXd::Identity(3, 3);
  VectorXd a = VectorXd::Zero(3);
  MatrixXd C = MatrixXd::Random(3, 5);
  VectorXd bl = -VectorXd::Ones(5);
  VectorXd bu = VectorXd::Ones(5);
  VectorXd xl(0);
  VectorXd xu(0);

  GoldfarbIdnaniSolver qp(3, 5, false);

  // unconstraint solution
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));

  // At least one constraint activated
  bl[1] = -2;
  bu[1] = -1;
  G.setIdentity();
  qp.solve(G, a, C, bl, bu, xl, xu);

  G.setIdentity();
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}

TEST_CASE("Simple problem paper")
{
  // example from the Goldfarb-Idnani paper
  MatrixXd G(2, 2);
  G << 4, -2, -2, 4;
  VectorXd a(2);
  a << 6, 0;
  MatrixXd C(2, 1);
  C << 1, 1;
  VectorXd bl(1);
  bl << 2;
  VectorXd bu(1);
  bu << 10;
  VectorXd xl = VectorXd::Zero(2);
  VectorXd xu = VectorXd::Constant(2, 10);

  GoldfarbIdnaniSolver qp(2, 1, true);

  qp.solve(G, a, C, bl, bu, xl, xu);

  G << 4, -2, -2, 4;
  FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, a, C, bl, bu, xl, xu, true));
}

TEST_CASE("Random problems")
{
  std::vector problems = {
      randomProblem(ProblemCharacteristics(5, 5)), randomProblem(ProblemCharacteristics(5, 5).nEq(2)),
      randomProblem(ProblemCharacteristics(5, 5).nIneq(8).nStrongActIneq(4)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(3)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(1).bounds(true).nStrongActBounds(2))};

  for(const auto & pb : problems)
  {
    QPProblem qpp(pb);
    MatrixXd G = qpp.G; // copy for later check
    GoldfarbIdnaniSolver solver(qpp.G.rows(), qpp.C.rows(), pb.bounds);
    jrl::qp::internal::set_is_malloc_allowed(false);
    auto ret = solver.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
    jrl::qp::internal::set_is_malloc_allowed(true);
    FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
    FAST_CHECK_UNARY(
        test::testKKT(solver.solution(), solver.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu, false));
    FAST_CHECK_UNARY(solver.solution().isApprox(pb.x, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().head(pb.E.rows()).isApprox(pb.lambdaEq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().segment(pb.E.rows(), pb.C.rows()).isApprox(pb.lambdaIneq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().tail(pb.xl.size()).isApprox(pb.lambdaBnd, 1e-6));
  }
}

TEST_CASE("Multiple uses")
{
  std::vector problems = {
      randomProblem(ProblemCharacteristics(5, 5)), randomProblem(ProblemCharacteristics(5, 5).nEq(2)),
      randomProblem(ProblemCharacteristics(5, 5).nIneq(8).nStrongActIneq(4)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(3)),
      randomProblem(ProblemCharacteristics(5, 5, 2, 6).nStrongActIneq(1).bounds(true).nStrongActBounds(2))};

  GoldfarbIdnaniSolver solver(5, 8, true);
  for(const auto & pb : problems)
  {
    QPProblem qpp(pb);
    MatrixXd G = qpp.G; // copy for later check
    jrl::qp::internal::set_is_malloc_allowed(false);
    auto ret = solver.solve(qpp.G, qpp.a, qpp.C.transpose(), qpp.l, qpp.u, qpp.xl, qpp.xu);
    jrl::qp::internal::set_is_malloc_allowed(true);
    FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
    FAST_CHECK_UNARY(
        test::testKKT(solver.solution(), solver.multipliers(), G, qpp.a, qpp.C, qpp.l, qpp.u, qpp.xl, qpp.xu, false));
    FAST_CHECK_UNARY(solver.solution().isApprox(pb.x, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().head(pb.E.rows()).isApprox(pb.lambdaEq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().segment(pb.E.rows(), pb.C.rows()).isApprox(pb.lambdaIneq, 1e-6));
    FAST_CHECK_UNARY(solver.multipliers().tail(pb.xl.size()).isApprox(pb.lambdaBnd, 1e-6));
  }
}

#ifdef QPS_TESTS_DIR
struct QPSPbData
{
  std::string name; // name of the problem
  double fstar; // value of the objective at the optimum
  double cond; //(estimated) condition number of G
  int nbCstr; // number of constraints
  int nbVar; // number of variables
  int nz; // number of nonzeros in C
  int qn; // number of quadratic variables
  int qnz; // number of off-diagonal entries in the lower triangular part of G
};
TEST_CASE("Test Suite")
{
  constexpr double Inf = std::numeric_limits<double>::infinity();
  const std::vector<QPSPbData> pbList = {
      //  name      opt. obj value  cond(G) (est.) nbCstr   nbVar      NZ      QN      QNZ
      {"aug2d", 1.6874118e+06, Inf, 10000, 20200, 40000, 19800, 0},
      {"aug2dc", 1.8183681e+06, 1, 10000, 20200, 40000, 20200, 0},
      {"aug2dcqp", 6.4981348e+06, 1, 10000, 20200, 40000, 20200, 0},
      {"aug2dqp", 6.2370121e+06, Inf, 10000, 20200, 40000, 19800, 0},
      {"aug3d", 5.5406773e+02, Inf, 1000, 3873, 6546, 2673, 0},
      {"aug3dc", 7.7126244e+02, 1, 1000, 3873, 6546, 3873, 0},
      {"aug3dcqp", 9.9336215e+02, 1, 1000, 3873, 6546, 3873, 0},
      {"aug3dqp", 6.7523767e+02, Inf, 1000, 3873, 6546, 2673, 0},
      {"boyd1", -6.1735220e+07, 1782, 18, 93261, 558985, 93261, 0},
      {"boyd2", 2.1256767e+01, Inf, 186531, 93263, 423784, 2, 0},
      {"cont-050", -4.5638509e+00, 2, 2401, 2597, 12005, 2597, 0},
      {"cont-100", -4.6443979e+00, 1, 9801, 10197, 49005, 10197, 0},
      {"cont-101", 1.9552733e-01, Inf, 10098, 10197, 49599, 2700, 0},
      {"cont-200", -4.6848759e+00, 2, 39601, 40397, 198005, 40397, 0},
      {"cont-201", 1.9248337e-01, Inf, 40198, 40397, 199199, 10400, 0},
      {"cont-300", 1.9151232e-01, Inf, 90298, 90597, 448799, 23100, 0},
      {"cvxqp1_l", 1.0870480e+08, Inf, 5000, 10000, 14998, 10000, 29984},
      {"cvxqp1_m", 1.0875116e+06, 7.9548418e+17, 500, 1000, 1498, 1000, 2984},
      {"cvxqp1_s", 1.1590718e+04, 1.3398455e+17, 50, 100, 148, 100, 286},
      {"cvxqp2_l", 8.1842458e+07, Inf, 2500, 10000, 7499, 10000, 29984},
      {"cvxqp2_m", 8.2015543e+05, 7.9548418e+17, 250, 1000, 749, 1000, 2984},
      {"cvxqp2_s", 8.1209405e+03, 1.3398455e+17, 25, 100, 74, 100, 286},
      {"cvxqp3_l", 1.1571110e+08, Inf, 7500, 10000, 22497, 10000, 29984},
      {"cvxqp3_m", 1.3628287e+06, 7.9548418e+17, 750, 1000, 2247, 1000, 2984},
      {"cvxqp3_s", 1.1943432e+04, 1.3398455e+17, 75, 100, 222, 100, 286},
      {"dpklo1", 3.7009622e-01, Inf, 77, 133, 1575, 77, 0},
      {"dtoc3", 2.3526248e+02, Inf, 9998, 14999, 34993, 14997, 0},
      {"dual1", 3.5012966e-02, 8604.2029, 1, 85, 85, 85, 3473},
      {"dual2", 3.3733676e-02, 2865.7763, 1, 96, 96, 96, 4412},
      {"dual3", 1.3575584e-01, 987.4926, 1, 111, 111, 111, 5997},
      {"dual4", 7.4609084e-01, 103.0244, 1, 75, 75, 75, 2724},
      {"dualc1", 6.1552508e+03, 1107045.8821, 215, 9, 1935, 9, 36},
      {"dualc2", 3.5513077e+03, 5.0415126e+17, 229, 7, 1603, 7, 21},
      {"dualc5", 4.2723233e+02, 1744.856, 278, 8, 2224, 8, 28},
      {"dualc8", 1.8309359e+04, 1.0107421e+17, 503, 8, 4024, 8, 28},
      {"exdata", -1.4184343e+02, Inf, 3001, 3000, 7500, 1500, 1124250},
      {"genhs28", 9.2717369e-01, 3.0394937e+16, 8, 10, 24, 10, 9},
      {"gouldqp2", 1.8427534e-04, Inf, 349, 699, 1047, 349, 348},
      {"gouldqp3", 2.0627840e+00, 2.9462113e+16, 349, 699, 1047, 698, 697},
      {"hs118", 6.6482045e+02, 1.5, 17, 15, 39, 15, 0},
      {"hs21", -9.9960000e+01, 100, 1, 2, 2, 2, 0},
      {"hs268", 5.7310705e-07, 1176920.3779, 5, 5, 25, 5, 10},
      {"hs35", 1.1111111e-01, 16.3937, 1, 3, 3, 3, 2},
      {"hs35mod", 2.5000000e-01, 16.3937, 1, 3, 3, 3, 2},
      {"hs51", 8.8817842e-16, 2.3486094e+16, 3, 5, 7, 5, 2},
      {"hs52", 5.3266476e+00, 6.6637185e+16, 3, 5, 7, 5, 2},
      {"hs53", 4.0930233e+00, 2.3486094e+16, 3, 5, 7, 5, 2},
      {"hs76", -4.6818182e+00, 16.3937, 3, 4, 10, 4, 2},
      {"hues-mod", 3.4824690e+07, 1, 2, 10000, 19899, 10000, 0},
      {"huestis", 3.4824690e+11, 1, 2, 10000, 19899, 10000, 0},
      {"ksip", 5.7579794e-01, 20, 1001, 20, 18411, 20, 0},
      {"laser", 2.4096014e+06, 9.4835780e+10, 1000, 1002, 3000, 1002, 3000},
      {"liswet1", 3.6122402e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet10", 4.9485785e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet11", 4.9523957e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet12", 1.7369274e+03, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet2", 2.4998076e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet3", 2.5001220e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet4", 2.5000112e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet5", 2.5034253e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet6", 2.4995748e+01, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet7", 4.9884089e+02, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet8", 7.1447006e+03, 1, 10000, 10002, 30000, 10002, 0},
      {"liswet9", 1.9632513e+03, 1, 10000, 10002, 30000, 10002, 0},
      {"lotschd", 2.3984159e+03, Inf, 7, 12, 54, 6, 0},
      {"mosarqp1", -9.5287544e+02, 3.6673, 700, 2500, 3422, 2500, 45},
      {"mosarqp2", -1.5974821e+03, 20.0855, 600, 900, 2930, 900, 45},
      {"powell20", 5.2089583e+10, 1, 10000, 10000, 20000, 10000, 0},
      {"primal1", -3.5012965e-02, Inf, 85, 325, 5815, 324, 0},
      {"primal2", -3.3733676e-02, Inf, 96, 649, 8042, 648, 0},
      {"primal3", -1.3575584e-01, Inf, 111, 745, 21547, 744, 0},
      {"primal4", -7.4609083e-01, Inf, 75, 1489, 16031, 1488, 0},
      {"primalc1", -6.1552508e+03, Inf, 9, 230, 2070, 229, 0},
      {"primalc2", -3.5513077e+03, Inf, 7, 231, 1617, 230, 0},
      {"primalc5", -4.2723233e+02, Inf, 8, 287, 2296, 286, 0},
      {"primalc8", -1.8309430e+04, Inf, 8, 520, 4160, 519, 0},
      {"q25fv47", 1.3744448e+07, Inf, 820, 1571, 10400, 446, 59053},
      {"qadlittl", 4.8031886e+05, Inf, 56, 97, 383, 17, 70},
      {"qafiro", -1.5907818e+00, Inf, 27, 32, 83, 3, 3},
      {"qbandm", 1.6352342e+04, Inf, 305, 472, 2494, 25, 16},
      {"qbeaconf", 1.6471206e+05, Inf, 173, 262, 3375, 18, 9},
      {"qbore3d", 3.1002008e+03, Inf, 233, 315, 1429, 28, 50},
      {"qbrandy", 2.8375115e+04, Inf, 220, 249, 2148, 16, 49},
      {"qcapri", 6.6793293e+07, 1.1686697e+11, 271, 353, 1767, 56, 838},
      {"qe226", 2.1265343e+02, Inf, 223, 282, 2578, 67, 897},
      {"qetamacr", 8.6760370e+04, Inf, 400, 688, 2409, 378, 4069},
      {"qfffff80", 8.7314747e+05, Inf, 524, 854, 6227, 278, 1638},
      //{"qforplan",  7.4566315e+09, Inf          ,    161,    421,   4563,     36,     546},  //requires the QPS reader
      //to handle names with spaces
      {"qgfrdxpn", 1.0079059e+11, Inf, 616, 1092, 2377, 54, 108},
      {"qgrow15", -1.0169364e+08, Inf, 300, 645, 5620, 38, 462},
      {"qgrow22", -1.4962895e+08, Inf, 440, 946, 8252, 65, 787},
      {"qgrow7", -4.2798714e+07, Inf, 140, 301, 2612, 30, 327},
      {"qisrael", 2.5347838e+07, Inf, 174, 142, 2269, 42, 656},
      {"qpcblend", -7.8425409e-03, 10, 74, 83, 491, 83, 0},
      //{"qpcboei1",  1.1503914e+07, 10           ,    351,    384,   3485,    384,       0},  //Both fail to seemingly
      //bad conditionning of the active set due to selectionning
      //{"qpcboei2",  8.1719623e+06, 10           ,    166,    143,   1196,    143,       0},  //one constraint to
      //activate instead of another by 1e-13 difference. Need to have a more robust constraint than the basic one.
      {"qpcstair", 6.2043875e+06, 10, 356, 467, 3856, 467, 0},
      {"qpilotno", 4.7285869e+06, Inf, 975, 2172, 13057, 94, 391},
      {"qptest", 4.3718750e+00, 1.6612, 2, 2, 4, 2, 1},
      {"qrecipe", -2.6661600e+02, Inf, 91, 180, 663, 20, 30},
      {"qsc205", -5.8139518e-03, Inf, 205, 203, 551, 11, 10},
      {"qscagr25", 2.0173794e+08, Inf, 471, 500, 1554, 28, 100},
      {"qscagr7", 2.6865949e+07, Inf, 129, 140, 420, 8, 17},
      {"qscfxm1", 1.6882692e+07, Inf, 330, 457, 2589, 56, 677},
      {"qscfxm2", 2.7776162e+07, Inf, 660, 914, 5183, 74, 1057},
      {"qscfxm3", 3.0816355e+07, Inf, 990, 1371, 7777, 89, 1132},
      {"qscorpio", 1.8805096e+03, Inf, 388, 358, 1426, 22, 18},
      {"qscrs8", 9.0456001e+02, Inf, 490, 1169, 3182, 33, 88},
      {"qscsd1", 8.6666667e+00, Inf, 77, 760, 2388, 54, 691},
      {"qscsd6", 5.0808214e+01, Inf, 147, 1350, 4316, 96, 1308},
      {"qscsd8", 9.4076357e+02, Inf, 397, 2750, 8584, 140, 2370},
      {"qsctap1", 1.4158611e+03, Inf, 300, 480, 1692, 36, 117},
      {"qsctap2", 1.7350265e+03, Inf, 1090, 1880, 6714, 141, 636},
      {"qsctap3", 1.4387547e+03, Inf, 1480, 2480, 8874, 186, 861},
      {"qseba", 8.1481801e+07, Inf, 515, 1028, 4352, 96, 550},
      {"qshare1b", 7.2007832e+05, Inf, 117, 225, 1151, 18, 21},
      {"qshare2b", 1.1703692e+04, Inf, 96, 79, 694, 10, 45},
      {"qshell", 1.5726368e+12, Inf, 536, 1775, 3556, 405, 34385},
      {"qship04l", 2.4200155e+06, Inf, 402, 2118, 6332, 14, 42},
      {"qship04s", 2.4249937e+06, Inf, 402, 1458, 4352, 14, 42},
      {"qship08l", 2.3760406e+06, Inf, 778, 4283, 12802, 940, 34025},
      {"qship08s", 2.3857289e+06, Inf, 778, 2387, 7114, 538, 11139},
      {"qship12l", 3.0188766e+06, Inf, 1151, 5427, 16170, 2023, 60205},
      {"qship12s", 3.0569623e+06, Inf, 1151, 2763, 8178, 1042, 16361},
      {"qsierra", 2.3750458e+07, Inf, 1227, 2036, 7302, 122, 61},
      {"qstair", 7.9854528e+06, Inf, 356, 467, 3856, 66, 952},
      {"qstandat", 6.4118384e+03, Inf, 359, 1075, 3031, 138, 666},
      {"s268", 5.7310705e-07, 1176920.3779, 5, 5, 25, 5, 10},
      {"stadat1", -2.8526864e+07, Inf, 3999, 2001, 9997, 2000, 0},
      {"stadat2", -3.2626665e+01, Inf, 3999, 2001, 9997, 2000, 0},
      {"stadat3", -3.5779453e+01, Inf, 7999, 4001, 19997, 4000, 0},
      {"stcqp1", 1.5514356e+05, 831.5172, 2052, 4097, 13338, 4097, 22506},
      {"stcqp2", 2.2327313e+04, 1090.1896, 2052, 4097, 13338, 4097, 22506},
      {"tame", 0.0000000e+00, 1.1568581e+17, 1, 2, 2, 2, 1},
      {"ubh1", 1.1160008e+00, Inf, 12000, 18009, 48000, 6003, 0},
      {"values", -1.3966211e+00, 409752866.825, 1, 202, 202, 202, 3620},
      {"yao", 1.9770426e+02, 1, 2000, 2002, 6000, 2002, 0},
      {"zecevic2", -4.1250000e+00, Inf, 2, 2, 4, 1, 0}};

  for(const auto & p : pbList)
  {
    auto [name, fstar, cond, nbCstr, nbVar, nz, qn, qnz] = p;

    std::cout << name;
    if(cond > 1e8 && cond < Inf)
    {
      std::cout << " skip (cond)" << std::endl;
      continue;
    }
    if(nbVar > 500)
    {
      std::cout << " skip (nbVar)" << std::endl;
      continue;
    }
    if(nbCstr > 1000)
    {
      std::cout << " skip (nbCstr)" << std::endl;
      continue;
    }
    std::cout << std::endl;

    test::QPSReader reader(true);
    std::string path = QPS_TESTS_DIR + name + ".QPS";
    auto [pb, properties] = reader.read(path);

    MatrixXd G = pb.G; // copy for later check
    GoldfarbIdnaniSolver qp(3, 5,
                            false); // Sizes are not the correct ones. We check by the way that the resize is working.
    SolverOptions opt;
    opt.logFlags(LogFlags::ITERATION_BASIC_DETAILS | LogFlags::ACTIVE_SET
                 | LogFlags::ACTIVE_SET_DETAILS
                 //           | LogFlags::ITERATION_ADVANCE_DETAILS
                 | LogFlags::INPUT);
    std::ofstream aof("C:/Work/code/optim/jrl-qp/c++/tests/qplog.m");
    opt.logStream_ = &aof;
    aof.precision(16);
    opt.maxIter_ = std::max(50, 10 * std::max(nbCstr, nbVar));
    qp.options(opt);
    auto ret = qp.solve(pb.G, pb.a, pb.C.transpose(), pb.l, pb.u, pb.xl, pb.xu);
    aof.close();
    if(cond == Inf)
    {
      FAST_CHECK_EQ(ret, TerminationStatus::NON_POS_HESSIAN);
    }
    else
    {
      FAST_CHECK_EQ(ret, TerminationStatus::SUCCESS);
      FAST_CHECK_UNARY(test::testKKT(qp.solution(), qp.multipliers(), G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu, false));
      FAST_CHECK_EQ(qp.objectiveValue() + pb.objCst, doctest::Approx(fstar).epsilon(1e-6));
    }
  }
}

#endif
