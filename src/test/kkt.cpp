/* Copyright 2020 CNRS-AIST JRL */

#include <jrl-qp/test/kkt.h>

using namespace Eigen;

namespace
{
// For a constraint bl <= c(x) <= bu with Lagrange multiplier u, and threshold tau_x and tau_u,
// Check that one of the following case is true:
// (1) c(x) == bl and u <= 0
// (2) bl <= c(x) <= bu and u == 0
// (3) cx == bu and u >= 0
bool checkKKTConstraint(double cx, double bl, double bu, double u, double tau_x, double tau_u)
{
  double li = cx - bl;
  double ui = cx - bu;
  bool b1 = std::abs(li) <= tau_x && u <= -tau_u; // constraint active at the lower bound
  bool b2 = li >= -tau_x && ui <= tau_x && std::abs(u) <= tau_u; // constraint inactive
  bool b3 = std::abs(ui) <= tau_x && u >= tau_u; // constraint active at the upper bound
  return b1 || b2 || b3;
}
} // namespace

namespace jrl::qp::test
{
void checkDimensions([[maybe_unused]] int n,
                     [[maybe_unused]] const MatrixConstRef & C,
                     [[maybe_unused]] const VectorConstRef & bl,
                     [[maybe_unused]] const VectorConstRef & bu,
                     [[maybe_unused]] const VectorConstRef & xl,
                     [[maybe_unused]] const VectorConstRef & xu,
                     [[maybe_unused]] bool transposedC)
{
  [[maybe_unused]] int m = static_cast<int>(bl.size());
  [[maybe_unused]] int nb = static_cast<int>(xl.size());
  assert((!transposedC && C.rows() == m && C.cols() == n) || (transposedC && C.rows() == n && C.cols() == m));
  assert(bu.size() == m);
  assert(nb == 0 || nb == n);
  assert(xu.size() == nb);
}

void checkDimensions([[maybe_unused]] const MatrixConstRef & G,
                     [[maybe_unused]] const VectorConstRef & a,
                     const MatrixConstRef & C,
                     const VectorConstRef & bl,
                     const VectorConstRef & bu,
                     const VectorConstRef & xl,
                     const VectorConstRef & xu,
                     bool transposedC)
{
  [[maybe_unused]] int n = static_cast<int>(a.size());
  assert(G.rows() == n && G.cols() == n);
  assert(a.size() == n);
  checkDimensions(n, C, bl, bu, xl, xu, transposedC);
}

void checkDimensions([[maybe_unused]] const VectorConstRef & x,
                     [[maybe_unused]] const VectorConstRef & u,
                     const MatrixConstRef & G,
                     const VectorConstRef & a,
                     const MatrixConstRef & C,
                     const VectorConstRef & bl,
                     const VectorConstRef & bu,
                     const VectorConstRef & xl,
                     const VectorConstRef & xu,
                     bool transposedC)
{
  assert(x.size() == a.size());
  assert(u.size() == bl.size() + xl.size());
  checkDimensions(G, a, C, bl, bu, xl, xu, transposedC);
}

void checkDimensions([[maybe_unused]] const VectorConstRef & x,
                     [[maybe_unused]] const VectorConstRef & u,
                     const MatrixConstRef & C,
                     const VectorConstRef & bl,
                     const VectorConstRef & bu,
                     const VectorConstRef & xl,
                     const VectorConstRef & xu,
                     bool transposedC)
{
  assert(u.size() == bl.size() + xl.size());
  checkDimensions(static_cast<int>(x.size()), C, bl, bu, xl, xu, transposedC);
}

bool testKKT(const VectorConstRef & x,
             const VectorConstRef & u,
             const MatrixConstRef & G,
             const VectorConstRef & a,
             const MatrixConstRef & C,
             const VectorConstRef & bl,
             const VectorConstRef & bu,
             const VectorConstRef & xl,
             const VectorConstRef & xu,
             bool transposedC,
             double tau_p,
             double tau_d)
{
  bool b1 = testKKTStationarity(x, u, G, a, C, bl, bu, xl, xu, transposedC, tau_d);
  bool b2 = testKKTFeasibility(x, u, C, bl, bu, xl, xu, transposedC, tau_p, tau_d);
  return b1 && b2;
}

bool JRLQP_DLLAPI
    testKKT(const VectorConstRef & x, const VectorConstRef & u, const QPProblem<> & pb, double tau_p, double tau_d)
{
  return testKKT(x, u, pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu, pb.transposedMat, tau_p, tau_d);
}

bool testKKTStationarity(const VectorConstRef & x,
                         const VectorConstRef & u,
                         const MatrixConstRef & G,
                         const VectorConstRef & a,
                         const MatrixConstRef & C,
                         const VectorConstRef & bl,
                         const VectorConstRef & bu,
                         const VectorConstRef & xl,
                         const VectorConstRef & xu,
                         bool transposedC,
                         double tau_d)
{
  checkDimensions(x, u, G, a, C, bl, bu, xl, xu, transposedC);
  int n = static_cast<int>(x.size());
  int m = static_cast<int>(bl.size());
  double tau_u = tau_d * (1 + u.template lpNorm<Infinity>());
  VectorXd dL = G * x + a;
  if(xl.size())
  {
    dL += u.tail(n);
  }
  if(transposedC)
  {
    dL.noalias() += C * u.head(m);
  }
  else
  {
    dL.noalias() += C.transpose() * u.head(m);
  }
  double ndL = dL.template lpNorm<Infinity>();
  return ndL <= tau_u;
}

bool JRLQP_DLLAPI testKKTStationarity(const VectorConstRef & x,
                                      const VectorConstRef & u,
                                      const QPProblem<> & pb,
                                      double tau_d)
{
  return testKKTStationarity(x, u, pb.G, pb.a, pb.C, pb.l, pb.u, pb.xl, pb.xu, pb.transposedMat, tau_d);
}

bool testKKTFeasibility(const VectorConstRef & x,
                        const VectorConstRef & u,
                        const MatrixConstRef & C,
                        const VectorConstRef & bl,
                        const VectorConstRef & bu,
                        const VectorConstRef & xl,
                        const VectorConstRef & xu,
                        bool transposedC,
                        double tau_p,
                        double tau_d)
{
  checkDimensions(x, u, C, bl, bu, xl, xu, transposedC);

  double tau_x = tau_p * (1 + x.template lpNorm<Infinity>());
  double tau_u = tau_d * (1 + u.template lpNorm<Infinity>());

  // Check general constraints, see S. Brossette PhD thesis (2016), sec 4.3.5
  VectorXd cx;
  if(transposedC)
    cx = C.transpose() * x;
  else
    cx = C * x;
  for(int i = 0; i < bl.size(); ++i)
  {
    if(!checkKKTConstraint(cx[i], bl[i], bu[i], u[i], tau_x, tau_u)) return false;
  }

  // Check the bounds, if any
  for(int i = 0; i < xl.size(); ++i)
  {
    if(!checkKKTConstraint(x[i], xl[i], xu[i], u[bl.size() + i], tau_x, tau_u)) return false;
  }

  return true;
}

bool JRLQP_DLLAPI testKKTFeasibility(const VectorConstRef & x,
                                     const VectorConstRef & u,
                                     const FeasibilityConstraints & cstr,
                                     double tau_p,
                                     double tau_d)
{
  return testKKTFeasibility(x, u, cstr.C, cstr.l, cstr.u, cstr.xl, cstr.xu, cstr.transposedMat, tau_p, tau_d);
}

} // namespace jrl::qp::test
