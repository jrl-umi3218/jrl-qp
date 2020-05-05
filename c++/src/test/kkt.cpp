/* Copyright 2020 CNRS-AIST JRL
 */

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
    bool b3 = ui <= tau_x && u >= tau_u; // constraint active at the upper bound
    return b1 || b2 || b3;
  }
}

namespace jrlqp::test
{
  void checkDimensions([[maybe_unused]] int n,
                       [[maybe_unused]] const MatrixConstRef& C, [[maybe_unused]] const VectorConstRef& bl, [[maybe_unused]] const VectorConstRef& bu,
                       [[maybe_unused]] const VectorConstRef& xl, [[maybe_unused]] const VectorConstRef& xu,
                       [[maybe_unused]] bool transposedC)
  {
    [[maybe_unused]] int m = bl.size();
    [[maybe_unused]] int nb = xl.size();
    assert((!transposedC && C.rows() == m && C.cols() == n)
      || (transposedC && C.rows() == n && C.cols() == m));
    assert(bu.size() == m);
    assert(nb == 0 || nb == n);
    assert(xu.size() == nb);
  }

  void checkDimensions([[maybe_unused]] const MatrixConstRef& G, [[maybe_unused]] const VectorConstRef& a,
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                       const VectorConstRef& xl, const VectorConstRef& xu,
                       bool transposedC)
  {
    [[maybe_unused]] int n = a.size();
    assert(G.rows() == n && G.cols() == n);
    assert(a.size() == n);
    checkDimensions(n, C, bl, bu, xl, xu, transposedC);
  }

  void checkDimensions([[maybe_unused]] const VectorConstRef& x, [[maybe_unused]] const VectorConstRef& u,
                       const MatrixConstRef& G, const VectorConstRef& a,
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                       const VectorConstRef& xl, const VectorConstRef& xu,
                       bool transposedC)
  {
    assert(x.size() == a.size());
    assert(u.size() == bl.size() + xl.size());
    checkDimensions(G, a, C, bl, bu, xl, xu, transposedC);
  }

  void checkDimensions([[maybe_unused]] const VectorConstRef& x, [[maybe_unused]] const VectorConstRef& u, 
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu, const VectorConstRef& xl, const VectorConstRef& xu, bool transposedC)
  {
    assert(u.size() == bl.size() + xl.size());
    checkDimensions(x.size(), C, bl, bu, xl, xu, transposedC);
  }

  bool testKKT(const VectorConstRef& x, const VectorConstRef& u,
               const MatrixConstRef& G, const VectorConstRef& a, 
               const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu, 
               const VectorConstRef& xl, const VectorConstRef& xu, 
                bool transposedC, double tau_p, double tau_d)
  {
    bool b1 = testKKTStationarity(x, u, G, a, C, bl, bu, xl, xu, transposedC, tau_d);
    bool b2 = testKKTFeasibility(x, u, C, bl, bu, xl, xu, transposedC, tau_p, tau_d);
    return b1 && b2;
  }

  bool testKKTStationarity(const VectorConstRef& x, const VectorConstRef& u, 
                           const MatrixConstRef& G, const VectorConstRef& a, 
                           const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu, 
                           const VectorConstRef& xl, const VectorConstRef& xu, 
                           bool transposedC, double tau_d)
  {
    checkDimensions(x, u, G, a, C, bl, bu, xl, xu, transposedC);
    int n = x.size();
    int m = bl.size();
    double tau_u = tau_d * (1 + u.template lpNorm<Infinity>());
    VectorXd dL = G * x + a;
    if (xl.size())
    {
      dL += u.tail(n);
    }
    if (transposedC)
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

  bool testKKTFeasibility(const VectorConstRef& x, const VectorConstRef& u,
                          const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                          const VectorConstRef& xl, const VectorConstRef& xu, 
                          bool transposedC, double tau_p, double tau_d)
  {
    checkDimensions(x, u, C, bl, bu, xl, xu, transposedC);
    int n = x.size();
    int m = bl.size();

    double tau_x = tau_p * (1 + x.template lpNorm<Infinity>());
    double tau_u = tau_d * (1 + u.template lpNorm<Infinity>());

    // Check general constraints see S. Brossette PhD thesis (2016), sec 4.3.5
    VectorXd cx;
    if (transposedC) cx = C.transpose() * x;
    else cx = C * x;
    for (int i = 0; i < m; ++i)
    {
      if (!checkKKTConstraint(cx[i], bl[i], bu[i], u[i], tau_x, tau_u))
        return false;
    }

    // Check the bounds, if any
    for (int i = 0; i < xl.size(); ++i)
    {
      if (!checkKKTConstraint(x[i], xl[i], xu[i], u[m+i], tau_x, tau_u))
        return false;
    }

    return true;
  }

}