/* Copyright 2020 CNRS-AIST JRL
 */

#include "utils.h"

using namespace Eigen;

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
      double li = cx[i] - bl[i];
      double ui = cx[i] - bu[i];
      bool b1 = std::abs(li) <= tau_x && u[i] <= -tau_u; // constraint active at the lower bound
      bool b2 = li >= -tau_x && ui <= tau_x && std::abs(u[i]) <= tau_u; // constraint inactive
      bool b3 = ui <= tau_x && u[i] >= tau_u; // constraint active at the upper bound
      if (!(b1 || b2 || b3)) return false;
    }

    // Check the bounds, if any
    for (int i = 0; i < xl.size(); ++i)
    {
      double li = x[i] - xl[i];
      double ui = x[i] - xu[i];
      bool b1 = std::abs(li) <= tau_x && u[m+i] <= -tau_u; // constraint active at the lower bound
      bool b2 = li >= tau_x && ui <= tau_x && std::abs(u[m+i]) <= tau_u; // constraint inactive
      bool b3 = ui <= tau_x && u[m+i] >= tau_u; // constraint active at the upper bound
      if (!(b1 || b2 || b3)) return false;
    }

    return true;
  }

}