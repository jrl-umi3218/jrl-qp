/* Copyright 2020 CNRS-AIST JRL
 */

#include "problemAdaptors.h"

using namespace jrl::qp::test;
using namespace Eigen;

namespace
{
  constexpr double Inf = std::numeric_limits<double>::infinity();


  bool hasFiniteElements(const Eigen::VectorXd& x)
  {
    return x.size() == 0 || ((x.array() > -Inf).any() && (x.array() < Inf).any());
  }

  bool doubleSided(const QPProblem<true>& pb)
  {
    return hasFiniteElements(pb.l) && hasFiniteElements(pb.u);
  }


  //number of equality in the problem
  int me(const QPProblem<true>& pb)
  {
    if (pb.transposedMat)
      return pb.E.cols();
    else
      return pb.E.rows();
  }

  //number of inequality in the problem
  int mi(const QPProblem<true>& pb)
  {
    if (pb.transposedMat)
      return pb.C.cols();
    else
      return pb.C.rows();
  }
}

namespace jrl::qp::test
{

  EigenQuadprogPb::EigenQuadprogPb(const QPProblem<true>& pb)
  {
    assert(pb.wellFormed());
    Q = pb.G;
    c = pb.a;

    int nVar = pb.a.size();

    int nIneq = 0;
    int nl, nu, nxl, nxu;
    nl = nu = nxl = nxu = 0;
    int startl = 0;
    if (hasFiniteElements(pb.l))
    {
      nl = mi(pb);
      nIneq += nl;
    }
    int startu = nIneq;
    if (hasFiniteElements(pb.u))
    {
      nu = mi(pb);
      nIneq += nu;
    }
    int startxl = nIneq;
    if (hasFiniteElements(pb.xl))
    {
      nxl = pb.xl.size();
      nIneq += nxl;
    }
    int startxu = nIneq;
    if (hasFiniteElements(pb.xu))
    {
      nxu = pb.xu.size();
      nIneq += nxu;
    }
    Aineq.resize(nIneq, nVar);
    bineq.resize(nIneq);

    if (pb.transposedMat)
    {
      Aeq = pb.E.transpose();
      if (nl) Aineq.middleRows(startl, nl) = -pb.C.transpose();
      if (nu) Aineq.middleRows(startu, nu) = pb.C.transpose();
    }
    else
    {
      Aeq = pb.E;
      if (nl) Aineq.middleRows(startl, nl) = -pb.C;
      if (nu) Aineq.middleRows(startu, nu) = pb.C;
    }
    Aineq.middleRows(startxl, nxl + nxu).setZero();
    Aineq.middleRows(startxl, nxl).diagonal().setConstant(-1); // -I
    Aineq.middleRows(startxu, nxu).diagonal().setOnes();       // I
    beq = pb.f;
    if (nl) bineq.segment(startl, nl) = -pb.l;
    if (nu) bineq.segment(startu, nu) = pb.u;
    if (nxl) bineq.segment(startxl, nxl) = -pb.xl;
    if (nxu) bineq.segment(startxu, nxu) = pb.xu;
  }


  LssolPb::LssolPb(const QPProblem<true>& pb)
  {
    assert(pb.wellFormed());
    Q = pb.G;
    p = pb.a;
    int nVar = p.size();
    int nEq = me(pb);
    int nIneq = mi(pb);

    C.resize(nEq + nIneq, nVar);
    if (pb.transposedMat)
    {
      C.topRows(nEq) = pb.E.transpose();
      C.bottomRows(nIneq) = pb.C.transpose();
    }
    else
    {
      C.topRows(nEq) = pb.E;
      C.bottomRows(nIneq) = pb.C;
    }

    l.resize(nVar + nEq + nIneq);
    u.resize(nVar + nEq + nIneq);

    if (pb.xl.size() > 0)
    {
      l.head(nVar) = pb.xl;
      u.head(nVar) = pb.xu;
    }
    else
    {
      l.head(nVar).setConstant(-1e30);
      u.head(nVar).setConstant(1e30);
    }
    l.segment(nVar, nEq) = pb.f;
    u.segment(nVar, nEq) = pb.f;
    l.tail(nIneq) = pb.l;
    u.tail(nIneq) = pb.u;
  }

  QLDPb::QLDPb(const QPProblem<true>& pb)
  {
    assert(pb.wellFormed());
    Q = pb.G;
    c = pb.a;

    int nVar = pb.a.size();
    int nEq = me(pb);
    
    int nIneq = nEq;
    int nl, nu;
    nl = nu = 0;
    int startl = nIneq;
    if (hasFiniteElements(pb.l))
    {
      nl = mi(pb);
      nIneq += nl;
    }
    int startu = nIneq;
    if (hasFiniteElements(pb.u))
    {
      nu = mi(pb);
      nIneq += nu;
    }
    A.resize(nIneq, nVar);
    b.resize(nIneq);

    if (pb.transposedMat)
    {
      A.topRows(nEq) = pb.E.transpose();
      if (nl) A.middleRows(startl, nl) = pb.C.transpose();
      if (nu) A.middleRows(startu, nu) = -pb.C.transpose();
    }
    else
    {
      A.topRows(nEq) = pb.E;
      if (nl) A.middleRows(startl, nl) = pb.C;
      if (nu) A.middleRows(startu, nu) = -pb.C;
    }
    b.head(nEq) = -pb.f;
    if (nl) b.segment(startl, nl) = -pb.l;
    if (nu) b.segment(startu, nu) = pb.u;

    if (pb.xl.size() > 0)
      xl = pb.xl;
    else
      xl = VectorXd::Constant(nVar, -Inf);

    if (pb.xu.size() > 0)
      xu = pb.xu;
    else
      xu = VectorXd::Constant(nVar, Inf);
  }
  
  EiQuadprogPb::EiQuadprogPb(const QPProblem<true>& pb)
  {
    EigenQuadprogPb tmp(pb);
    G = tmp.Q;
    g0 = tmp.c;
    CE = tmp.Aeq.transpose();
    ce0 = -tmp.beq;
    CI = -tmp.Aineq.transpose();
    ci0 = tmp.bineq;
  }

  GIPb::GIPb(const QPProblem<true>& pb)
    : QPProblem<>(pb)
  {
    if (!transposedMat)
      C.transposeInPlace();
  }
}
