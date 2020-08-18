/* Copyright 2020 CNRS-AIST JRL */

#include <jrl-qp/test/problems.h>

namespace jrl::qp::test
{
  FeasibilityConstraints::FeasibilityConstraints(const SeparatedFeasibilityConstraints& cstr)
    : C(cstr.transposedMat?cstr.C.rows():cstr.C.rows()+cstr.E.rows(), cstr.transposedMat?cstr.C.cols()+cstr.E.cols():cstr.C.cols())
    , l(cstr.l.size()+cstr.f.size())
    , u(cstr.u.size() + cstr.f.size())
    , xl(cstr.xl)
    , xu(cstr.xu)
    , transposedMat(cstr.transposedMat)
  {
    assert(cstr.wellFormed());
    if (transposedMat)
    {
      C.leftCols(cstr.E.cols()) = cstr.E;
      C.rightCols(cstr.C.cols()) = cstr.C;
    }
    else
    {
      C.topRows(cstr.E.rows()) = cstr.E;
      C.bottomRows(cstr.C.rows()) = cstr.C;
    }
    l << cstr.f, cstr.l;
    u << cstr.f, cstr.u;
  }

  bool FeasibilityConstraints::wellFormed(bool noEq) const
  {
    // compatible sizes for l, C and u
    bool b1 = l.size() == u.size();
    bool b2 = transposedMat ? (l.size() == C.cols()) : (l.size() == C.rows());
    // compatible sizes for xl, C, xu
    bool b3 = xl.size() == xu.size();
    bool b4 = true;
    if (xl.size() != 0)
    {
      b4 = transposedMat ? (xl.size() == C.rows()) : (xl.size() == C.cols());
    }
    // compatibility of bounds values
    bool b5 = noEq ? (l.array() < u.array()).all() : (l.array() <= u.array()).all();
    bool b6 = (xl.array() <= xu.array()).all();

    return b1 && b2 && b3 && b4 && b5 && b6;
  }

  SeparatedFeasibilityConstraints::SeparatedFeasibilityConstraints(const FeasibilityConstraints& feas)
  {
    assert(feas.wellFormed());
    transposedMat = feas.transposedMat;

    int neq = 0;
    int nineq = 0;
    for (int i = 0; i < feas.l.size(); ++i)
    {
      if (l[i] == u[i]) ++neq;
    }

    neq = 0;
    if (transposedMat)
    {
      C.resize(feas.C.rows(), feas.C.cols() - neq);
      E.resize(feas.C.rows(), neq);
      for (int i = 0; i < feas.l.size(); ++i)
      {
        if (feas.l[i] == feas.u[i])
        {
          E.col(neq) = feas.C.col(i);
          f[neq] = feas.l[i];
          ++neq;
        }
        else
        {
          C.col(nineq) = feas.C.col(i);
          l[nineq] = feas.l[i];
          u[nineq] = feas.u[i];
          ++nineq;
        }
      }
    }
    else
    {
      C.resize(feas.C.rows()-neq, feas.C.cols());
      E.resize(neq, feas.C.cols());
      for (int i = 0; i < feas.l.size(); ++i)
      {
        if (feas.l[i] == feas.u[i])
        {
          E.row(neq) = feas.C.row(i);
          f[neq] = feas.l[i];
          ++neq;
        }
        else
        {
          C.row(nineq) = feas.C.row(i);
          l[nineq] = feas.l[i];
          u[nineq] = feas.u[i];
          ++nineq;
        }
      }
    }
    xl = feas.xl;
    xu = feas.xu;
  }

  bool SeparatedFeasibilityConstraints::wellFormed() const
  {
    bool b1 = FeasibilityConstraints::wellFormed(true);
    bool b2 = transposedMat ? (C.rows() == E.rows()) : (C.cols() == E.cols());
    bool b3 = transposedMat ? (f.size() == E.cols()) : (f.size() == E.rows());
    return b1 && b2 && b3;
  }
}