/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/decomposition/blockArrowLLT.h>

#include <Eigen/Cholesky>

#include <numeric>

namespace jrl::qp::decomposition
{
bool blockArrowLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & side, bool up)
{
  assert(diag.size() == side.size() + 1);

  int b = static_cast<int>(diag.size());
  int start, end, last;
  if(up)
  {
    start = 1;
    end = b;
    last = 0;
  }
  else
  {
    start = 0;
    end = b - 1;
    last = b - 1;
  }

  for(int i = start; i < end; ++i)
  {
    // Li = chol(Di)
    auto Di = diag[i];
    auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Di);
    if(ret > 0) return false;

    // Si = Si*Li^-T
    // D[b-1] -= Si Si^T
    auto Li = Di.template triangularView<Eigen::Lower>();
    auto Db = diag[last];
    if(up)
    {
      Li.solveInPlace<Eigen::OnTheLeft>(side[i - 1]);
      Db.template selfadjointView<Eigen::Lower>().rankUpdate(side[i - 1].transpose(), -1.);
    }
    else
    {
      Li.transpose().solveInPlace<Eigen::OnTheRight>(side[i]);
      Db.template selfadjointView<Eigen::Lower>().rankUpdate(side[i], -1.);
    }
  }
  // Lb = chol(Db)
  auto Db = diag[last];
  auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Db);
  if(ret > 0) return false;

  return true;
}

void blockArrowLSolve(const std::vector<MatrixRef> & diag,
                      const std::vector<MatrixRef> & side,
                      bool up,
                      MatrixRef v,
                      int start)
{
  assert(diag.size() == side.size() + 1);

  int b = static_cast<int>(diag.size());
  int n = 0;
  int shift = up ? 1 : 0;
  bool zero = true;

  for(size_t i = 0; i <b; ++i)
  {
    auto Di = diag[(i+shift)%b];
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    if(n + ni >= start)
    {
      if(zero)
      {
        int r = n + ni - start;
        auto Li = Di.bottomRightCorner(r, r).template triangularView<Eigen::Lower>();
        auto vi = v.middleRows(start, r);
        Li.solveInPlace(vi);
        if(i < b - 1)
        {
          if(up)
            v.bottomRows(diag.back().rows()).noalias() -= side[i].bottomRows(r).transpose() * vi;
          else
            v.bottomRows(diag.back().rows()).noalias() -= side[i].rightCols(r) * vi;
        }
        zero = false;
      }
      else
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        auto vi = v.middleRows(n, ni);
        Li.solveInPlace(vi);
        if(i < b - 1)
        {
          if(up)
            v.bottomRows(diag.back().rows()).noalias() -= side[i].transpose() * vi;
          else
            v.bottomRows(diag.back().rows()).noalias() -= side[i] * vi;
        }
      }
    }

    n += ni;
  }
}

void blockArrowLTransposeSolve(const std::vector<MatrixRef> & diag,
                               const std::vector<MatrixRef> & side,
                               bool up,
                               MatrixRef v,
                               int end)
{
  int b = static_cast<int>(diag.size());
  int s = static_cast<int>(v.rows());
  int shift = up ? 1 : 0;
  bool zero = false;

  if(end < 0) end = s;

  auto Db = up ? diag.front() : diag.back();
  int nb = static_cast<int>(Db.rows());
  auto vb = up ? v.topRows(nb) : v.bottomRows(nb);
  int n = up ? 0 : nb;
  if(end > s - nb)
  {
    int r = end - s + nb;
    auto Lb = Db.topLeftCorner(r, r).template triangularView<Eigen::Lower>();
    Lb.transpose().solveInPlace(vb.topRows(r));
  }
  else
    zero = up ? false : true; //[OPTIM] if v starts with 0, we can do better

  //[OPTIM] This loop can be fully parallelized
  for(int i = 0; i < b-1; ++i)
  {
    auto Di = diag[i + shift];
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    auto Li = Di.template triangularView<Eigen::Lower>();
    auto vi = v.middleRows(n, ni);
    if(!zero)
      vi.noalias() -= side[i].transpose() * vb;
    if(end >= n)
    {
      if(end>=n+ni)
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        Li.transpose().solveInPlace(vi);
      }
      else
      {
        assert(zero);
        int r = n +ni - end;
        auto Li = Di.topLeftCorner(r,r).template triangularView<Eigen::Lower>();
        Li.transpose().solveInPlace(vi.topRows(r));
      }
    }
    n += ni;
  }
}
} // namespace jrl::qp::decomposition
