/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/decomposition/blockArrowLLT.h>

#include <Eigen/Cholesky>

#include <numeric>

namespace
{
using namespace jrl::qp;

/** An helper struct that helps seeing a couple (diag, side as the representaion
 * of a matrix with block diagonal + last block row
 */
template<bool Up>
struct get
{
};

template<>
struct get<false>
{
  static auto D(const std::vector<MatrixRef> & diag, int i)
  {
    return diag[i];
  }

  static auto B(const std::vector<MatrixRef> & side, int i)
  {
    return side[i];
  }
};

template<>
struct get<true>
{
  static auto D(const std::vector<MatrixRef> & diag, int i)
  {
    return diag[(i + 1) % diag.size()];
  }

  static auto B(const std::vector<MatrixRef> & side, int i)
  {
    return side[i].transpose();
  }
};
} // namespace

namespace jrl::qp::decomposition
{
template<bool Up>
bool blockArrowLLT_(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & side)
{
  assert(diag.size() == side.size() + 1);

  int b = static_cast<int>(diag.size());
  auto Db = get<Up>::D(diag, b - 1);

  for(int i = 0; i < b - 1; ++i)
  {
    // Li = chol(Di)
    auto Di = get<Up>::D(diag, i);
    auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Di);
    if(ret > 0) return false;

    // Bi = Bi*Li^-T
    auto Li = Di.template triangularView<Eigen::Lower>();
    if constexpr(Up)
      Li.solveInPlace<Eigen::OnTheLeft>(side[i]);
    else
      Li.transpose().solveInPlace<Eigen::OnTheRight>(side[i]);

    // Db -= Bi Bi^T
    Db.template selfadjointView<Eigen::Lower>().rankUpdate(get<Up>::B(side, i), -1.);
  }
  // Lb = chol(Db)
  auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Db);

  return ret <= 0;
}

bool blockArrowLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & side, bool up)
{
  if(up)
    return blockArrowLLT_<true>(diag, side);
  else
    return blockArrowLLT_<false>(diag, side);
}

template<bool Up>
void blockArrowLSolve_(const std::vector<MatrixRef> & diag,
                       const std::vector<MatrixRef> & side,
                       MatrixRef v,
                       int start,
                       int end)
{
  assert(diag.size() == side.size() + 1);

  int b = static_cast<int>(diag.size());
  int n = 0;

  for(int i = 0; i < b - 1; ++i)
  {
    auto Di = get<Up>::D(diag, i);
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    int s = std::max(start - n, 0);
    if((ni < s || end <= n))
    {
      assert(v.middleRows(n, ni).isZero());
      n += ni;
      continue;
    }

    auto Li = Di.bottomRightCorner(ni - s, ni - s).template triangularView<Eigen::Lower>();
    auto vi = v.middleRows(n + s, ni - s);
    Li.solveInPlace(vi);
    v.bottomRows(diag.back().rows()).noalias() -= get<Up>::B(side, i).middleCols(s, ni - s) * vi;
    n += ni;
  }
  auto Db = get<Up>::D(diag, b - 1);
  assert(Db.rows() == Db.cols());
  int nb = static_cast<int>(Db.rows());
  auto Lb = Db.template triangularView<Eigen::Lower>();
  auto vb = v.middleRows(n, nb);
  Lb.solveInPlace(vb);
}

void blockArrowLSolve(const std::vector<MatrixRef> & diag,
                      const std::vector<MatrixRef> & side,
                      bool up,
                      MatrixRef v,
                      int start,
                      int end)
{
  if(end < 0) end = static_cast<int>(v.rows());

  if(up)
  {
    Eigen::MatrixXd tmp = v;
    int n0 = static_cast<int>(diag.front().rows());
    auto s = v.rows() - n0;
    v.topRows(s) = tmp.bottomRows(s);
    v.bottomRows(n0) = tmp.topRows(n0);
    blockArrowLSolve_<true>(diag, side, v, std::max(0, start - n0), std::max(0, end - n0));
  }
  else
  {
    blockArrowLSolve_<false>(diag, side, v, start, end);
  }
}

template<bool Up>
void blockArrowLTransposeSolve_(const std::vector<MatrixRef> & diag,
                                const std::vector<MatrixRef> & side,
                                MatrixRef v,
                                int start,
                                int end)
{
  int b = static_cast<int>(diag.size());
  int s = static_cast<int>(v.rows());
  bool zero = false;

  auto Db = get<Up>::D(diag, b - 1);
  int nb = static_cast<int>(Db.rows());
  auto vb = v.bottomRows(nb);
  if(end > s - nb)
  {
    int r = end - s + nb;
    auto Lb = Db.topLeftCorner(r, r).template triangularView<Eigen::Lower>();
    Lb.transpose().solveInPlace(vb.topRows(r));
  }
  else
    zero = true;

  int n = 0;
  //[OPTIM] This loop can be fully parallelized
  for(int i = 0; i < b - 1; ++i)
  {
    auto Di = get<Up>::D(diag, i);
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    auto vi = v.middleRows(n, ni);
    if(zero) // vb is zero
    {
      if(start >= n + ni) // vi is zero
      {
        n += ni;
        continue; // rhs is zero, no need to perform the inversion
      }
    }
    else
      vi.noalias() -= get<Up>::B(side, i).transpose() * vb;

    if(end >= n) // if not, we know that both vb and vi are zero
    {
      if(end >= n + ni)
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        Li.transpose().solveInPlace(vi);
      }
      else
      {
        assert(zero);
        int r = end - n;
        auto Li = Di.topLeftCorner(r, r).template triangularView<Eigen::Lower>();
        Li.transpose().solveInPlace(vi.topRows(r));
      }
    }
    n += ni;
  }
}

void blockArrowLTransposeSolve(const std::vector<MatrixRef> & diag,
                               const std::vector<MatrixRef> & side,
                               bool up,
                               MatrixRef v,
                               int start,
                               int end)
{
  if(end < 0) end = static_cast<int>(v.rows());

  if(up)
  {
    blockArrowLTransposeSolve_<true>(diag, side, v, start, end);
    Eigen::MatrixXd tmp = v;
    int n0 = static_cast<int>(diag.front().rows());
    auto s = v.rows() - n0;
    v.bottomRows(s) = tmp.topRows(s);
    v.topRows(n0) = tmp.bottomRows(n0);
  }
  else
  {
    blockArrowLTransposeSolve_<false>(diag, side, v, start, end);
  }
}

} // namespace jrl::qp::decomposition
