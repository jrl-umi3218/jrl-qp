/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/decomposition/triBlockDiagLLT.h>

#include "..\..\include\jrl-qp\decomposition\triBlockDiagLLT.h"
#include <Eigen/Cholesky>

namespace jrl::qp::decomposition
{
/** Cholesky decomposition of a block-tridiagonal (symmetric) matrix.
 *
 * \param diag blocks on the diagonal. Only the lower triangular part is actually used.
 * \param subDiag blocks under the diagonal
 *
 * The decomposition is done in place.
 */
bool triBlockDiagLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & subDiag)
{
  assert(diag.size() == subDiag.size() + 1);

  size_t b = diag.size();
  for(size_t i = 0; i < b - 1; ++i)
  {
    // Li = chol(Di)
    auto Di = diag[i];
    auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Di);
    if(ret > 0) return false;

    // Si = Si*Li^-T
    auto Li = Di.template triangularView<Eigen::Lower>();
    Li.transpose().solveInPlace<Eigen::OnTheRight>(subDiag[i]);

    // L[i+1] -= Si Si^T
    auto Li1 = diag[i + 1];
    Li1.template selfadjointView<Eigen::Lower>().rankUpdate(subDiag[i], -1.);
  }
  // Lb = chol(Db)
  auto Db = diag.back();
  auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Db);
  if(ret > 0) return false;

  return true;
}
void triBlockDiagLSolve(const std::vector<MatrixRef> & diag,
                        const std::vector<MatrixRef> & subDiag,
                        MatrixRef v,
                        int start)
{
  assert(diag.size() == subDiag.size() + 1);

  int n = 0;
  int ni = 0;
  int l = 0;
  int li = 0;
  bool zero = true;
  for(size_t i = 0; i < diag.size(); ++i)
  {
    auto Di = diag[i];
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
        zero = false;
      }
      else
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        auto vi = v.middleRows(n, ni);
        vi.noalias() -= subDiag[i - 1] * v.middleRows(l, li);
        Li.solveInPlace(vi);
      }
    }

    l = n;
    li = ni;
    n += ni;
  }
  assert(n == v.rows());
}

void triBlockDiagLTransposeSolve(const std::vector<MatrixRef> & diag,
                                 const std::vector<MatrixRef> & subDiag,
                                 MatrixRef v,
                                 int end)
{
  int n = static_cast<int>(v.rows());
  int ni = 0;
  int l = 0;
  int li = 0;
  bool zero = true;

  if(end < 0) end = n;

  for(int i = static_cast<int>(diag.size()) - 1; i >= 0; --i)
  {
    auto Di = diag[i];
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    if(n - ni < end)
    {
      if(zero)
      {
        int r = end - n + ni;
        auto Li = Di.topLeftCorner(r, r).template triangularView<Eigen::Lower>();
        auto vi = v.middleRows(n - ni, r);
        Li.transpose().solveInPlace(vi);
        zero = false;
      }
      else
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        auto vi = v.middleRows(n - ni, ni);
        vi.noalias() -= subDiag[i].transpose() * v.middleRows(l - li, li);
        Li.transpose().solveInPlace(vi);
      }
    }

    l = n;
    li = ni;
    n -= ni;
  }
  assert(n == 0);
}
} // namespace jrl::qp::decomposition