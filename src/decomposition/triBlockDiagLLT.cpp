/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/decomposition/triBlockDiagLLT.h>

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
} // namespace jrl::qp::decomposition