/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/decomposition/triBlockDiagLLT.h>

#include <Eigen/Cholesky>

namespace jrl::qp::decomposition
{
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
    Li.transpose().template solveInPlace<Eigen::OnTheRight>(subDiag[i]);

    // D[i+1] -= Si Si^T
    auto Di1 = diag[i + 1];
    Di1.template selfadjointView<Eigen::Lower>().rankUpdate(subDiag[i], -1.);
  }
  // Lb = chol(Db)
  auto Db = diag.back();
  auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(Db);
  if(ret > 0) return false;

  return true;
}

void triBlockDiagLSolve(const std::vector<MatrixRef> & diag,
                        const std::vector<MatrixRef> & subDiag,
                        MatrixRef M,
                        int start)
{
  // We want to solve
  // | L1   0   0 ...| | X1 |   | M1 |
  // | B1  L2   0 ...| | X2 | = | M2 |
  // |  0  B2  L3 ...| | X3 |   | M3 |
  // |       ...     | |... |   |... |
  //
  // We have L1 X1 = M1
  //         L2 X2 = M2 - B1 X1
  //         L3 X3 = M3 - B2 X2
  //         ...
  // Whenever Mi - B[i-1] X[i-1] is zero, Xi is zero and we can skip this block.
  // If it is not the case for i0, then it is not the case for any i>i0.

  assert(diag.size() == subDiag.size() + 1);

  int n = 0;
  int ni = 0;
  int l = 0;
  int li = 0;
  bool zero = true; //  Mi - B[i-1] X[i-1] is zero
  for(size_t i = 0; i < diag.size(); ++i)
  {
    auto Di = diag[i];
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    // If Mi is non zero, we perform the solve, otherwise, we can skip it
    if(n + ni >= start)
    {
      if(zero)
      {
        // If previous Xi was zero, this is the first time we have to solve Li Xi = Mi.
        // The first rows of Mi can still be zero, so we solve only for the non-zero rows
        int r = n + ni - start;
        auto Li = Di.bottomRightCorner(r, r).template triangularView<Eigen::Lower>();
        auto Mi = M.middleRows(start, r);
        Li.solveInPlace(Mi);
        zero = false; // From now Mi - B[i-1] X[i-1] can't be zero
      }
      else
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        auto Mi = M.middleRows(n, ni);
        // Mi = Mi - B[i-1] X[i-1]
        Mi.noalias() -= subDiag[i - 1] * M.middleRows(l, li);
        // Mi = Li^-1 Mi
        Li.solveInPlace(Mi);
      }
    }

    l = n;
    li = ni;
    n += ni;
  }
  assert(n == M.rows());
}

void triBlockDiagLTransposeSolve(const std::vector<MatrixRef> & diag,
                                 const std::vector<MatrixRef> & subDiag,
                                 MatrixRef M,
                                 int end)
{
  // We want to solve
  // |                ...              | |  ...   |   |  ...   |
  // | ... L[b-2]^T B[b-2]^T      0    | | X[b-2] |   | M[b-2] |
  // | ...     0    L[b-1]^T  B[b-1]^T | | X[b-1] | = | M[b-1] |
  // | ...     0        0       Lb^T   | |   Xb   |   |   Mb   |
  //
  // We have Lb^T Xb = Mb
  //         L[b-1]^T X[b-1] = M[b-1] - B[b-1]^T Xb
  //         L[b-2]^T X[b-2] = M[b-2] - B[b-2]^T X[b-1]
  //         ...
  // Whenever Mi - B[i] X[i+1] is zero, Xi is zero and we can skip this block.
  // If it is not the case for i0, then it is not the case for any i<i0.

  int n = static_cast<int>(M.rows());
  int ni = 0;
  int l = 0;
  int li = 0;
  bool zero = true; //Mi - B[i] X[i+1] is zero

  if(end < 0) end = n;

  for(int i = static_cast<int>(diag.size()) - 1; i >= 0; --i)
  {
    auto Di = diag[i];
    assert(Di.rows() == Di.cols());
    int ni = static_cast<int>(Di.rows());

    // If Mi is non zero, we perform the solve, otherwise, we can skip it
    if(n - ni < end)
    {
      if(zero)
      {
        // If previous Xi was zero, this is the first time we have to solve Li Xi = Mi.
        // The last rows of Mi can still be zero, so we solve only for the non-zero rows.
        int r = end - n + ni;
        auto Li = Di.topLeftCorner(r, r).template triangularView<Eigen::Lower>();
        auto Mi = M.middleRows(n - ni, r);
        Li.transpose().solveInPlace(Mi);
        zero = false;
      }
      else
      {
        auto Li = Di.template triangularView<Eigen::Lower>();
        auto Mi = M.middleRows(n - ni, ni);
        // Mi = Mi - Bi^T X[i+1]
        Mi.noalias() -= subDiag[i].transpose() * M.middleRows(l - li, li);
        // Mi = Li^-T Mi
        Li.transpose().solveInPlace(Mi);
      }
    }

    l = n;
    li = ni;
    n -= ni;
  }
  assert(n == 0);
}
} // namespace jrl::qp::decomposition