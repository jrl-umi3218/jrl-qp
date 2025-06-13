/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

namespace jrl::qp::decomposition
{
/** Cholesky decomposition of a matrix of the form
 *
 * \f$ \begin{bmatrix}
 *    D_1  & S_1^T &       &       &       \\
 *    S_1  &  D_2  & S_2^T &       &       \\
 *         &  S_2  &  D_3  & S_3^T &       \\
 *         &       &\ddots &\ddots &\ddots \\
 * \end{bmatrix}\f$
 *
 * \param diag blocks \f$ D_i \f$ on the diagonal. Only the lower triangular
 * part is actually used.
 * \param subDiag blocks \f$ S_i \f$ under the diagonal.
 *
 * The decomposition is done in place: the Cholesky factor is
 *
 * \f$ \begin{bmatrix}
 *    L_1  &       &       &       \\
 *    B_1  &  L_2  &       &       \\
 *         &  B_2  &  L_3  &       \\
 *         &       &\ddots &\ddots \\
 * \end{bmatrix}\f$
 *
 * and upon return \f$ D_i \f$ contains \f$ L_i \f$ and \f$ S_i \f$ contains
 * \f$ B_i \f$.
 * Only the lower triangular part of \f$ D_i \f$ is used to store \f$ L_i \f$.
 * Its upper part remains whatever was there originally.
 */
JRLQP_DLLAPI bool triBlockDiagLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & subDiag);

/** Solve in place the system L X = M where L is the triangular factor obtained
 * from triBlockDiagLLT.
 *
 * \param diag blocks on the diagonal of L. Only the lower triangular part is
 * actually used.
 * \param subDiag blocks under the diagonal.
 * \param M right hand side of the equation (matrix or vector). Contains the
 * solution upon return.
 * \param start First row of M that is not 0. Useful for optimizing computations.
 */
JRLQP_DLLAPI void triBlockDiagLSolve(const std::vector<MatrixRef> & diag,
                                     const std::vector<MatrixRef> & subDiag,
                                     MatrixRef M,
                                     int start = 0);

/** Solve in place the system L^T X = M where L is the triangular factor
 * obtained from triBlockDiagLLT.
 *
 * \param diag blocks on the diagonal of L. Only the lower triangular part is
 * actually used.
 * \param subDiag blocks under the diagonal.
 * \param M right hand side of the equation (matrix or vector). Contains the
 * solution upon return.
 * \param end Writing M = [N 0]^T, \p end is the index of the first row of the
 * terminal 0 block. If \p end<0 (default value), M has no terminal 0 block.
 */
JRLQP_DLLAPI void triBlockDiagLTransposeSolve(const std::vector<MatrixRef> & diag,
                                              const std::vector<MatrixRef> & subDiag,
                                              MatrixRef M,
                                              int end = -1);
} // namespace jrl::qp::decomposition
