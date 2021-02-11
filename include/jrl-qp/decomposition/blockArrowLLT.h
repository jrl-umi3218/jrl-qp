/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

namespace jrl::qp::decomposition
{
/** Cholesky decomposition of a matrix of the form (arrow-like)
 *
 * \f$ \begin{bmatrix}
 *    D_1  & S_1^T & S_2^T & \cdots \\
 *    S_1  &  D_2  &       &        \\
 *    S_2  &       &  D_3  &        \\
 *  \vdots &       &       & \ddots \\
 * \end{bmatrix}\f$ (up = \c true) or 
  *\f$ \begin{bmatrix}
 *    D_1  &       &         &   S_1^T   \\
 *         &\ddots &         &   \vdots  \\
 *         &       & D_{b-1} & S_{b-1}^T \\
 *    S_1  &\cdots & S_{b-1} &    Db     \\
 * \end{bmatrix}\f$ (up = \c false)
 *
 * \param diag blocks \f$ D_i \f$ on the diagonal. Only the lower triangular
 * part is actually used.
 * \param side blocks \f$ S_i \f$.
 * \param up Whether the arrow is poiting up or down.
 *
 * If up = \c false, the decomposition \f$ M = L L^T\f$ is performed.
 * If up = \c true, the decomposition \f$ P^T M P = L L^T \f$ is performed, where
 * 
 * \f$ \begin{bmatrix}
 *    0 &\cdots & 0 & I \\
 *    I &       &   & 0 \\
 *      &\ddots &   & \vdots \\
 *      &       & I & 0 \\
 * \end{bmatrix}\f$
 * is such that \f$ P^T M P  = \begin{bmatrix}
 *    D_2   &       &           &   S_1    \\
 *          &\ddots &           &   \vdots \\
 *          &       &    D_b    &  S_{b-1} \\
 *   S_1^T  &\cdots & S_{b-1}^T &    D1    \\
 * \end{bmatrix} \f$
 * 
 * Then, \f$ L \f$ is such that
 *
 * \f$ L = \begin{bmatrix}
 *    L_1  &       &       &       \\
 *         &  L_2  &       &       \\
 *         &       &\ddots &       \\
 *    B_1  &  B_2  &\cdots &  L_b  \\
 * \end{bmatrix}\f$ (up = \c false) or 
 * \f$ L = \begin{bmatrix}
 *    L_2  &       &         &       \\
 *         &\ddots &         &       \\
 *         &       &   L_b   &       \\
 *    B_1  &\cdots & B_{b-1} &  L_1  \\
 * \end{bmatrix}\f$ (up = \c true)
 *
 * and upon return \f$ D_i \f$ contains \f$ L_i \f$, and \f$ S_i \f$ contains
 * \f$ B_i \f$ if up = \c false or \f$ B_i^T \f$ if up = \c true.
 * Only the lower triangular part of \f$ D_i \f$ is used to store \f$ L_i \f$.
 * Its upper part remains whatever was there originally.
 */
JRLQP_DLLAPI bool blockArrowLLT(const std::vector<MatrixRef> & diag,
                                const std::vector<MatrixRef> & side,
                                bool up = false);

/** Solve in place the system P L X = M where L is the triangular factor
 * obtained from blockArrowLLT and P is a permutation depending on \p up.
 *
 * \param diag blocks on the diagonal of L. Only the lower triangular part is
 * actually used.
 * \param side non zero blocks under the diagonal.
 * \param M right hand side of the equation (matrix or vector). Contains the
 * solution upon return.
 * \param start First row of M that is not 0. Useful for optimizing computations.
 * \param end First row of the terminal 0 block in M. If end < 0, M has no
 * terminal 0 block. Useful for optimizing computations.
 */
JRLQP_DLLAPI void blockArrowLSolve(const std::vector<MatrixRef> & diag,
                                   const std::vector<MatrixRef> & side,
                                   bool up,
                                   MatrixRef M,
                                   int start = 0,
                                   int end = -1);

/** Solve in place the system L^T P^T X = M where L is the triangular factor
 * obtained from blockArrowLLT and P is a permutation depending on \p up.
 *
 * \param diag blocks on the diagonal of L. Only the lower triangular part is
 * actually used.
 * \param side non zero blocks under the diagonal.
 * \param M right hand side of the equation (matrix or vector). Contains the
 * solution upon return.
 * \param start First row of M that is not 0. Useful for optimizing computations.
 * \param end First row of the terminal 0 block in M. If end < 0, M has no
 * terminal 0 block. Useful for optimizing computations.
 */
JRLQP_DLLAPI void blockArrowLTransposeSolve(const std::vector<MatrixRef> & diag,
                                            const std::vector<MatrixRef> & side,
                                            bool up,
                                            MatrixRef M,
                                            int start = 0,
                                            int end = -1);
} // namespace jrl::qp::decomposition
