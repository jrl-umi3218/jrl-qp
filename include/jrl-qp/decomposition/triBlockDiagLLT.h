/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

namespace jrl::qp::decomposition
{
JRLQP_DLLAPI bool triBlockDiagLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & subDiag);

JRLQP_DLLAPI void triBlockDiagLSolve(const std::vector<MatrixRef> & diag,
                                     const std::vector<MatrixRef> & subDiag,
                                     MatrixRef v,
                                     int start = 0);

JRLQP_DLLAPI void triBlockDiagLTransposeSolve(const std::vector<MatrixRef> & diag,
                                              const std::vector<MatrixRef> & subDiag,
                                              MatrixRef v,
                                              int end = -1);
} // namespace jrl::qp::decomposition