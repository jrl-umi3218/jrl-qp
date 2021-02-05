/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

namespace jrl::qp::decomposition
{
JRLQP_DLLAPI bool blockArrowLLT(const std::vector<MatrixRef> & diag,
                                const std::vector<MatrixRef> & side,
                                bool up = false);
JRLQP_DLLAPI void blockArrowLSolve(const std::vector<MatrixRef> & diag,
                                   const std::vector<MatrixRef> & side,
                                   bool up,
                                   MatrixRef v,
                                   int start = 0);
JRLQP_DLLAPI void blockArrowLTransposeSolve(const std::vector<MatrixRef> & diag,
                                            const std::vector<MatrixRef> & side,
                                            bool up,
                                            MatrixRef v,
                                            int end = -1);
} // namespace jrl::qp::decomposition
