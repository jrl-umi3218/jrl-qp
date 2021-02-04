/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>


namespace jrl::qp::decomposition
{
JRLQP_DLLAPI bool triBlockDiagLLT(const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & subDiag);

}