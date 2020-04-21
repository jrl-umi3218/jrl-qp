/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

namespace jrlqp
{
  //definitions
  using MatrixConstRef = Eigen::Ref<const Eigen::MatrixXd>;
  using MatrixRef = Eigen::Ref<Eigen::MatrixXd>;
  using VectorConstRef = Eigen::Ref<const Eigen::VectorXd>;
  using VectorRef = Eigen::Ref<Eigen::VectorXd>;

  namespace constant
  {
    inline constexpr std::uint32_t noIterationFlag = 1 << 31;
  }
}
