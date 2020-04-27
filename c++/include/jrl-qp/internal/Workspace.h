/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>

namespace jrlqp::internal
{
  struct NotConst {};

  template<typename Scalar = double>
  class JRLQP_DLLAPI Workspace
  {
  public:
    Workspace() : buffer_(0) {}
    Workspace(int size) : buffer_(size) {}
    Workspace(int rows, int cols) : Workspace(rows* cols) {}

    void resize(int size, bool fit = false)
    {
      if (size > buffer_.size() || (size<buffer_.size() && fit))
      {
        buffer_.resize(size);
      }
    }

    int size() const { return static_cast<int>(buffer_.size()); }

    void resize(int rows, int cols, bool fit = false)
    {
      resize(rows * cols, fit);
    }

    auto asVector(int size, NotConst = {})
    {
      assert(size <= buffer_.size());
      return buffer_.head(size);
    }

    auto asVector(int size) const
    {
      assert(size <= buffer_.size());
      return buffer_.head(size);
    }

    auto asMatrix(int rows, int cols, int ld, NotConst = {})
    {
      assert(ld * cols <= buffer_.size());
      assert(ld >= rows);
      return Eigen::Map<Eigen::MatrixXd,0, Eigen::Stride<Eigen::Dynamic, 1> >(buffer_.data(), rows, cols, Eigen::Stride<Eigen::Dynamic,1>(ld,1));
    }

    auto asMatrix(int rows, int cols, int ld) const
    {
      assert(ld * cols <= buffer_.size());
      assert(ld >= rows);
      return Eigen::Map<const Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1> >(buffer_.data(), rows, cols, Eigen::Stride<Eigen::Dynamic, 1>(ld, 1));
    }

    void setZero()
    {
      buffer_.setZero();
    }

  private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> buffer_;
  };
}

namespace jrlqp
{
  using WVector = decltype(internal::Workspace<double>().asVector(0));
  using WConstVector = decltype(std::add_const_t<internal::Workspace<double>>().asVector(0));
}