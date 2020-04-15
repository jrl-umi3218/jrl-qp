/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/enums.h>

namespace jrlqp::internal
{
  class ConstraintNormal
  {
  public:
    ConstraintNormal() : p_(-1), status_(ActivationStatus::INACTIVE), C_(Eigen::MatrixXd(0, 0)) {}
    ConstraintNormal(const MatrixConstRef& C, int p, ActivationStatus status)
      : p_(p), status_(status), C_(C) 
    {
      assert(status != ActivationStatus::INACTIVE);
    }

    ConstraintNormal(const ConstraintNormal& other)
      : p_(other.p_), status_(other.status_), C_(other.C_) {}
    ConstraintNormal(ConstraintNormal&& other) noexcept
      : p_(std::move(other.p_)), status_(std::move(other.status_)), C_(std::move(other.C_)) {}
    ConstraintNormal& operator=(const ConstraintNormal& other)
    {
      p_ = other.p_;
      status_ = other.status_;
      new(&C_) MatrixConstRef(other.C_);
      return *this;
    }

    int index() const { return p_; }
    int bndIndex() const { assert(status_ >= ActivationStatus::LOWER_BOUND); return p_ - static_cast<int>(C_.rows()); }
    ActivationStatus status() const { return status_; }


    void preMultiplyByMt(Eigen::VectorXd& out, const Eigen::MatrixXd& M) const
    {
      switch (status_)
      {
      case ActivationStatus::EQUALITY: //fallthrough
      case ActivationStatus::LOWER:
        out.noalias() = M.transpose() * C_.col(p_);
        break;
      case ActivationStatus::UPPER:
        out.noalias() = -M.transpose() * C_.col(p_);
        break;
      case ActivationStatus::LOWER_BOUND:
        out = M.row(p_ - C_.rows());
        break;
      case ActivationStatus::UPPER_BOUND:
        out = -M.row(p_ - C_.rows());
        break;
      default:
        assert(false);
      }
    }

    double dot(const Eigen::VectorXd& v) const
    {
      switch (status_)
      {
      case ActivationStatus::EQUALITY: //fallthrough
      case ActivationStatus::LOWER: return v.dot(C_.col(p_));
      case ActivationStatus::UPPER: return -v.dot(C_.col(p_));
      case ActivationStatus::LOWER_BOUND: return v[bndIndex()];
      case ActivationStatus::UPPER_BOUND: return -v[bndIndex()];
      default: assert(false); return 0;
      }
    }

  private:
    int p_;
    ActivationStatus status_;
    MatrixConstRef C_;
  };
}