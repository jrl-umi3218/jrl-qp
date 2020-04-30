/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/enums.h>
#include <jrl-qp/utils/toMatlab.h>

namespace jrlqp::internal
{
  class ConstraintNormal
  {
  public:
    ConstraintNormal() : p_(-1), status_(ActivationStatus::INACTIVE), C_(Eigen::MatrixXd(0, 0)) {}
    ConstraintNormal(const MatrixConstRef& C, int p, ActivationStatus status)
      : p_(p), status_(status), C_(C) 
    {
      assert((status < ActivationStatus::LOWER_BOUND&& p < C_.cols())
        || (status >= ActivationStatus::LOWER_BOUND && p >= C.cols()));
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
    int bndIndex() const 
    { assert(status_ >= ActivationStatus::LOWER_BOUND); return p_ - static_cast<int>(C_.cols()); }
    ActivationStatus status() const { return status_; }


    void preMultiplyByMt(VectorRef out, const MatrixConstRef& M) const
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
      case ActivationStatus::FIXED: //fallthrough
      case ActivationStatus::LOWER_BOUND:
        out = M.row(p_ - C_.cols());
        break;
      case ActivationStatus::UPPER_BOUND:
        out = -M.row(p_ - C_.cols());
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
      case ActivationStatus::FIXED: //fallthrough
      case ActivationStatus::LOWER_BOUND: return v[bndIndex()];
      case ActivationStatus::UPPER_BOUND: return -v[bndIndex()];
      default: assert(false); return 0;
      }
    }

    friend std::ostream& operator<<(std::ostream& os, const ConstraintNormal& n)
    {
      os << "{" << n.p_ << ", " << static_cast<int>(n.status_) << ", ";
      switch (n.status_)
      {
      case ActivationStatus::EQUALITY: //fallthrough
      case ActivationStatus::LOWER: os << (utils::toMatlab)n.C_.col(n.p_); break;
      case ActivationStatus::UPPER: os << (utils::toMatlab) (-n.C_.col(n.p_)); break;
      case ActivationStatus::FIXED: //fallthrough
      case ActivationStatus::LOWER_BOUND: os << (utils::toMatlab)Eigen::MatrixXd::Identity(n.C_.rows(), n.C_.rows()).col(n.bndIndex()); break;
      case ActivationStatus::UPPER_BOUND: os << (utils::toMatlab)(-Eigen::MatrixXd::Identity(n.C_.rows(),n.C_.rows()).col(n.bndIndex())); break;
      default: assert(false);
      }
      os << "}";
      return os;
    }

  private:
    int p_;
    ActivationStatus status_;
    MatrixConstRef C_;
  };

}