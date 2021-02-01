/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <iosfwd>

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/enums.h>
#include <jrl-qp/internal/SelectedConstraint.h>
#include <jrl-qp/utils/toMatlab.h>

namespace jrl::qp::internal
{
/** A class to represent the vector \p n of a constraint \f$ n^T x op b \f$.
 *
 * It is meant to abstract the difference between general constraints, where
 * \p n is a column of a dense matrix \p C, and bounds, where \p n is a column
 * of the identity matrix, to provide optimized computations.
 */
class ConstraintNormal
{
public:
  /** Default constructor.*/
  ConstraintNormal() : p_(-1), status_(ActivationStatus::INACTIVE), C_(Eigen::MatrixXd(0, 0)) {}
  /** Usual constructor
   *
   * \param C Matrix from which the column is taken. Useful only if
   * \p p < C.cols(). Note that the object only keep a reference on the
   * matrix, that must remain valid for the lifetime of the object.
   * \param p If smaller than C.cols(), refers to a column of C (general
   * constraints). If not, refer to the bound with index p-C.Cols().
   * \param status Activation status of the associated constraint. Must be
   * consistent with the index \p p.
   */
  ConstraintNormal(const MatrixConstRef & C, int p, ActivationStatus status) : p_(p), status_(status), C_(C)
  {
    assert((status < ActivationStatus::LOWER_BOUND && p < C_.cols())
           || (status >= ActivationStatus::LOWER_BOUND && p >= C.cols()));
  }

  /** Same as the usual construtor, but with (p, status) given by a SelectedConstraint class.*/
  ConstraintNormal(const MatrixConstRef & C, const SelectedConstraint& sc) : ConstraintNormal(C, sc.index(), sc.status()) {}

  ConstraintNormal(const ConstraintNormal & other) : p_(other.p_), status_(other.status_), C_(other.C_) {}
  ConstraintNormal(ConstraintNormal && other) noexcept
  : p_(std::move(other.p_)), status_(std::move(other.status_)), C_(std::move(other.C_))
  {
  }
  ConstraintNormal & operator=(const ConstraintNormal & other)
  {
    p_ = other.p_;
    status_ = other.status_;
    new(&C_) MatrixConstRef(other.C_);
    return *this;
  }

  /** Underlying index of the constraint.*/
  int index() const
  {
    return p_;
  }
  /** Underlying index of the constraint seen as a bound constraint, i.e. index() - C.cols().*/
  int bndIndex() const
  {
    assert(status_ >= ActivationStatus::LOWER_BOUND);
    return p_ - static_cast<int>(C_.cols());
  }
  /** Activation status of the corresponding constraint.*/
  ActivationStatus status() const
  {
    return status_;
  }

  /** Performs \f$ out = M^T * n \f$. */
  void preMultiplyByMt(VectorRef out, const MatrixConstRef & M) const
  {
    switch(status_)
    {
      case ActivationStatus::EQUALITY: // fallthrough
      case ActivationStatus::LOWER:
        out.noalias() = M.transpose() * C_.col(p_);
        break;
      case ActivationStatus::UPPER:
        out.noalias() = -M.transpose() * C_.col(p_);
        break;
      case ActivationStatus::FIXED: // fallthrough
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

  /** Performs \f$ n^T v \f$. */
  double dot(const VectorConstRef & v) const
  {
    switch(status_)
    {
      case ActivationStatus::EQUALITY: // fallthrough
      case ActivationStatus::LOWER:
        return v.dot(C_.col(p_));
      case ActivationStatus::UPPER:
        return -v.dot(C_.col(p_));
      case ActivationStatus::FIXED: // fallthrough
      case ActivationStatus::LOWER_BOUND:
        return v[bndIndex()];
      case ActivationStatus::UPPER_BOUND:
        return -v[bndIndex()];
      default:
        assert(false);
        return 0;
    }
  }

  friend std::ostream & operator<<(std::ostream & os, const ConstraintNormal & n)
  {
    os << "{" << n.p_ << ", " << static_cast<int>(n.status_) << ", ";
    switch(n.status_)
    {
      case ActivationStatus::EQUALITY: // fallthrough
      case ActivationStatus::LOWER:
        os << (utils::toMatlab)n.C_.col(n.p_);
        break;
      case ActivationStatus::UPPER:
        os << (utils::toMatlab)(-n.C_.col(n.p_));
        break;
      case ActivationStatus::FIXED: // fallthrough
      case ActivationStatus::LOWER_BOUND:
        os << (utils::toMatlab)Eigen::MatrixXd::Identity(n.C_.rows(), n.C_.rows()).col(n.bndIndex());
        break;
      case ActivationStatus::UPPER_BOUND:
        os << (utils::toMatlab)(-Eigen::MatrixXd::Identity(n.C_.rows(), n.C_.rows()).col(n.bndIndex()));
        break;
      default:
        assert(false);
    }
    os << "}";
    return os;
  }

private:
  int p_;
  ActivationStatus status_;
  MatrixConstRef C_;
};

} // namespace jrl::qp::internal