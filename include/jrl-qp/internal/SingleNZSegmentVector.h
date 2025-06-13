/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/defs.h>

namespace jrl::qp::internal
{
/** A vector [0; v; 0]*/
class SingleNZSegmentVector
{
public:
  /** \param v Non-zero segment of the vector
   *  \param start Size of the zero before v
   *  \param size Size of this vector
   */
  SingleNZSegmentVector(const VectorConstRef & v, int start, int size) : start_(start), size_(size), v_(v)
  {
    assert(start >= 0);
    assert(start + v.size() <= size);
  }

  int start() const
  {
    return start_;
  }

  int end() const
  {
    return start_ + static_cast<int>(v_.size());
  }

  int size() const
  {
    return size_;
  }

  const VectorConstRef & nzSegment() const
  {
    return v_;
  }

  void toFullVector(VectorRef u) const
  {
    assert(u.size() == size_);
    u.head(start_).setZero();
    u.segment(start_, v_.size()) = v_;
    u.tail(size_ - v_.size() - start_).setZero();
  }

  double dot(const VectorConstRef & u) const
  {
    assert(u.size() == size_);
    return v_.dot(u.segment(start_, v_.size()));
  }

private:
  int start_;
  int size_;
  VectorConstRef v_;
};
} // namespace jrl::qp::internal
