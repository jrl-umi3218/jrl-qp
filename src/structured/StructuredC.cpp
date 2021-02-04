/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/structured/StructuredC.h>

namespace jrl::qp::structured
{
StructuredC::StructuredC() {}
StructuredC & StructuredC::operator=(const StructuredC & other)
{
  type_ = other.type_;
  diag_.clear();
  std::copy(other.diag_.begin(), other.diag_.end(), std::back_inserter(diag_));
  start_ = other.start_;
  toBlock_ = other.toBlock_;
  nbVar_ = other.nbVar_;
  nbCstr_ = other.nbCstr_;
  return *this;
}

const MatrixConstRef & StructuredC::diag(int i) const
{
  return diag_[i];
}
int StructuredC::nbVar() const
{
  return nbVar_;
}
int StructuredC::nbVar(int i) const
{
  return static_cast<int>(diag_[i].rows());
}
int StructuredC::nbCstr() const
{
  return nbCstr_;
}
int StructuredC::nbCstr(int i) const
{
  return static_cast<int>(diag_[i].cols());
}
const internal::SingleNZSegmentVector & StructuredC::col(int i) const
{
  assert(i < nbCstr_);
  int bi = toBlock_[i];
  return {diag_[bi].col(i - start_[bi]), start_[bi], nbVar(bi)};
}
} // namespace jrl::qp::structured