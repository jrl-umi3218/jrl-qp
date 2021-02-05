/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/structured/StructuredC.h>

namespace jrl::qp::structured
{
StructuredC::StructuredC() {}

StructuredC::StructuredC(std::vector<MatrixConstRef> C) : type_(Type::Diagonal), nbVar_(0), nbCstr_(0)
{
  for(const auto& Ci: C)
  {
    int ni = static_cast<int>(Ci.rows());
    int mi = static_cast<int>(Ci.cols());
    cumulNbVar_.push_back(nbVar_);
    cumulNbCstr_.push_back(nbCstr_);
    std::fill_n(std::back_inserter(toBlock_), mi, static_cast<int>(diag_.size()));
    nbVar_ += ni;
    nbCstr_ += mi;
    diag_.push_back(Ci);
  }
  cumulNbVar_.push_back(nbVar_);
  cumulNbCstr_.push_back(nbCstr_);
}

StructuredC & StructuredC::operator=(const StructuredC & other)
{
  type_ = other.type_;
  diag_.clear();
  std::copy(other.diag_.begin(), other.diag_.end(), std::back_inserter(diag_));
  cumulNbVar_ = other.cumulNbVar_;
  cumulNbCstr_ = other.cumulNbCstr_;
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
internal::SingleNZSegmentVector StructuredC::col(int i) const
{
  assert(i < nbCstr_);
  int bi = toBlock_[i];
  return {diag_[bi].col(i - cumulNbCstr_[bi]), cumulNbVar_[bi], nbVar_};
}

void StructuredC::transposeMult(VectorRef out, const VectorConstRef & in) const
{
  assert(type_ == Type::Diagonal);
  for(size_t i = 0; i < diag_.size(); ++i)
  {
    out.segment(cumulNbCstr_[i], diag_[i].cols()).noalias() = diag_[i].transpose() * in.segment(cumulNbVar_[i], diag_[i].rows());
  }
}
} // namespace jrl::qp::structured