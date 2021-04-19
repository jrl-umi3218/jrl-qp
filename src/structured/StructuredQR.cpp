/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/structured/StructuredQR.h>
#include <jrl-qp/utils/debug.h>

namespace jrl::qp::structured
{
StructuredQR::StructuredQR() {}

void StructuredQR::adjustLdR(int q) const
{
  assert(q <= nbVar_);
  if(q > ldR_)
  {
    int newLdR = std::min(2 * ldR_, nbVar_);
    work_R_.changeLd(q_, q_, ldR_, newLdR);
    ldR_ = newLdR;
  }
}

auto StructuredQR::getR(int q)
{
  adjustLdR(q);
  return work_R_.asMatrix(q, q, ldR_, {});
}

auto StructuredQR::getR(int q) const
{
  adjustLdR(q);
  return work_R_.asMatrix(q, q, ldR_);
}

auto StructuredQR::getUpperTriangularR(int q) const
{
  return getR(q).template triangularView<Eigen::Upper>();
}

void StructuredQR::reset()
{
  q_ = 0;
  Q_.clear();
}

void StructuredQR::resize(int nbVar)
{
  Q_.resize(nbVar);
  work_R_.resize(nbVar, nbVar);
  work_tmp_.resize(nbVar);
  work_essential_.resize(nbVar);
  nbVar_ = nbVar;
  ldR_ = std::min(10, static_cast<int>(std::sqrt(nbVar))); // TODO Ability to change this heuristics.
}

internal::PartitionnedQ StructuredQR::getPartitionnedQ() const
{
  return {Q_, q_};
}

void StructuredQR::setRToZero()
{
  work_R_.setZero();
}

void StructuredQR::RSolve(VectorRef out, const VectorConstRef & in) const
{
  assert(in.size() == q_);
  assert(out.size() == q_);
  out = getUpperTriangularR(q_).solve(in);
}
bool StructuredQR::add(const VectorConstRef & d)
{
  assert(d.size() == nbVar_);
  double beta, tau;
  WVector e = work_essential_.asVector(nbVar_ - q_ - 1);
  d.tail(nbVar_ - q_).makeHouseholder(e, tau, beta);
  auto R = getR(q_ + 1);
  R.rightCols<1>().head(q_) = d.head(q_);
  R(q_, q_) = beta;
  Q_.prepare(internal::OSeqType::Householder, nbVar_ - q_, 1);
  Q_.add(q_, e, tau);
  ++q_;

  return true; //[NUMERIC]: add test on dependency
}

bool StructuredQR::remove(int l)
{
  --q_;
  auto R = getR(q_ + 1);

  Q_.prepare(internal::OSeqType::Givens, q_ - l + 1, q_ - l);
  for(int i = l; i < q_; ++i)
  {
    Givens Qi;
    R.col(i).head(i) = R.col(i + 1).head(i);
    Qi.makeGivens(R(i, i + 1), R(i + 1, i + 1), &R(i, i));
    JRLQP_DEBUG_ONLY(R(i + 1, i + 1) = 0);
    R.rightCols(q_ - i - 1).applyOnTheLeft(i, i + 1, Qi.transpose());
    Q_.add(i, Qi);
  }
  return false;
}

} // namespace jrl::qp::structured
