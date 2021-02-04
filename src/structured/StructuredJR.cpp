/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/structured/StructuredJR.h>
#include <jrl-qp/utils/debug.h>

namespace jrl::qp::structured
{
StructuredJR::StructuredJR() {}

void StructuredJR::adjustLdR(int q) const
{
  assert(q <= nbVar_);
  if(q > ldR_)
  {
    int newLdR = std::min(2 * ldR_, nbVar_);
    work_R_.changeLd(q_, q_, ldR_, newLdR);
    ldR_ = newLdR;
  }
}

auto StructuredJR::getR(int q)
{
  adjustLdR(q);
  return work_R_.asMatrix(q, q, ldR_, {});
}

auto StructuredJR::getR(int q) const
{
  adjustLdR(q);
  return work_R_.asMatrix(q, q, ldR_);
}

auto StructuredJR::getUpperTriangularR(int q) const
{
  return getR(q).template triangularView<Eigen::Upper>();
}

void StructuredJR::setL(const StructuredG & decomposedG)
{
  assert(decomposedG.decomposed());
  L_ = &decomposedG;
}

void StructuredJR::reset()
{
  q_ = 0;
  Q_.clear();
  L_ = nullptr;
}

void StructuredJR::resize(int nbVar)
{
  work_R_.resize(nbVar, nbVar);
  work_tmp_.resize(nbVar);
  work_essential_.resize(nbVar);
}

void StructuredJR::premultByJ2(VectorRef out, const VectorConstRef & in) const
{
  assert(out.size() == nbVar_);
  assert(in.size() == nbVar_ - q_);
  // [OPTIM] The 3 next line are emulating out = Q2 * in by performing out = Q * [0;in]. This can be optimized.
  out.tail(nbVar_ - q_) = in;
  out.head(q_).setZero();
  Q_.applyToTheLeft(out);
  L_->solveInPlaceLTranspose(out);
}

void StructuredJR::premultByJt(VectorRef out, const StructuredC& C, int i) const
{
  L_->solveL(out, C.col(i));
  Q_.applyTransposeToTheLeft(out);
}

void StructuredJR::setRToZero()
{
  work_R_.setZero();
}

void StructuredJR::RSolve(VectorRef out, const VectorConstRef & in) const
{
  assert(in.size() == q_);
  assert(out.size() == q_);
  out = getUpperTriangularR(q_).solve(in);
}
bool StructuredJR::add(const VectorConstRef & d)
{
  assert(d.size() == nbVar_);
  double beta, tau;
  WVector e = work_essential_.asVector(nbVar_ - q_ - 1);
  d.tail(nbVar_ - q_).makeHouseholder(e, tau, beta);
  auto R = getR(q_+1);
  R.rightCols<1>().head(q_) = d.head(q_);
  R(q_, q_) = beta;
  Q_.add(q_, e, tau);
  ++q_;

  return true; //[NUMERIC]: add test on dependency
}

bool StructuredJR::remove(int l)
{
  --q_;
  auto R = getR(q_ + 1);

  for(int i = l; i < q_; ++i)
  {
    Givens Qi;
    R.col(i).head(i) = R.col(i + 1).head(i);
    Qi.makeGivens(R(i, i + 1), R(i + 1, i + 1), &R(i, i));
    DEBUG_ONLY(R(i + 1, i + 1) = 0);
    R.rightCols(q_ - i - 1).applyOnTheLeft(i, i + 1, Qi.transpose());
    Q_.add(i, Qi);
  }
  return false;
}

} // namespace jrl::qp::structured
