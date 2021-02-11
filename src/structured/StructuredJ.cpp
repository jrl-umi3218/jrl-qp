/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/structured/StructuredJ.h>
#include <jrl-qp/utils/debug.h>

namespace jrl::qp::structured
{
StructuredJ::StructuredJ() {}

void StructuredJ::setL(const StructuredG & decomposedG)
{
  assert(decomposedG.decomposed());
  L_ = &decomposedG;
}

void StructuredJ::setQ(const internal::PartitionnedQ & Q)
{
  Q_ = Q;
}

void StructuredJ::reset()
{
  L_ = nullptr;
  Q_.reset();
}

void StructuredJ::resize(int nbVar)
{
  nbVar_ = nbVar;
}

void StructuredJ::premultByJ2(VectorRef out, const VectorConstRef & in) const
{
  assert(out.size() == nbVar_);
  assert(in.size() == Q_.m2());
  // [OPTIM] The 3 next line are emulating out = Q2 * in by performing out = Q * [0;in]. This can be optimized.
  out.tail(Q_.m2()) = in;
  out.head(Q_.m1()).setZero();
  Q_.Q().applyToTheLeft(out);
  L_->solveInPlaceLTranspose(out);
}

void StructuredJ::premultByJt(VectorRef out, const StructuredC & C, const internal::SelectedConstraint & sc) const
{
  if(sc.status() <= ActivationStatus::EQUALITY)
  {
    L_->solveL(out, C.col(sc.index()));
    if(sc.status() == ActivationStatus::UPPER) out *= -1;
  }
  else
  {
    Eigen::Matrix<double, 1, 1> e;
    e[0] = sc.status() == ActivationStatus::UPPER_BOUND ? -1 : 1;
    L_->solveL(out, internal::SingleNZSegmentVector(e, sc.index() - C.nbCstr(), nbVar_));
  }
  Q_.Q().applyTransposeToTheLeft(out);
}

} // namespace jrl::qp::structured
