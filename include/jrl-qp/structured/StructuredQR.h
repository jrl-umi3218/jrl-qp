/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

#include <jrl-qp/internal/OrthonormalSequence.h>
#include <jrl-qp/internal/SelectedConstraint.h>
#include <jrl-qp/internal/Workspace.h>
#include <jrl-qp/structured/StructuredC.h>
#include <jrl-qp/structured/StructuredG.h>

namespace jrl::qp::structured
{
class JRLQP_DLLAPI StructuredQR
{
public:
  StructuredQR();

  void setL(const StructuredG & decomposedG);

  void reset();
  void resize(int nbVar);

  internal::PartitionnedQ getPartitionnedQ() const;

  void setRToZero();
  void RSolve(VectorRef out, const VectorConstRef & in) const;

  bool add(const VectorConstRef & d);
  bool remove(int l);

private:
  void adjustLdR(int q) const;
  auto getR(int q);
  auto getR(int q) const;
  auto getUpperTriangularR(int q) const;

  int q_ = 0; // size of R (that is the number of active constraints)
  int nbVar_ = 0;
  mutable int ldR_ = 1; // Leading dimension used for R
  mutable internal::Workspace<> work_R_;
  internal::Workspace<> work_essential_;
  internal::OrthonormalSequence Q_;
  mutable internal::Workspace<> work_tmp_;
};
} // namespace jrl::qp::structured