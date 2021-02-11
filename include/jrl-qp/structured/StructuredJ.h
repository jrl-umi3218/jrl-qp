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
class JRLQP_DLLAPI StructuredJ
{
public:
  StructuredJ();

  void setL(const StructuredG & decomposedG);
  void setQ(const internal::PartitionnedQ & Q);

  void reset();
  void resize(int nbVar);

  void premultByJ2(VectorRef out, const VectorConstRef & in) const;
  void premultByJt(VectorRef out, const StructuredC & C, const internal::SelectedConstraint & sc) const;

private:
  int nbVar_ = 0;
  const StructuredG * L_ = nullptr;
  internal::PartitionnedQ Q_;
};
} // namespace jrl::qp::structured