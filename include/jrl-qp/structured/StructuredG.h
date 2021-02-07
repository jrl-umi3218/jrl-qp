/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <iosfwd>
#include <vector>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/internal/SingleNZSegmentVector.h>

namespace jrl::qp::structured
{
class JRLQP_DLLAPI StructuredG
{
public:
  enum class Type
  {
    TriBlockDiagonal,
    BlockArrowUp,
    BlockArrowDown
  };

  StructuredG() = default;
  StructuredG(Type t, const std::vector<MatrixRef> & diag, const std::vector<MatrixRef> & offDiag);

  Type type() const
  {
    return type_;
  }

  const MatrixRef & diag(int i) const
  {
    return diag_[static_cast<size_t>(i)];
  }

  const MatrixRef & offDiag(int i) const
  {
    return offDiag_[static_cast<size_t>(i)];
  }

  int nbVar() const
  {
    return nbVar_;
  }

  int nbVar(int i) const
  {
    return static_cast<int>(diag(i).cols());
  }

  bool lltInPlace();
  bool decomposed() const
  {
    return decomposed_;
  }

  void solveInPlaceLTranspose(VectorRef v) const;
  void solveL(VectorRef out, const VectorConstRef & in) const;
  void solveL(VectorRef out, const internal::SingleNZSegmentVector & in) const;

  friend std::ostream & operator<<(std::ostream & os, const StructuredG & G)
  {
    // TODO
    return os;
  }

private:
  Type type_;
  std::vector<MatrixRef> diag_;
  std::vector<MatrixRef> offDiag_;
  std::vector<int> start_;
  int nbVar_ = 0;

  bool decomposed_ = false; // Whether this contains the original matrix or its llt decomposition
};
} // namespace jrl::qp::structured