/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <iosfwd>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>

#include <jrl-qp/internal/SingleNZSegmentVector.h>

namespace jrl::qp::structured
{
class StructuredC
{
public:
  enum class Type
  {
    Diagonal,
  };

  StructuredC();

  StructuredC & operator=(const StructuredC & other);

  const MatrixConstRef & diag(int i) const;
  //const MatrixConstRef & offDiag(int i) const;
  int nbVar() const;
  int nbVar(int i) const;
  int nbCstr() const;
  int nbCstr(int i) const;
  const internal::SingleNZSegmentVector & col(int i) const;

  friend std::ostream & operator<<(std::ostream & os, const StructuredC & C)
  { 
    // TODO
    return os;
  }

private:
  Type type_;
  std::vector<MatrixConstRef> diag_;
  //std::vector<MatrixConstRef> offDiag_;
  std::vector<int> start_;
  std::vector<int> toBlock_;
  int nbVar_;
  int nbCstr_;
};
}
