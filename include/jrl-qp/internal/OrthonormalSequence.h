/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Jacobi>
#include <Eigen/Householder>

#include<jrl-qp/api.h>
#include<jrl-qp/defs.h>
#include<jrl-qp/internal/SingleNZSegmentVector.h>
#include<jrl-qp/internal/Workspace.h>

namespace jrl::qp::internal
{
enum class OSeqType
{
  Householder,
  Givens,
  Permutation,
};

/** Represents a matrix diag(I, Q, I) where Q is an orthonormal matrix
  * described as a sequence of elementary orthonormal matrices.
  */
class JRLQP_DLLAPI ElemOrthonormalSequence
{
public:
  ElemOrthonormalSequence(OSeqType type, int n, int size);

  /** Adding a Givens rotation.*/
  void add(const Givens & Q);
  /** Adding a Householder reflector.*/
  void add(const VectorConstRef & essential, double tau);

  void applyToTheLeft(VectorRef v, int start, int size) const;
  void applyTransposeToTheLeft(VectorRef v, int start, int size) const;

  OSeqType type() const
  {
    return type_;
  }

  int n() const
  {
    return n_;
  }

  int size() const
  {
    return size_;
  }

  bool full() const
  {
    return size_ >= n_ - 1;
  }

  // For debug purpose
  Eigen::MatrixXd toDense() const;

private:
  OSeqType type_;
  int n_;     //Size of the matrix represented by the sequence
  int size_;  //Number of elementary transformation
  Workspace<> work1_;
  Workspace<> work2_;
};

class JRLQP_DLLAPI OrthonormalSequence
{
public:
  OrthonormalSequence(int n = 0);

  /** Adding a Givens rotation.*/
  void add(int start, const Givens & Q);
  /** Adding a Householder reflector.*/
  void add(int start, const VectorConstRef & essential, double tau);

  void prepare(OSeqType type, int n, int seqSize);

  void clear();
  void resize(int n);

  void applyToTheLeft(VectorRef v) const;
  void applyTransposeToTheLeft(VectorRef v) const;

  void applyToTheLeft(VectorRef v, const SingleNZSegmentVector & in) const;
  void applyTransposeToTheLeft(VectorRef out, const SingleNZSegmentVector & in) const;

private:
  struct EmbeddedSeq
  {
    EmbeddedSeq(int start, OSeqType type, int n, int size) : start(start), H(type, n, size) {}
    int start;
    ElemOrthonormalSequence H;
  };

  int n_; // Size of the matrix represented by the sequence.
  std::vector<EmbeddedSeq> seq_;
};
}