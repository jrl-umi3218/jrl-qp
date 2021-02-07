/* Copyright 2020-2021 CNRS-AIST JRL */

#include <jrl-qp/internal/OrthonormalSequence.h>

namespace jrl::qp::internal
{
ElemOrthonormalSequence::ElemOrthonormalSequence(OSeqType type, int n, int size) : type_(type), n_(n), size_(0)
{
  switch(type)
  {
    case jrl::qp::internal::OSeqType::Householder:
      work1_.resize(n, size);
      work2_.resize(size);
      break;
    case jrl::qp::internal::OSeqType::Givens:
      work1_.resize(size);
      work2_.resize(size);
      break;
    case jrl::qp::internal::OSeqType::Permutation:
      assert(false && "Not implemented yet");
      break;
    default:
      break;
  }
}

void ElemOrthonormalSequence::add(const Givens & Q)
{
  assert(type_ == OSeqType::Givens);
  auto c = work1_.asVector(size_ + 1);
  auto s = work2_.asVector(size_ + 1);
  c[size_] = Q.c();
  s[size_] = Q.s();
  ++size_;
}

void ElemOrthonormalSequence::add(const VectorConstRef & essential, double tau)
{
  assert(type_ == OSeqType::Householder);
  assert(essential.size() == n_ - size_ - 1);
  auto e = work1_.asMatrix(n_, size_ + 1, n_);
  auto t = work2_.asVector(size_ + 1);
  e.col(size_).tail(n_ - size_ - 1) = essential;
  e(size_, size_) = 1;
  t[size_] = tau;
  ++size_;
}

void ElemOrthonormalSequence::applyToTheLeft(VectorRef v, int start, int size) const
{
  assert(v.size() == n_);
  assert(start >= 0 && size >= 0 && start + size <= n_);
  switch(type_)
  {
    case OSeqType::Householder:
    {
      auto E = work1_.asMatrix(n_, size_, n_);
      auto h = work2_.asVector(size_);
      if(size_ == 1)
      {
        double d = E.col(0).segment(start, size).dot(v.segment(start, size));
        v -= h[0] * d * E.col(0);
      }
      else
      {
        // [OPTIM] ? take start and size into account
        Eigen::HouseholderSequence H(E, h);
        H.applyThisOnTheLeft(v);
      }
    }
    break;
    case OSeqType::Givens:
    {
      auto c = work1_.asVector(size_);
      auto s = work2_.asVector(size_);
      int b = std::min(size_ - 1, start + size - 1);
      for(int i = b; i >= 0; --i) v.applyOnTheLeft(i, i + 1, Givens(c[i], s[i]));
    }
    break;
    case OSeqType::Permutation:
      assert(false && "Not implemented yet");
      break;
    default:
      assert(false);
  }
}

void ElemOrthonormalSequence::applyTransposeToTheLeft(VectorRef v, int start, int size) const
{
  assert(v.size() == n_);
  assert(start >= 0 && size >= 0 && start + size <= n_);
  switch(type_)
  {
    case OSeqType::Householder:
    {
      auto E = work1_.asMatrix(n_, size_, n_);
      auto h = work2_.asVector(size_);
      if(size_ == 1)
      {
        double d = E.col(0).segment(start, size).dot(v.segment(start, size));
        v -= h[0] * d * E.col(0);
      }
      else
      {
        // [OPTIM] ? take start and size into account
        Eigen::HouseholderSequence H(E, h);
        H.transpose().applyThisOnTheLeft(v);
      }
    }
    break;
    case OSeqType::Givens:
    {
      auto c = work1_.asVector(size_);
      auto s = work2_.asVector(size_);
      int b = std::max(0, start - 1);
      for(int i = b; i < size_; ++i) v.applyOnTheLeft(i, i + 1, Givens(c[i], s[i]).transpose());
    }
    break;
    case OSeqType::Permutation:
      assert(false && "Not implemented yet");
      break;
    default:
      assert(false);
  }
}
Eigen::MatrixXd ElemOrthonormalSequence::toDense() const
{
  Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(n_, n_);
  for(int i = 0; i < n_; ++i)
  {
    applyToTheLeft(ret.col(i), 0, n_);
  }
  return ret;
}

OrthonormalSequence::OrthonormalSequence(int n) : n_(n) {}

void OrthonormalSequence::add(int start, const Givens & Q)
{
  auto & last = seq_.back();
  assert(last.H.type() == OSeqType::Givens);
  if(last.H.size() == 0)
  {
    last.start = start;
  }
  else
  {
    assert(last.start + last.H.size() == start);
  }
  last.H.add(Q);
}

void OrthonormalSequence::add(int start, const VectorConstRef & essential, double tau)
{
  auto & last = seq_.back();
  assert(last.H.type() == OSeqType::Householder);
  if(last.H.size() == 0)
  {
    last.start = start;
  }
  else
  {
    assert(last.start + last.H.size() == start);
  }
  last.H.add(essential, tau);
}

void OrthonormalSequence::prepare(OSeqType type, int n, int seqSize)
{
  seq_.emplace_back(0, type, n, seqSize);
}

void OrthonormalSequence::clear()
{
  seq_.clear();
}

void OrthonormalSequence::resize(int n)
{
  n_ = n;
}

void OrthonormalSequence::applyToTheLeft(VectorRef v) const
{
  assert(v.size() == n_);
  for(int i = static_cast<int>(seq_.size()) - 1; i >= 0; --i)
  {
    const auto & Hi = seq_[i];
    Hi.H.applyToTheLeft(v.segment(Hi.start, Hi.H.n()), 0, Hi.H.n());
  }
}

void OrthonormalSequence::applyTransposeToTheLeft(VectorRef v) const
{
  assert(v.size() == n_);
  for(size_t i = 0; i < seq_.size(); ++i)
  {
    const auto & Hi = seq_[i];
    Hi.H.applyTransposeToTheLeft(v.segment(Hi.start, Hi.H.n()), 0, Hi.H.n());
  }
}

void OrthonormalSequence::applyToTheLeft(VectorRef out, const SingleNZSegmentVector & in) const
{
  assert(in.size() == n_);
  assert(out.size() == n_);
  in.toFullVector(out);
  int start = in.start(); // first non zero
  int end = in.start() + static_cast<int>(in.nzSegment().size()); // first zero after segment
  for(int i = static_cast<int>(seq_.size()) - 1; i >= 0; --i)
  {
    const auto & Hi = seq_[i];
    if(Hi.start >= end || Hi.start + Hi.H.n() < start) continue; // We skip if Hi would multiply zero
    int s = std::max(0, start - Hi.start);
    int n = std::min(Hi.H.n(), end - Hi.start);
    Hi.H.applyToTheLeft(out.segment(Hi.start, Hi.H.n()), s, n - s);
    start = std::min(start, Hi.start);
    end = std::max(end, Hi.start + Hi.H.n());
  }
}

void OrthonormalSequence::applyTransposeToTheLeft(VectorRef out, const SingleNZSegmentVector & in) const
{
  assert(in.size() == n_);
  assert(out.size() == n_);
  in.toFullVector(out);
  int start = in.start(); // first non zero
  int end = in.start() + static_cast<int>(in.nzSegment().size()); // first zero after segment
  for(size_t i = 0; i < seq_.size(); ++i)
  {
    const auto & Hi = seq_[i];
    if(Hi.start >= end || Hi.start + Hi.H.n() < start) continue; // We skip if Hi would multiply zero
    int s = std::max(0, start - Hi.start);
    int n = std::min(Hi.H.n(), end - Hi.start);
    Hi.H.applyTransposeToTheLeft(out.segment(Hi.start, Hi.H.n()), s, n - s);
    start = std::min(start, Hi.start);
    end = std::max(end, Hi.start + Hi.H.n());
  }
}
} // namespace jrl::qp::internal