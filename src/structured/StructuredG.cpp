#include <jrl-qp/structured/StructuredG.h>

jrl::qp::structured::StructuredG::StructuredG(Type t,
                                              const std::vector<MatrixRef> & diag,
                                              const std::vector<MatrixRef> & offDiag)
: type_(t), nbVar_(0)
{
  for(const auto& D : diag)
  {
    assert(D.cols() == D.rows());
    diag_.push_back(D);
    start_.push_back(nbVar_);
    nbVar_ += static_cast<int>(D.cols());
  }
  start_.push_back(nbVar_);
  std::copy(offDiag.begin(), offDiag.end(), std::back_inserter(offDiag_));
}

bool jrl::qp::structured::StructuredG::lltInPlace()
{
  return false;
}

void jrl::qp::structured::StructuredG::solveInPlaceLTranspose(VectorRef v) const {}

void jrl::qp::structured::StructuredG::solveL(VectorRef out, const internal::SingleNZSegmentVector & in) const {}
