#include <jrl-qp/structured/StructuredG.h>

#include <jrl-qp/decomposition/blockArrowLLT.h>
#include <jrl-qp/decomposition/triBlockDiagLLT.h>

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
  bool done;
  switch(type_)
  {
    case Type::TriBlockDiagonal:
      done = decomposition::triBlockDiagLLT(diag_, offDiag_);
      break;
    case Type::BlockArrowUp:
      done = decomposition::blockArrowLLT(diag_, offDiag_, true);
      break;
    case Type::BlockArrowDown:
      done = decomposition::blockArrowLLT(diag_, offDiag_, false);
      break;
    default:
      assert(false);
      done = false;
      break;
  }
  decomposed_ = done;
  return done;
}

void jrl::qp::structured::StructuredG::solveInPlaceLTranspose(VectorRef v) const
{
  assert(decomposed_);
  assert(v.size() == nbVar_);

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      decomposition::triBlockDiagLTransposeSolve(diag_, offDiag_, v);
      break;
    case Type::BlockArrowUp:
      decomposition::blockArrowLTransposeSolve(diag_, offDiag_, true, v);
      break;
    case Type::BlockArrowDown:
      decomposition::blockArrowLTransposeSolve(diag_, offDiag_, false, v);
      break;
    default:
      assert(false);
      break;
  }
}

void jrl::qp::structured::StructuredG::solveL(VectorRef out, const VectorConstRef & in) const
{
  assert(decomposed_);
  assert(in.size() == nbVar_);
  assert(out.size() == nbVar_);
  out = in;

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      decomposition::triBlockDiagLSolve(diag_, offDiag_, out);
      break;
    case Type::BlockArrowUp:
      decomposition::blockArrowLSolve(diag_, offDiag_, true, out);
      break;
    case Type::BlockArrowDown:
      decomposition::blockArrowLSolve(diag_, offDiag_, false, out);
      break;
    default:
      assert(false);
      break;
  }
}

void jrl::qp::structured::StructuredG::solveL(VectorRef out, const internal::SingleNZSegmentVector & in) const
{
  assert(decomposed_);
  assert(in.size() == nbVar_);
  assert(out.size() == nbVar_);
  in.toFullVector(out);

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      decomposition::triBlockDiagLSolve(diag_, offDiag_, out, in.start());
      break;
    case Type::BlockArrowUp:
      decomposition::blockArrowLSolve(diag_, offDiag_, true, out, in.start(), in.end());
      break;
    case Type::BlockArrowDown:
      decomposition::blockArrowLSolve(diag_, offDiag_, false, out, in.start(), in.end());
      break;
    default:
      assert(false);
      break;
  }
}
