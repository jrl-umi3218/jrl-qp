#include <jrl-qp/structured/StructuredG.h>

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
    case Type::BlockArrow:
      assert(false);
      done = false;
      break;
    case Type::BlockArrowWithDiagOffBlocks:
      assert(false);
      done =  false;
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
  assert(v.size() == nbVar_);

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      return decomposition::triBlockDiagLTransposeSolve(diag_, offDiag_, v);
      break;
    case Type::BlockArrow:
      assert(false);
      break;
    case Type::BlockArrowWithDiagOffBlocks:
      assert(false);
      break;
    default:
      assert(false);
      break;
  }
}

void jrl::qp::structured::StructuredG::solveL(VectorRef out, const VectorConstRef & in) const
{
  assert(in.size() == nbVar_);
  assert(out.size() == nbVar_);
  out = in;

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      return decomposition::triBlockDiagLSolve(diag_, offDiag_, out);
      break;
    case Type::BlockArrow:
      assert(false);
      break;
    case Type::BlockArrowWithDiagOffBlocks:
      assert(false);
      break;
    default:
      assert(false);
      break;
  }
}

void jrl::qp::structured::StructuredG::solveL(VectorRef out, const internal::SingleNZSegmentVector & in) const
{
  assert(in.size() == nbVar_);
  assert(out.size() == nbVar_);
  in.toFullVector(out);

  switch(type_)
  {
    case Type::TriBlockDiagonal:
      return decomposition::triBlockDiagLSolve(diag_, offDiag_, out, in.start());
      break;
    case Type::BlockArrow:
      assert(false);
      break;
    case Type::BlockArrowWithDiagOffBlocks:
      assert(false);
      break;
    default:
      assert(false);
      break;
  }
}
