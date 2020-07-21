/* Copyright 2020 CNRS-AIST JRL
 */

#include <jrl-qp/experimental/BoxAndSingleConstraintSolver.h>

namespace jrlqp::experimental
{
  BoxAndSingleConstraintSolver::BoxAndSingleConstraintSolver()
    : GoldfarbIdnaniSolver()
  {
  }
  
  BoxAndSingleConstraintSolver::BoxAndSingleConstraintSolver(int nbVar)
    : GoldfarbIdnaniSolver(nbVar, 1, true)
  {
  }

  TerminationStatus BoxAndSingleConstraintSolver::solve(const VectorConstRef& x0, const VectorConstRef& c, double bl, const VectorConstRef& xl, const VectorConstRef& xu)
  {
    int nbVar = x0.size();
    int nbCstr = c.cols();
    bool useBnd = xl.size() > 0;

    LOG_RESET(log_);
    LOG(log_, LogFlags::INPUT | LogFlags::NO_ITER, x0, c, bl, xl, xu);

    assert(c.rows() == nbVar);
    assert(c.cols() == 1);
    assert(xl.size() == nbVar);
    assert(xu.size() == xl.size());

    //TODO check input: bl<=bu, xl<=xu, ...

    bl_[0] = bl;
    bu_[0] = std::numeric_limits<double>::infinity();

    new (&pb_.a) VectorConstRef(x0); // a will contain x0, not -x0 as it should be.
    new (&pb_.C) MatrixConstRef(c);
    new (&pb_.bl) VectorConstRef(bl_);
    new (&pb_.bu) VectorConstRef(bu_);
    new (&pb_.xl) VectorConstRef(xl);
    new (&pb_.xu) VectorConstRef(xu);

    resize(nbVar, nbCstr, useBnd);

    return DualSolver::solve();
  }

  internal::InitTermination BoxAndSingleConstraintSolver::init_()
  {
    // J = L^-t = I
    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    J.setIdentity();

    // x = x0
    auto x = work_x_.asVector(nbVar_);
    x = pb_.a;
    f_ = 0;

    A_.reset();
    DEBUG_ONLY(work_R_.setZero());

    return TerminationStatus::SUCCESS;
  }
}