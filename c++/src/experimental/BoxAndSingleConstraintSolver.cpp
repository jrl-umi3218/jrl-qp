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
    A_.reset();
    DEBUG_ONLY(work_R_.setZero());

    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    auto R = work_R_.asMatrix(nbVar_, nbVar_, nbVar_);

    auto x = work_x_.asVector(nbVar_);
    auto u = work_u_.asVector(nbVar_);
    auto x0 = pb_.a;

    int q = 0;
    J.setZero();
    for (int i = 0; i < nbVar_; ++i)
    {
      if (x0[i] < pb_.xl[i])
      {
        x[i] = pb_.xl[i];
        u[q] = x[i] - x0[i];
        R(q, q) = 1;
        R.col(q).head(q).setZero();
        J(i, q) = 1;
        A_.activate(i + 1, ActivationStatus::LOWER_BOUND); // i+1 because we have one non-bound constraint
        ++q;
      }
      else if (x0[i] > pb_.xu[i])
      {
        x[i] = pb_.xu[i];
        u[q] = x0[i] - x[i];
        R(q, q) = -1;
        R.col(q).head(q).setZero();
        J(i, q) = 1;
        A_.activate(i + 1, ActivationStatus::UPPER_BOUND); // i+1 because we have one non-bound constraint
        ++q;
      }
      else
      {
        x[i] = x0[i];
        J(i, nbVar_-i+q-1) = 1;
      }
    }
    LOG(log_, LogFlags::ACTIVE_SET_DETAILS, x, u, J, R);

    return TerminationStatus::SUCCESS;
  }
}