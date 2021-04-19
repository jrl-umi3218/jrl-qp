/* Copyright 2020 CNRS-AIST JRL */

#include <jrl-qp/experimental/BoxAndSingleConstraintSolver.h>

namespace jrl::qp::experimental
{
BoxAndSingleConstraintSolver::BoxAndSingleConstraintSolver() : GoldfarbIdnaniSolver() {}

BoxAndSingleConstraintSolver::BoxAndSingleConstraintSolver(int nbVar) : GoldfarbIdnaniSolver(nbVar, 1, true) {}

TerminationStatus BoxAndSingleConstraintSolver::solve(const VectorConstRef & x0,
                                                      const VectorConstRef & c,
                                                      double bl,
                                                      const VectorConstRef & xl,
                                                      const VectorConstRef & xu)
{
  int nbVar = static_cast<int>(x0.size());
  int nbCstr = static_cast<int>(c.cols());
  bool useBnd = xl.size() > 0;

  JRLQP_LOG_RESET(log_);
  JRLQP_LOG(log_, LogFlags::INPUT | LogFlags::NO_ITER, x0, c, bl, xl, xu);

  assert(c.rows() == nbVar);
  assert(c.cols() == 1);
  assert(xl.size() == nbVar);
  assert(xu.size() == xl.size());

  // TODO check input: bl<=bu, xl<=xu, ...

  bl_[0] = bl;
  bu_[0] = std::numeric_limits<double>::infinity();

  new(&pb_.a) VectorConstRef(x0); // a will contain x0, not -x0 as it should be.
  new(&pb_.C) MatrixConstRef(c);
  new(&pb_.bl) VectorConstRef(bl_);
  new(&pb_.bu) VectorConstRef(bu_);
  new(&pb_.xl) VectorConstRef(xl);
  new(&pb_.xu) VectorConstRef(xu);

  resize(nbVar, nbCstr, useBnd);

  return DualSolver::solve();
}

internal::InitTermination BoxAndSingleConstraintSolver::init_()
{
  A_.reset();
  JRLQP_DEBUG_ONLY(work_R_.setZero());

  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  auto R = work_R_.asMatrix(nbVar_, nbVar_, nbVar_);

  auto x = work_x_.asVector(nbVar_);
  auto u = work_u_.asVector(nbVar_);
  auto x0 = pb_.a;

  int q = 0;
  f_ = 0;
  J.setZero();
  for(int i = 0; i < nbVar_; ++i)
  {
    if(x0[i] < pb_.xl[i])
    {
      x[i] = pb_.xl[i];
      u[q] = x[i] - x0[i];
      f_ += 0.5 * u[q] * u[q];
      R(q, q) = 1;
      R.col(q).head(q).setZero();
      J(i, q) = 1;
      A_.activate(i + 1, ActivationStatus::LOWER_BOUND); // i+1 because we have one non-bound constraint
      ++q;
    }
    else if(x0[i] > pb_.xu[i])
    {
      x[i] = pb_.xu[i];
      u[q] = x0[i] - x[i];
      f_ += 0.5 * u[q] * u[q];
      R(q, q) = -1;
      R.col(q).head(q).setZero();
      J(i, q) = 1;
      A_.activate(i + 1, ActivationStatus::UPPER_BOUND); // i+1 because we have one non-bound constraint
      ++q;
    }
    else
    {
      x[i] = x0[i];
      J(i, nbVar_ - i + q - 1) = 1;
    }
  }
  JRLQP_LOG(log_, LogFlags::ACTIVE_SET_DETAILS, x, u, J, R);

  return TerminationStatus::SUCCESS;
}
} // namespace jrl::qp::experimental

namespace jrl::qp::test
{
LeastSquareProblem<> generateBoxAndSingleConstraintProblem(int nbVar, bool act, double actLevel)
{
  assert((!act || (actLevel > 0 && actLevel < 1)) && "when act is true, actLevel must be strictly between 0 and 1.");
  Eigen::VectorXd x0 = Eigen::VectorXd::Random(nbVar);
  Eigen::VectorXd r1 = Eigen::VectorXd::Random(nbVar);
  Eigen::VectorXd r2 = Eigen::VectorXd::Random(nbVar);
  Eigen::VectorXd xl(nbVar);
  Eigen::VectorXd xu(nbVar);
  Eigen::VectorXd xb(nbVar); // closest point to x0 satisfying the bounds
  for(int i = 0; i < nbVar; ++i)
  {
    // xl = min(r1, r2)
    // xu = max(r1, r2)
    if(r1[i] < r2[i])
    {
      xl[i] = r1[i];
      xu[i] = r2[i];
    }
    else
    {
      xl[i] = r2[i];
      xu[i] = r1[i];
    }
    // clamp x0 in the box [xl, xu]
    if(x0[i] < xl[i])
      xb[i] = xl[i];
    else if(x0[i] > xu[i])
      xb[i] = xu[i];
    else
      xb[i] = x0[i];
  }

  Eigen::VectorXd c = Eigen::VectorXd::Random(nbVar);
  // Points of the box the closer and further away in the direction of c
  Eigen::VectorXd sl(nbVar);
  Eigen::VectorXd su(nbVar);
  for(int i = 0; i < nbVar; ++i)
  {
    if(c[i] > 0)
    {
      sl[i] = xl[i];
      su[i] = xu[i];
    }
    else
    {
      sl[i] = xu[i];
      su[i] = xl[i];
    }
  }

  double b;
  if(act)
  {
    // If we want the constraint c'x >= b active we chose b so that d2 <= c'x <= d1
    double d1 = c.dot(xb); // minimum b so that the constraint is active
    double d2 = c.dot(su); // maximum b so that the problem is feasible
    b = actLevel * d1 + (1 - actLevel) * d2;
  }
  else
  {
    b = c.dot(sl); // the constraint doesn't intersect the box
  }

  test::LeastSquareProblem<> pb;
  pb.A = Eigen::MatrixXd::Identity(nbVar, nbVar);
  pb.b = x0;
  pb.C = c;
  pb.l = Eigen::VectorXd(1);
  pb.l[0] = b;
  pb.u = Eigen::VectorXd(1);
  pb.u[0] = std::numeric_limits<double>::infinity();
  pb.xl = xl;
  pb.xu = xu;

  return pb;
}
} // namespace jrl::qp::test
