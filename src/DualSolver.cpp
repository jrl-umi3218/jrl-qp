/* Copyright 2020 CNRS-AIST JRL */

#include <jrl-qp/DualSolver.h>

namespace jrl::qp
{
  DualSolver::DualSolver()
    : options_(SolverOptions())
    , log_(SolverOptions::defaultStream_, "log")
    , nbVar_(0)
    , A_(0)
    , f_(0)
    , work_x_(0)
    , work_z_(0)
    , work_u_(0)
    , work_r_(0)
    , needToExpandMultipliers_(false)
  {
  }

  DualSolver::DualSolver(int nbVar, int nbCstr, bool useBounds)
    : DualSolver()
  {
    resize_p(nbVar, nbCstr, useBounds);
  }

  void DualSolver::resize(int nbVar, int nbCstr, bool useBounds)
  {
    // We resize first the derived class so that it can make comparisons
    // based on the current problem size.
    resize_(nbVar, nbCstr, useBounds);
    resize_p(nbVar, nbCstr, useBounds);
  }

  void DualSolver::options(const SolverOptions& options)
  {
    options_ = options;
    log_.setFlag(options.logFlags_);
    log_.setOutputStream(*options_.logStream_);
  }

  WConstVector DualSolver::solution() const
  {
    return work_x_.asVector(nbVar_);
  }

  WConstVector DualSolver::multipliers() const
  {
    int m = A_.nbAll();
    // During computations, we only keep track of the Lagrange multipliers for the active
    // constraints. We also use conventions on the constraints so that multipliers are 
    // always positive. We need to adapt our internal values to the external representation.
    if (needToExpandMultipliers_)
    {
      needToExpandMultipliers_ = false;
      int q = A_.nbActiveCstr();
      //we temporarily store the condensed multipliers in work_r;
      auto r = work_r_.asVector(q, {});
      r = work_u_.asVector(q);

      //we copy the non-zero values to their correct positions, with the correct signs
      work_u_.setZero();
      auto u = work_u_.asVector(m, {});
      for (int k = 0; k < A_.nbActiveCstr(); ++k)
      {
        int i = A_[k];
        if (A_.activationStatus(i) == ActivationStatus::UPPER
          || A_.activationStatus(i) == ActivationStatus::UPPER_BOUND)
        {
          u[i] = r[k];
        }
        else
        {
          u[i] = -r[k];
        }
      }
    }
    return const_cast<const internal::Workspace<>&>(work_u_).asVector(m);
  }

  double DualSolver::objectiveValue() const
  {
    return f_;
  }

  TerminationStatus DualSolver::solve()
  {
    bool skipStep1 = false;
    internal::ConstraintNormal np;
    WVector x = work_x_.asVector(nbVar_);
    WVector z = work_z_.asVector(nbVar_);
    WVector u = work_u_.asVector(0);
    WVector r = work_r_.asVector(0);

    if (auto rt = init(); !rt) //step 0
      return terminate(rt);

    for (int it = 0; it < options_.maxIter_; ++it)
    {
      LOG_NEW_ITER(log_, it);
      LOG(log_, LogFlags::ACTIVE_SET_DETAILS, A_);
      int q = A_.nbActiveCstr();
      LOG(log_, LogFlags::ITERATION_BASIC_DETAILS, x, u, f_);

      // Step 1
      if (!skipStep1)
      {
        np = selectViolatedConstraint(x);
        if (np.status() == ActivationStatus::INACTIVE)
          return terminate(TerminationStatus::SUCCESS);

        LOG(log_, LogFlags::ACTIVE_SET, np);
        //LOG_AS(log_, LogFlags::ACTIVE_SET, "selectedConstraint", np.index(), "status", static_cast<int>(np.status()));
        new (&r) WVector(work_r_.asVector(q));
        new (&u) WVector(work_u_.asVector(q + 1));
        u[q] = 0;
      }

      // Step 2
      computeStep(z, r, np);
      auto [t1, t2, l] = computeStepLength(np, x, u, z, r);
      double t = std::min(t1, t2);
      LOG_COMMENT(log_, LogFlags::ITERATION_BASIC_DETAILS, "Step computation");
      LOG(log_, LogFlags::ITERATION_BASIC_DETAILS, z, r, t);

      if (t >= options_.bigBnd_)
        return terminate(TerminationStatus::INFEASIBLE);

      if (t2 >= options_.bigBnd_)
      {
        // u = u + t*[-r;1]
        u.head(q) -= t * r; u[q] += t;
        LOG_AS(log_, LogFlags::ACTIVE_SET, "Activate", false, "l", l);
        removeConstraint(l, u);
        new (&r) WVector(work_r_.asVector(q - 1));
        new (&u) WVector(work_u_.asVector(q));
        skipStep1 = true;
      }
      else
      {
        x += t * z;
        f_ += t * np.dot(z) * (.5 * t + u[q]);
        // u = u + t*[-r;1]
        u.head(q) -= t * r; u[q] += t;
        if (t == t2)
        {
          LOG_AS(log_, LogFlags::ACTIVE_SET, "Activate", true, "l", l);
          if (!addConstraint(np))
            return terminate(TerminationStatus::LINEAR_DEPENDENCY_DETECTED);
          skipStep1 = false;
        }
        else
        {
          LOG_AS(log_, LogFlags::ACTIVE_SET, "Activate", false, "l", l);
          removeConstraint(l, u);
          new (&r) WVector(work_r_.asVector(q - 1));
          new (&u) WVector(work_u_.asVector(q));
          skipStep1 = true;
        }
      }
    }
    return terminate(TerminationStatus::MAX_ITER_REACHED);
  }

  TerminationStatus DualSolver::terminate(TerminationStatus status)
  {
    switch (status)
    {
    case TerminationStatus::SUCCESS:
      LOG_COMMENT(log_, LogFlags::TERMINATION, "Optimum reached.");
      break;
    case TerminationStatus::INCONSISTENT_INPUT:
      LOG_COMMENT(log_, LogFlags::TERMINATION, "Inconsistent inputs.");
      break;
    case TerminationStatus::NON_POS_HESSIAN:
      LOG_COMMENT(log_, LogFlags::TERMINATION, 
        "This version of the solver requires the quadratic matrix to be positive definite." 
        "The input matrix is not (at least numerically)");
      break;
    case TerminationStatus::INFEASIBLE:
      LOG_COMMENT(log_, LogFlags::TERMINATION, "Infeasible problem.");
      break;
    case TerminationStatus::MAX_ITER_REACHED:
      LOG_COMMENT(log_, LogFlags::TERMINATION, "Maximum number of iteration reached.");
      break;
    case TerminationStatus::LINEAR_DEPENDENCY_DETECTED:
      LOG_COMMENT(log_, LogFlags::TERMINATION, "Attempting to add a linearly dependent constraint.");
      break;
    default: assert(false);
    }
    return status;
  }

  internal::InitTermination DualSolver::init()
  {
    DEBUG_ONLY(work_u_.setZero());
    DEBUG_ONLY(work_r_.setZero());

    needToExpandMultipliers_ = true;
    A_.reset();
    return init_();
  }

  internal::ConstraintNormal DualSolver::selectViolatedConstraint(const VectorConstRef& x) const
  {
    return selectViolatedConstraint_(x);
  }

  void DualSolver::computeStep(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const
  {
    computeStep_(z, r, np);
  }

  DualSolver::StepLenghth DualSolver::computeStepLength(const internal::ConstraintNormal& np, const VectorConstRef& x, const VectorConstRef& u,
    const VectorConstRef& z, const VectorConstRef& r) const
  {
    return computeStepLength_(np, x, u, z, r);
  }

  bool DualSolver::addConstraint(const internal::ConstraintNormal& np)
  {
    A_.activate(np.index(), np.status());
    return addConstraint_(np);
  }

  bool DualSolver::removeConstraint(int l, VectorRef u)
  {
    int q = A_.nbActiveCstr();
    u.segment(l, q - l) = u.tail(q - l);
    DEBUG_ONLY(u[q] = 0);
    A_.deactivate(l);
    return removeConstraint_(l);
  }

  void DualSolver::resize_p(int nbVar, int nbCstr, bool useBounds)
  {
    if (nbVar != nbVar_)
    {
      nbVar_ = nbVar;
      work_x_.resize(nbVar);
      work_z_.resize(nbVar);
    }
    int nbBnd = useBounds ? nbVar : 0;
    if (nbCstr + nbBnd != A_.nbAll())
    {
      A_.resize(nbCstr, nbBnd);
      // We need to have work_u_ size at least as big as the number of constraints because we
      // are using it for storing the final multipliers.
      // work_r_ size could be restricted to at most nbVar, as long as we make sure that we'll
      // never have more than nbVar active constraints, which should always be the case.
      // However, this would complicate the resize logic.
      work_u_.resize(nbCstr + nbBnd);
      work_r_.resize(nbCstr + nbBnd);
    }

  }
}