#include <jrl-qp/DualSolver.h>

namespace jrlqp
{
  TerminationStatus DualSolver::solve()
  {
    bool skipStep1 = false;
    internal::ConstraintNormal np;
    EigenHead u = work_u_.head(0);
    EigenHead r = work_r_.head(0);

    init(); //step 0

    for (int it = 0; it < options_.maxIter_; ++it)
    {
      LOG_NEW_ITER(log_, it);
      int q = A_.nbActiveCstr();

      // Step 1
      if (!skipStep1)
      {
        np = selectViolatedConstraint(x_);
        if (np.status() == ActivationStatus::INACTIVE)
        {
          LOG_COMMENT(log_, LogFlags::TERMINATION, "Optimum reached");
          return TerminationStatus::SUCCESS;
        }

        r = work_r_.head(q);
        u = work_u_.head(q + 1);
      }

      // Step 2
      computeStep(z_, r, np);
      auto [t1, t2, l] = computeStepLength(np, x_, u, z_, r);
      double t = std::min(t1, t2);

      if (t >= options_.bigBnd_)
      {
        LOG_COMMENT(log_, LogFlags::TERMINATION, "Infeasible problem");
        return TerminationStatus::INFEASIBLE;
      }

      u.head(q) -= t * r;
      u[q] = t;
      if (t2 >= options_.bigBnd_)
      {
        removeConstraint(l, u);
        skipStep1 = true;
      }
      else
      {
        x_ += t * z_;
        f_ += t * np.dot(z_) * (.5 * t + u[q]);
        if (t == t2)
        {
          if (!addConstraint(np))
          {
            LOG_COMMENT(log_, LogFlags::TERMINATION, "Attempting to add a linearly dependent constraint.");
            return TerminationStatus::LINEAR_DEPENDENCY_DETECTED;
          }
        }
        else
        {
          removeConstraint(l, u);
        }
      }
    }
    LOG_COMMENT(log_, LogFlags::TERMINATION, "Maximum number of iteration reached");
    return TerminationStatus::MAX_ITER_REACHED;
  }

  void DualSolver::init()
  {
    init_();
  }

  internal::ConstraintNormal DualSolver::selectViolatedConstraint(const Eigen::VectorXd& x) const
  {
    return selectViolatedConstraint_(x);
  }

  void DualSolver::computeStep(Eigen::VectorXd& z, EigenHead& r, const internal::ConstraintNormal& np) const
  {
    computeStep_(z, r, np);
  }

  std::tuple<double, double, int> DualSolver::computeStepLength(const internal::ConstraintNormal& np, const Eigen::VectorXd& x, const EigenHead& u, const Eigen::VectorXd& z, const EigenHead& r) const
  {
    return computeStepLength_(np, x, u, z, r);
  }

  bool DualSolver::addConstraint(const internal::ConstraintNormal& np)
  {
    A_.activate(np.index(), np.status());
    return addConstraint_(np);
  }

  bool DualSolver::removeConstraint(int l, EigenHead& u)
  {
    int q = A_.nbActiveCstr();
    A_.deactivate(l);
    u.segment(l, q - l - 1) = u.tail(l - q - 1);
    u = work_u_.head(q - 1);
    return removeConstraint_(l);
  }
}