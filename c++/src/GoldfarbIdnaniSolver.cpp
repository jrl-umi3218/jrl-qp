/* Copyright 2020 CNRS-AIST JRL
 */

#include <jrl-qp/GoldfarbIdnaniSolver.h>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Jacobi>

using Givens = Eigen::JacobiRotation<double>;

namespace jrlqp
{
  GoldfarbIdnaniSolver::GoldfarbIdnaniSolver()
    : work_d_(0)
    , work_J_(0)
    , work_R_(0)
  {
  }

  GoldfarbIdnaniSolver::GoldfarbIdnaniSolver(int nbVar, int nbCstr, bool useBounds)
    : GoldfarbIdnaniSolver()
  {
    resize_(nbVar, nbCstr, useBounds);
  }

  TerminationStatus GoldfarbIdnaniSolver::solve(MatrixRef G, const VectorConstRef& a, const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu, const VectorConstRef& xl, const VectorConstRef& xu)
  {
    int nbVar = G.rows();
    int nbCstr = C.cols();
    bool useBnd = xl.size() > 0;

    LOG_RESET(log_);
    LOG(log_, LogFlags::INPUT|LogFlags::NO_ITER, G, a, C, bl, bu, xl, xu);

    assert(G.cols() == nbVar);
    assert(a.size() == nbVar);
    assert(C.rows() == nbVar);
    assert(bl.size() == nbCstr);
    assert(bu.size() == nbCstr);
    assert(xl.size() == nbVar || xl.size() == 0);
    assert(xu.size() == xl.size());

    new (&pb_.G) MatrixRef(G);
    new (&pb_.a) VectorConstRef(a);
    new (&pb_.C) MatrixConstRef(C);
    new (&pb_.bl) VectorConstRef(bl);
    new (&pb_.bu) VectorConstRef(bu);
    new (&pb_.xl) VectorConstRef(xl);
    new (&pb_.xu) VectorConstRef(xu);

    resize(nbVar, nbCstr, useBnd);

    return DualSolver::solve();
  }

  void GoldfarbIdnaniSolver::init_()
  {
    Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb_.G);
    auto L = pb_.G.template triangularView<Eigen::Lower>();

    // J = L^-t
    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    J.setIdentity();
    L.transpose().solveInPlace(J);

    // x = -G^-1 * a
    auto x = work_x_.asVector(nbVar_);
    x = L.solve(pb_.a);
    L.transpose().solveInPlace(x); // possible [OPTIM]: J already contains L^-T 
    x = -x;
    f_ = 0.5 * pb_.a.dot(x);

    A_.reset();
    DEBUG_ONLY(work_R_.setZero());
  }

  internal::ConstraintNormal GoldfarbIdnaniSolver::selectViolatedConstraint_(const VectorConstRef& x) const
  {
    //We look for the constraint with the maximum violation
    //[NUMERIC] scale with constraint magnitude
    double smin = 0;
    int p = -1;
    ActivationStatus status = ActivationStatus::INACTIVE;

    //Check general constraints
    for (int i = 0; i < A_.nbCstr(); ++i)
    {
      if (!A_.isActive(i))
      {
        double cx = pb_.C.col(i).dot(x); //possible [OPTIM]: should we compute C^T x at once ?
        if (double sl = cx - pb_.bl[i]; sl < smin)
        {
          smin = sl;
          p = i;
          status = ActivationStatus::LOWER;
        }
        else if (double su = pb_.bu[i] - cx; su < smin)
        {
          smin = su;
          p = i;
          status = ActivationStatus::UPPER;
        }
      }
    }

    //Check bound constraints
    for (int i = 0; i < A_.nbBnd(); ++i)
    {
      if (!A_.isActiveBnd(i))
      {
        if (double sl = x[i] - pb_.xl[i]; sl < smin)
        {
          smin = sl;
          p = A_.nbCstr() + i;
          status = ActivationStatus::LOWER_BOUND;
        }
        else if (double su = pb_.xu[i] - x[i]; su < smin)
        {
          smin = su;
          p = A_.nbCstr() + i;
          status = ActivationStatus::UPPER_BOUND;
        }
      }
    }

    return { pb_.C, p, status };
  }

  void GoldfarbIdnaniSolver::computeStep_(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const
  {
    int q = A_.nbActiveCstr();
    auto d = work_d_.asVector(nbVar_, {});
    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    auto R = work_R_.asMatrix(q, q, nbVar_).template triangularView<Eigen::Upper>();

    np.preMultiplyByMt(d, J);
    z.noalias() = J.rightCols(nbVar_ - q) * d.tail(nbVar_ - q);
    r = R.solve(d.head(q));
    DBG(log_, LogFlags::ITERATION_ADVANCE_DETAILS, J, R, d);
  }

  DualSolver::StepLenghth GoldfarbIdnaniSolver::computeStepLength_(const internal::ConstraintNormal& np, const VectorConstRef& x, const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const
  {
    double t1 = options_.bigBnd_;
    double t2 = options_.bigBnd_;
    int l = 0;

    for (int k = 0; k < A_.nbActiveCstr(); ++k)
    {
      if (A_.activationStatus(k) != ActivationStatus::EQUALITY && r[k] > 0)
      {
        if (double tk = u[k] / r[k]; tk < t1)
        {
          t1 = tk;
          l = k;
        }
      }
    }

    if (z.norm() > 1e-14) //[NUMERIC] better criterion
    {
      double b, cx, cz;
      int p, pb;
      switch (np.status())
      {
      case ActivationStatus::LOWER:
        p = np.index(); b = pb_.bl[p]; cx = pb_.C.col(p).dot(x); cz = pb_.C.col(p).dot(z);
        break;
      case ActivationStatus::UPPER:
        p = np.index(); b = pb_.bu[p]; cx = pb_.C.col(p).dot(x); cz = pb_.C.col(p).dot(z);
      break;
      case ActivationStatus::EQUALITY: 
        assert(false); 
        break;
      case ActivationStatus::LOWER_BOUND:
        pb = np.bndIndex(); b = pb_.xl[pb]; cx = x[pb]; cz = z[pb];
        break;
      case ActivationStatus::UPPER_BOUND:
        pb = np.bndIndex(); b = pb_.xu[pb]; cx = x[pb]; cz = z[pb];
        break;
      default:
        assert(false);
      }
      t2 = (b - cx) / cz;
    }

    return { t1, t2, l };
  }

  bool GoldfarbIdnaniSolver::addConstraint_(const internal::ConstraintNormal& np)
  {
    int q = A_.nbActiveCstr(); // This already counts the new constraint
    auto d = work_d_.asVector(nbVar_);
    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    for (int i = nbVar_ - 2; i >= q-1; --i) //[OPTIM] use Householder transformation instead
    {
      Givens Qi;
      Qi.makeGivens(d[i], d[i + 1], &d[i]);
      DEBUG_ONLY(d[i + 1] = 0);
      J.applyOnTheRight(i, i + 1, Qi);
    }
    auto R = work_R_.asMatrix(q, q, nbVar_);
    R.rightCols<1>() = d.head(q);

    return true; //[NUMERIC]: add test on dependency
  }

  bool GoldfarbIdnaniSolver::removeConstraint_(int l)
  {
    int q = A_.nbActiveCstr(); // This already counts that the constraint was removed
    auto d = work_d_.asVector(nbVar_);
    auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
    auto R = work_R_.asMatrix(q + 1, q + 1, nbVar_);
    
    for (int i = l; i < q; ++i)
    {
      Givens Qi;
      R.col(i).head(i) = R.col(i + 1).head(i);
      Qi.makeGivens(R(i, i + 1), R(i + 1, i + 1), &R(i, i));
      DEBUG_ONLY(R(i + 1, i + 1) = 0);
      R.rightCols(q - i - 1).applyOnTheLeft(i, i + 1, Qi.transpose());
      J.applyOnTheRight(i, i + 1, Qi);
    }

    return true;
  }

  void GoldfarbIdnaniSolver::resize_(int nbVar, int nbCstr, bool useBounds)
  {
    if (nbVar != nbVar_)
    {
      work_d_.resize(nbVar);
      work_J_.resize(nbVar, nbVar);
      work_R_.resize(nbVar, nbVar);
    }
  }
}