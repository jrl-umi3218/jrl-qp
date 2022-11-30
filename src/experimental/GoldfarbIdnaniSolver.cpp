/* Copyright 2020 CNRS-AIST JRL */

#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/QR>
#include <jrl-qp/experimental/GoldfarbIdnaniSolver.h>
#include <jrl-qp/internal/ConstraintNormal.h>

using Givens = Eigen::JacobiRotation<double>;

namespace jrl::qp::experimental
{
GoldfarbIdnaniSolver::GoldfarbIdnaniSolver() : DualSolver(), work_d_(0), work_J_(0), work_R_(0) {}

GoldfarbIdnaniSolver::GoldfarbIdnaniSolver(int nbVar, int nbCstr, bool useBounds) : GoldfarbIdnaniSolver()
{
  resize(nbVar, nbCstr, useBounds);
}

TerminationStatus GoldfarbIdnaniSolver::solve(MatrixRef G,
                                              const VectorConstRef & a,
                                              const MatrixConstRef & C,
                                              const VectorConstRef & bl,
                                              const VectorConstRef & bu,
                                              const VectorConstRef & xl,
                                              const VectorConstRef & xu,
                                              const std::vector<ActivationStatus> & as)
{
  int nbVar = static_cast<int>(G.rows());
  int nbCstr = static_cast<int>(C.cols());
  bool useBnd = xl.size() > 0;

  JRLQP_LOG_RESET(log_);
  JRLQP_LOG(log_, LogFlags::INPUT | LogFlags::NO_ITER, G, a, C, bl, bu, xl, xu, as);

  // Check coherence of the input sizes
  assert(G.cols() == nbVar);
  assert(a.size() == nbVar);
  assert(C.rows() == nbVar);
  assert(bl.size() == nbCstr);
  assert(bu.size() == nbCstr);
  assert(xl.size() == nbVar || xl.size() == 0);
  assert(xu.size() == xl.size());
  assert(static_cast<int>(as.size()) == nbCstr + xl.size() || as.size() == 0);
  assert(options_.warmStart_ || as.empty() && "Non-empty active set used with cold start option.");

  // TODO check input: bl<=bu, xl<=xu, ...

  new(&pb_.G) MatrixRef(G);
  new(&pb_.a) VectorConstRef(a);
  new(&pb_.C) MatrixConstRef(C);
  new(&pb_.bl) VectorConstRef(bl);
  new(&pb_.bu) VectorConstRef(bu);
  new(&pb_.xl) VectorConstRef(xl);
  new(&pb_.xu) VectorConstRef(xu);

  resize(nbVar, nbCstr, useBnd);
  if(options_.warmStart_)
  {
    pb_.as = as.empty() ? A_.activationStatus() : as;
  }

  return DualSolver::solve();
}

internal::InitTermination GoldfarbIdnaniSolver::init_()
{
  JRLQP_DEBUG_ONLY(work_R_.setZero());

  // Decide the initial active set given the data and the options
  auto retAS = processInitialActiveSet();
  if(!retAS) return retAS;

  // [OPTIM]: this is not necessary if there are nbVar equality constraints
  auto ret = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(pb_.G);

  if(ret >= 0) return TerminationStatus::NON_POS_HESSIAN;

  initializeComputationData();
  initializePrimalDualPoints();

  // If some constraints have ben activated with u<0, we deactivate them
  while(true)
  {
    int q = A_.nbActiveCstr();
    WVector u = work_u_.asVector(q);
    WVector b_act = work_bact_.asVector(q);
    double umin = -1e-14; // [Numerics] Do better.
    int lmin = -1;
    for(int l = 0; l < q; ++l)
    {
      int i = A_[l];
      if(u[l] < umin && A_.activationStatus(i) != ActivationStatus::FIXED
         && A_.activationStatus(i) != ActivationStatus::EQUALITY)
      {
        umin = u[l];
        lmin = l;
      }
    }
    if(lmin < 0) break; // no more constraint to deactivate

    ++it_;
    b_act.segment(lmin, q - 1 - lmin) = b_act.tail(q - 1 - lmin);
    JRLQP_DEBUG_ONLY(u[q - 1] = 0);
    A_.deactivate(lmin);
    removeConstraint_(lmin);
    initializePrimalDualPoints();
  }

  return TerminationStatus::SUCCESS;
}

internal::SelectedConstraint GoldfarbIdnaniSolver::selectViolatedConstraint_(const VectorConstRef & x) const
{
  // We look for the constraint with the maximum violation
  //[NUMERIC] scale with constraint magnitude
  double smin = 0;
  int p = -1;
  ActivationStatus status = ActivationStatus::INACTIVE;

  // Check general constraints
  for(int i = 0; i < A_.nbCstr(); ++i)
  {
    if(!A_.isActive(i))
    {
      double cx = pb_.C.col(i).dot(x); // possible [OPTIM]: should we compute C^T x at once ?
      if(double sl = cx - pb_.bl[i]; sl < smin)
      {
        smin = sl;
        p = i;
        status = ActivationStatus::LOWER;
      }
      else if(double su = pb_.bu[i] - cx; su < smin)
      {
        smin = su;
        p = i;
        status = ActivationStatus::UPPER;
      }
    }
  }

  // Check bound constraints
  for(int i = 0; i < A_.nbBnd(); ++i)
  {
    if(!A_.isActiveBnd(i))
    {
      if(double sl = x[i] - pb_.xl[i]; sl < smin)
      {
        smin = sl;
        p = A_.nbCstr() + i;
        status = ActivationStatus::LOWER_BOUND;
      }
      else if(double su = pb_.xu[i] - x[i]; su < smin)
      {
        smin = su;
        p = A_.nbCstr() + i;
        status = ActivationStatus::UPPER_BOUND;
      }
    }
  }

  return {p, status};
}

void GoldfarbIdnaniSolver::computeStep_(VectorRef z, VectorRef r, const internal::SelectedConstraint & sc) const
{
  int q = A_.nbActiveCstr();
  auto d = work_d_.asVector(nbVar_, {});
  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  auto R = work_R_.asMatrix(q, q, nbVar_).template triangularView<Eigen::Upper>();

  internal::ConstraintNormal np(pb_.C, sc);
  np.preMultiplyByMt(d, J);
  z.noalias() = J.rightCols(nbVar_ - q) * d.tail(nbVar_ - q);
  r = R.solve(d.head(q));
  JRLQP_DBG(log_, LogFlags::ITERATION_ADVANCE_DETAILS, J, R, d);
}

DualSolver::StepLength GoldfarbIdnaniSolver::computeStepLength_(const internal::SelectedConstraint & sc,
                                                                const VectorConstRef & x,
                                                                const VectorConstRef & u,
                                                                const VectorConstRef & z,
                                                                const VectorConstRef & r) const
{
  double t1 = options_.bigBnd_;
  double t2 = options_.bigBnd_;
  int l = 0;

  for(int k = 0; k < A_.nbActiveCstr(); ++k)
  {
    if(A_.activationStatus(k) != ActivationStatus::EQUALITY && A_.activationStatus(k) != ActivationStatus::FIXED
       && r[k] > 0)
    {
      if(double tk = u[k] / r[k]; tk < t1)
      {
        t1 = tk;
        l = k;
      }
    }
  }

  internal::ConstraintNormal np(pb_.C, sc);
  if(z.norm() > 1e-14) //[NUMERIC] better criterion
  {
    double b, cx, cz;
    int p, pb;
    switch(np.status())
    {
      case ActivationStatus::LOWER:
        p = np.index();
        b = pb_.bl[p];
        cx = pb_.C.col(p).dot(x);
        cz = pb_.C.col(p).dot(z);
        break;
      case ActivationStatus::UPPER:
        p = np.index();
        b = pb_.bu[p];
        cx = pb_.C.col(p).dot(x);
        cz = pb_.C.col(p).dot(z);
        break;
      case ActivationStatus::LOWER_BOUND:
        pb = np.bndIndex();
        b = pb_.xl[pb];
        cx = x[pb];
        cz = z[pb];
        break;
      case ActivationStatus::UPPER_BOUND:
        pb = np.bndIndex();
        b = pb_.xu[pb];
        cx = x[pb];
        cz = z[pb];
        break;
      case ActivationStatus::EQUALITY:
      case ActivationStatus::FIXED:
      default:
#ifdef __GNUC__
        __builtin_unreachable();
#elif defined _MSC_VER
        __assume(false);
#else
        assert(false);
#endif
    }
    t2 = (b - cx) / cz;
  }

  return {t1, t2, l};
}

bool GoldfarbIdnaniSolver::addConstraint_(const internal::SelectedConstraint & /*sc*/)
{
  int q = A_.nbActiveCstr(); // This already counts the new constraint
  auto d = work_d_.asVector(nbVar_);
  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  for(int i = nbVar_ - 2; i >= q - 1; --i) //[OPTIM] use Householder transformation instead
  {
    Givens Qi;
    Qi.makeGivens(d[i], d[i + 1], &d[i]);
    JRLQP_DEBUG_ONLY(d[i + 1] = 0);
    J.applyOnTheRight(i, i + 1, Qi);
  }
  auto R = work_R_.asMatrix(q, q, nbVar_);
  R.rightCols<1>() = d.head(q);

  return true; //[NUMERIC]: add test on dependency
}

bool GoldfarbIdnaniSolver::removeConstraint_(int l)
{
  int q = A_.nbActiveCstr(); // This already counts that the constraint was removed
  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  auto R = work_R_.asMatrix(q + 1, q + 1, nbVar_);

  for(int i = l; i < q; ++i)
  {
    Givens Qi;
    R.col(i).head(i) = R.col(i + 1).head(i);
    Qi.makeGivens(R(i, i + 1), R(i + 1, i + 1), &R(i, i));
    JRLQP_DEBUG_ONLY(R(i + 1, i + 1) = 0);
    R.rightCols(q - i - 1).applyOnTheLeft(i, i + 1, Qi.transpose());
    J.applyOnTheRight(i, i + 1, Qi);
  }

  return true;
}

double GoldfarbIdnaniSolver::dot_(const internal::SelectedConstraint & sc, const VectorConstRef & z)
{
  internal::ConstraintNormal np(pb_.C, sc);
  return np.dot(z);
}

void GoldfarbIdnaniSolver::resize_(int nbVar, int /*nbCstr*/, bool /*useBounds*/)
{
  if(nbVar != nbVar_)
  {
    work_d_.resize(nbVar);
    work_J_.resize(nbVar, nbVar);
    work_R_.resize(nbVar, nbVar);
    work_tmp_.resize(nbVar);
    work_hCoeffs_.resize(nbVar);
    work_bact_.resize(nbVar);
  }
}

internal::TerminationType GoldfarbIdnaniSolver::processInitialActiveSet()
{
  A_.reset();

  // We look first for the bounds then the general constraints and proceed as follows:
  //  - if the problem data indicates an equality, we add the constraint as such, irrespective
  //    of any warm start data
  //  - if not, and warm start is on, we look at the warm start data. If it indicates an equality,
  //    we ignore it, because it we saw it can't be one.
  for(int i = 0; i < A_.nbBnd(); ++i)
  {
    int bi = A_.nbCstr() + i;
    if(pb_.xl[i] == pb_.xu[i])
    {
      A_.activate(bi, ActivationStatus::FIXED);
    }
    else if(!pb_.as.empty() && options_.warmStart_ && pb_.as[bi] != ActivationStatus::INACTIVE)
    {
      auto s = pb_.as[bi];
      assert(s > ActivationStatus::EQUALITY);
      if(s == ActivationStatus::FIXED)
      {
        JRLQP_LOG_COMMENT(log_, LogFlags::ACTIVE_SET, "Ignoring activation status for bound ", i,
                          " (bounds not equal)");
      }
      else
      {
        if((s == ActivationStatus::LOWER_BOUND && pb_.xl[i] < -options_.bigBnd_)
           || (s == ActivationStatus::UPPER_BOUND && pb_.xu[i] > +options_.bigBnd_))
          JRLQP_LOG_COMMENT(log_, LogFlags::ACTIVE_SET, "Ignoring activation status for bound ", i,
                            " (infinite bound)");
        else
          A_.activate(bi, s);
      }
    }
  }
  for(int i = 0; i < A_.nbCstr(); ++i)
  {
    if(pb_.bl[i] == pb_.bu[i])
    {
      A_.activate(i, ActivationStatus::EQUALITY);
    }
    else if(!pb_.as.empty() && options_.warmStart_ && pb_.as[i] != ActivationStatus::INACTIVE)
    {
      auto s = pb_.as[i];
      assert(s <= ActivationStatus::EQUALITY);
      if(s == ActivationStatus::FIXED)
      {
        JRLQP_LOG_COMMENT(log_, LogFlags::ACTIVE_SET, "Ignoring activation status for constraint ", i);
      }
      else
      {
        if((s == ActivationStatus::LOWER && pb_.bl[i] < -options_.bigBnd_)
           || (s == ActivationStatus::UPPER && pb_.bu[i] > +options_.bigBnd_))
          JRLQP_LOG_COMMENT(log_, LogFlags::ACTIVE_SET, "Ignoring activation status for constraint ", i,
                            " (infinite bound)");
        else
          A_.activate(i, s);
      }
    }
  }

  // We check that not too many constraints were activated
  if(A_.nbActiveCstr() > nbVar_)
  {
    if(A_.nbActiveEquality() + A_.nbFixedVariable() > nbVar_) return TerminationStatus::OVERCONSTRAINED_PROBLEM;

    // Lambda testing if the l-th activated constraint is an equality constraint or fixed variable.
    auto isEqualityOrFixed = [this](int i) {
      auto a = A_.activationStatus(A_[i]);
      return a == ActivationStatus::EQUALITY || a == ActivationStatus::FIXED;
    };
    // Work backward to deactivate inequality constraints until the number of constraints is nbVar_
    int i = A_.nbActiveCstr();
    while(A_.nbActiveCstr() > nbVar_)
    {
      --i;
      while(isEqualityOrFixed(i)) --i;
      A_.deactivate(i);
    }
  }

  return TerminationStatus::SUCCESS;
}

internal::TerminationType GoldfarbIdnaniSolver::initializeComputationData()
{
  auto L = pb_.G.template triangularView<Eigen::Lower>();

  int q = A_.nbActiveCstr();
  auto N = work_R_.asMatrix(nbVar_, q, nbVar_);
  auto b_act = work_bact_.asVector(q);
  for(int i = 0; i < q; ++i)
  {
    int cstrIdx = A_[i];
    switch(A_.activationStatus(cstrIdx))
    {
      case ActivationStatus::LOWER: // fallthrough
      case ActivationStatus::EQUALITY:
        N.col(i) = pb_.C.col(cstrIdx);
        b_act[i] = pb_.bl(cstrIdx);
        break;
      case ActivationStatus::UPPER:
        N.col(i) = -pb_.C.col(cstrIdx);
        b_act[i] = -pb_.bu(cstrIdx);
        break;
      case ActivationStatus::LOWER_BOUND: // fallthrough
      case ActivationStatus::FIXED:
        N.col(i).setZero();
        N.col(i)[cstrIdx - A_.nbCstr()] = 1;
        b_act[i] = pb_.xl(cstrIdx - A_.nbCstr());
        break;
      case ActivationStatus::UPPER_BOUND:
        N.col(i).setZero();
        N.col(i)[cstrIdx - A_.nbCstr()] = -1;
        b_act[i] = -pb_.xu(cstrIdx - A_.nbCstr());
        break;
      default:
        break;
    }
  }

  // J = L^-t
  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  J.setIdentity();
  L.transpose().solveInPlace(J);

  // B = L^-1 N
  L.solveInPlace(N); // [OPTIM] There are other possible ways to do this:
                     //  - use solveInPlace while filling N
                     //  - multiply N by J^T = L^-t (this would require that multiplication by triangular matrix can be
                     //  done in place with Eigen)

  // QR in place
  Eigen::Map<Eigen::VectorXd> hCoeffs(work_hCoeffs_.asVector(q).data(),
                                      q); // Should be simply hCoeffs = work_hCoeffs_.asVector(q), but his does compile
                                          // with householder_qr_inplace_blocked
  WVector tmp = work_tmp_.asVector(nbVar_);
  bool b = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::internal::householder_qr_inplace_blocked<decltype(N), decltype(hCoeffs)>::run(N, hCoeffs, 48, tmp.data());

  // J = J*Q
  Eigen::HouseholderSequence Q(N, hCoeffs);
  Q.applyThisOnTheRight(J, tmp);
  Eigen::internal::set_is_malloc_allowed(b);

  // Set lower part of R to 0
  JRLQP_DEBUG_ONLY(for(int i = 0; i < q; ++i) N.col(i).tail(nbVar_ - i - 1).setZero(););

  JRLQP_LOG(log_, LogFlags::INIT | LogFlags::NO_ITER, N, J, b_act);

  return TerminationStatus::SUCCESS;
}

internal::TerminationType GoldfarbIdnaniSolver::initializePrimalDualPoints()
{
  int q = A_.nbActiveCstr();
  auto J = work_J_.asMatrix(nbVar_, nbVar_, nbVar_);
  auto R = work_R_.asMatrix(q, q, nbVar_).template triangularView<Eigen::Upper>();
  WVector b_act = work_bact_.asVector(q);
  WVector alpha = work_tmp_.asVector(nbVar_);
  WVector beta = work_r_.asVector(q);
  WVector x = work_x_.asVector(nbVar_);
  WVector u = work_u_.asVector(q);
  auto alpha1 = alpha.head(q);
  auto alpha2 = alpha.tail(nbVar_ - q);

  alpha.noalias() = J.transpose() * pb_.a;
  beta = R.transpose().solve(b_act);
  x.noalias() = J.leftCols(q) * beta - J.rightCols(nbVar_ - q) * alpha2;
  u = alpha1 + beta;
  R.solveInPlace(u);

  f_ = beta.dot(0.5 * beta + alpha1) - 0.5 * alpha2.squaredNorm();

  JRLQP_LOG(log_, LogFlags::INIT | LogFlags::NO_ITER, alpha, beta, x, u, f_);

  return TerminationStatus::SUCCESS;
}

} // namespace jrl::qp::experimental
