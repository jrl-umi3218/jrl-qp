/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/internal/ActiveSet.h>
#include <jrl-qp/internal/ConstraintNormal.h>
#include <jrl-qp/utils/Debug.h>
#include <jrl-qp/utils/Logger.h>


namespace jrlqp
{
  class JRLQP_DLLAPI DualSolver
  {
  public:
    DualSolver();
    DualSolver(int nbVar, int nbCstr, bool useBounds);

    void resize(int nbVar, int nbCstr, bool useBounds);

    struct Options
    {
      int maxIter_;
      double bigBnd_;
    };

  protected:
    TerminationStatus solve();

    void init();
    internal::ConstraintNormal selectViolatedConstraint(const Eigen::VectorXd& x) const;
    void computeStep(Eigen::VectorXd& z, EigenHead& r, const internal::ConstraintNormal& np) const;
    std::tuple<double, double, int>
      computeStepLength(const internal::ConstraintNormal& np, const Eigen::VectorXd& x, const EigenHead& u,
        const Eigen::VectorXd& z, const EigenHead& r) const;
    bool addConstraint(const internal::ConstraintNormal& np);
    bool removeConstraint(int l, EigenHead& u);


    virtual void init_() = 0;
    virtual internal::ConstraintNormal selectViolatedConstraint_(const Eigen::VectorXd& x) const = 0;
    virtual void computeStep_(Eigen::VectorXd& z, EigenHead& r, const internal::ConstraintNormal& np) const = 0;
    virtual std::tuple<double, double, int>
      computeStepLength_(const internal::ConstraintNormal& np, const Eigen::VectorXd& x, const EigenHead& u,
        const Eigen::VectorXd& z, const EigenHead& r) const = 0;
    virtual bool addConstraint_(const internal::ConstraintNormal& np) = 0;
    virtual bool removeConstraint_(int l) = 0;

    Options options_;
    utils::Logger log_;

    int nbVar_;
    internal::ActiveSet A_;

    double f_;
    Eigen::VectorXd x_;
    Eigen::VectorXd z_;
    Eigen::VectorXd work_u_;
    Eigen::VectorXd work_r_;
  };

}